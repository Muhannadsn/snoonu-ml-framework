"""Trending & Popularity Features.

Features that work well with limited historical data.
No ML required - based on aggregations and time patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class TrendingEngine:
    """Compute trending and popularity-based recommendations."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Event data with checkout_completed events
        """
        self.df = df
        self._extract_order_data()
        self._compute_base_stats()

    def _extract_order_data(self):
        """Extract and parse order data."""
        logger.info("Extracting order data for trending analysis...")

        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        # Handle different data formats
        if 'event_data_merchant_name' in checkouts.columns:
            checkouts['merchant'] = checkouts['event_data_merchant_name']
            checkouts['merchant_id'] = checkouts['event_data_merchant_id']
            checkouts['order_total'] = 0
        else:
            def parse_props(props):
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except:
                        return {}
                return props if isinstance(props, dict) else {}

            checkouts['props'] = checkouts['event_properties'].apply(parse_props)
            checkouts['merchant'] = checkouts['props'].apply(lambda x: x.get('merchant_name', 'Unknown'))
            checkouts['merchant_id'] = checkouts['props'].apply(lambda x: x.get('merchant_id'))
            checkouts['order_total'] = checkouts['props'].apply(lambda x: float(x.get('order_total', 0) or 0))
            checkouts['category'] = checkouts['props'].apply(lambda x: x.get('category_name', 'Unknown'))

        checkouts['hour'] = pd.to_datetime(checkouts['event_time']).dt.hour
        checkouts['dayofweek'] = pd.to_datetime(checkouts['event_time']).dt.dayofweek
        checkouts['date'] = pd.to_datetime(checkouts['event_time']).dt.date

        self.orders = checkouts[checkouts['merchant'].notna()].copy()
        logger.info(f"Processed {len(self.orders)} orders from {self.orders['merchant'].nunique()} merchants")

    def _compute_base_stats(self):
        """Pre-compute base statistics."""
        # Overall merchant stats
        self.merchant_stats = self.orders.groupby('merchant').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': 'sum'
        }).reset_index()
        self.merchant_stats.columns = ['merchant', 'order_count', 'unique_customers', 'total_revenue']
        self.merchant_stats['orders_per_customer'] = (
            self.merchant_stats['order_count'] / self.merchant_stats['unique_customers']
        )

    def popular_now(self, n: int = 20, min_orders: int = 5) -> pd.DataFrame:
        """Get currently popular merchants.

        Args:
            n: Number of merchants to return
            min_orders: Minimum orders to qualify

        Returns:
            DataFrame with popular merchants and stats
        """
        popular = self.merchant_stats[self.merchant_stats['order_count'] >= min_orders].copy()
        popular = popular.sort_values('order_count', ascending=False).head(n)
        popular['rank'] = range(1, len(popular) + 1)

        return popular[['rank', 'merchant', 'order_count', 'unique_customers', 'orders_per_customer']]

    def popular_by_time(self, time_slot: str = 'lunch', n: int = 10) -> pd.DataFrame:
        """Get popular merchants for specific time slots.

        Args:
            time_slot: 'breakfast', 'lunch', 'dinner', 'late_night'
            n: Number of merchants to return

        Returns:
            DataFrame with time-specific popular merchants
        """
        time_ranges = {
            'breakfast': (6, 11),
            'lunch': (11, 15),
            'afternoon': (15, 18),
            'dinner': (18, 22),
            'late_night': (22, 6)
        }

        if time_slot not in time_ranges:
            raise ValueError(f"Unknown time_slot. Choose from: {list(time_ranges.keys())}")

        start, end = time_ranges[time_slot]

        if start < end:
            time_orders = self.orders[(self.orders['hour'] >= start) & (self.orders['hour'] < end)]
        else:  # late_night wraps around midnight
            time_orders = self.orders[(self.orders['hour'] >= start) | (self.orders['hour'] < end)]

        if len(time_orders) == 0:
            return pd.DataFrame()

        popular = time_orders.groupby('merchant').agg({
            'amplitude_id': ['count', 'nunique']
        }).reset_index()
        popular.columns = ['merchant', 'order_count', 'unique_customers']
        popular = popular.sort_values('order_count', ascending=False).head(n)
        popular['time_slot'] = time_slot
        popular['rank'] = range(1, len(popular) + 1)

        return popular

    def popular_by_platform(self, platform: str = 'ios', n: int = 10) -> pd.DataFrame:
        """Get popular merchants by platform.

        Args:
            platform: 'ios' or 'android'
            n: Number of merchants to return

        Returns:
            DataFrame with platform-specific popular merchants
        """
        platform_orders = self.orders[self.orders['platform'].str.lower() == platform.lower()]

        if len(platform_orders) == 0:
            return pd.DataFrame()

        popular = platform_orders.groupby('merchant').agg({
            'amplitude_id': ['count', 'nunique']
        }).reset_index()
        popular.columns = ['merchant', 'order_count', 'unique_customers']
        popular = popular.sort_values('order_count', ascending=False).head(n)
        popular['platform'] = platform
        popular['rank'] = range(1, len(popular) + 1)

        return popular

    def trending_velocity(self, n: int = 10, min_orders: int = 3) -> pd.DataFrame:
        """Find merchants with highest order velocity (orders per hour).

        Good for finding "hot right now" merchants.

        Args:
            n: Number of merchants to return
            min_orders: Minimum orders to qualify

        Returns:
            DataFrame with trending merchants by velocity
        """
        # Calculate time span
        time_span_hours = (
            self.orders['event_time'].max() - self.orders['event_time'].min()
        ).total_seconds() / 3600

        if time_span_hours == 0:
            time_span_hours = 1

        merchant_velocity = self.orders.groupby('merchant').agg({
            'amplitude_id': 'count'
        }).reset_index()
        merchant_velocity.columns = ['merchant', 'order_count']
        merchant_velocity['orders_per_hour'] = merchant_velocity['order_count'] / time_span_hours

        trending = merchant_velocity[merchant_velocity['order_count'] >= min_orders]
        trending = trending.sort_values('orders_per_hour', ascending=False).head(n)
        trending['rank'] = range(1, len(trending) + 1)

        return trending[['rank', 'merchant', 'order_count', 'orders_per_hour']]

    def category_breakdown(self) -> pd.DataFrame:
        """Get order distribution by category.

        Returns:
            DataFrame with category stats
        """
        if 'category' not in self.orders.columns:
            return pd.DataFrame()

        category_stats = self.orders.groupby('category').agg({
            'amplitude_id': ['count', 'nunique'],
            'merchant': 'nunique'
        }).reset_index()
        category_stats.columns = ['category', 'order_count', 'unique_customers', 'merchant_count']
        category_stats = category_stats.sort_values('order_count', ascending=False)
        category_stats['pct_of_orders'] = (
            category_stats['order_count'] / category_stats['order_count'].sum() * 100
        ).round(1)

        return category_stats

    def new_customer_favorites(self, n: int = 10) -> pd.DataFrame:
        """Merchants popular with first-time orderers.

        Great for onboarding recommendations.

        Args:
            n: Number of merchants to return

        Returns:
            DataFrame with new customer favorites
        """
        # Find users with only 1 order (new customers)
        user_order_counts = self.orders.groupby('amplitude_id').size()
        new_users = user_order_counts[user_order_counts == 1].index

        new_user_orders = self.orders[self.orders['amplitude_id'].isin(new_users)]

        favorites = new_user_orders.groupby('merchant').size().reset_index(name='new_customer_orders')
        favorites = favorites.sort_values('new_customer_orders', ascending=False).head(n)
        favorites['rank'] = range(1, len(favorites) + 1)

        # Add overall stats for comparison
        favorites = favorites.merge(
            self.merchant_stats[['merchant', 'order_count']],
            on='merchant',
            how='left'
        )
        favorites['new_customer_pct'] = (
            favorites['new_customer_orders'] / favorites['order_count'] * 100
        ).round(1)

        return favorites[['rank', 'merchant', 'new_customer_orders', 'order_count', 'new_customer_pct']]

    def repeat_customer_favorites(self, n: int = 10) -> pd.DataFrame:
        """Merchants with highest repeat order rates.

        Indicates quality/satisfaction.

        Args:
            n: Number of merchants to return

        Returns:
            DataFrame with repeat customer favorites
        """
        # Find repeat customers per merchant
        merchant_user_orders = self.orders.groupby(['merchant', 'amplitude_id']).size().reset_index(name='orders')

        repeat_stats = merchant_user_orders.groupby('merchant').agg({
            'amplitude_id': 'count',  # Total customers
            'orders': lambda x: (x > 1).sum()  # Customers who ordered 2+
        }).reset_index()
        repeat_stats.columns = ['merchant', 'total_customers', 'repeat_customers']
        repeat_stats['repeat_rate'] = (
            repeat_stats['repeat_customers'] / repeat_stats['total_customers'] * 100
        ).round(1)

        # Filter to merchants with enough customers
        repeat_stats = repeat_stats[repeat_stats['total_customers'] >= 5]
        repeat_stats = repeat_stats.sort_values('repeat_rate', ascending=False).head(n)
        repeat_stats['rank'] = range(1, len(repeat_stats) + 1)

        return repeat_stats[['rank', 'merchant', 'total_customers', 'repeat_customers', 'repeat_rate']]

    def weekend_vs_weekday(self, n: int = 10) -> Dict[str, pd.DataFrame]:
        """Compare weekend vs weekday popularity.

        Returns:
            Dict with 'weekend' and 'weekday' DataFrames
        """
        weekend_orders = self.orders[self.orders['dayofweek'].isin([4, 5])]  # Fri, Sat
        weekday_orders = self.orders[~self.orders['dayofweek'].isin([4, 5])]

        def get_popular(orders, label):
            if len(orders) == 0:
                return pd.DataFrame()
            popular = orders.groupby('merchant').size().reset_index(name='order_count')
            popular = popular.sort_values('order_count', ascending=False).head(n)
            popular['period'] = label
            popular['rank'] = range(1, len(popular) + 1)
            return popular

        return {
            'weekend': get_popular(weekend_orders, 'weekend'),
            'weekday': get_popular(weekday_orders, 'weekday')
        }

    def get_recommendations_for_context(self,
                                        hour: Optional[int] = None,
                                        platform: Optional[str] = None,
                                        is_new_user: bool = False,
                                        n: int = 10) -> pd.DataFrame:
        """Get contextual recommendations based on current context.

        Args:
            hour: Current hour (0-23), defaults to now
            platform: User's platform ('ios', 'android')
            is_new_user: Whether user is new
            n: Number of recommendations

        Returns:
            DataFrame with contextual recommendations
        """
        if hour is None:
            hour = datetime.now().hour

        # Determine time slot
        if 6 <= hour < 11:
            time_slot = 'breakfast'
        elif 11 <= hour < 15:
            time_slot = 'lunch'
        elif 15 <= hour < 18:
            time_slot = 'afternoon'
        elif 18 <= hour < 22:
            time_slot = 'dinner'
        else:
            time_slot = 'late_night'

        # Start with time-based popularity
        recs = self.popular_by_time(time_slot, n=n*2)

        if len(recs) == 0:
            recs = self.popular_now(n=n*2)

        # Boost new customer favorites for new users
        if is_new_user and len(recs) > 0:
            new_faves = self.new_customer_favorites(n=n)
            if len(new_faves) > 0:
                # Merge and re-rank
                recs = recs.merge(
                    new_faves[['merchant', 'new_customer_orders']],
                    on='merchant',
                    how='left'
                )
                recs['new_customer_orders'] = recs['new_customer_orders'].fillna(0)
                recs['score'] = recs['order_count'] + recs['new_customer_orders'] * 0.5
                recs = recs.sort_values('score', ascending=False)

        # Filter by platform if specified
        if platform and 'platform' not in recs.columns:
            platform_pop = self.popular_by_platform(platform, n=n*2)
            if len(platform_pop) > 0:
                platform_merchants = set(platform_pop['merchant'])
                recs['platform_match'] = recs['merchant'].isin(platform_merchants)
                recs = recs.sort_values(['platform_match', 'order_count'], ascending=[False, False])

        recs = recs.head(n).reset_index(drop=True)
        recs['rank'] = range(1, len(recs) + 1)
        recs['context'] = f"{time_slot}" + (f" | {platform}" if platform else "") + (" | new_user" if is_new_user else "")

        return recs[['rank', 'merchant', 'order_count', 'context']]

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_orders': len(self.orders),
            'unique_merchants': self.orders['merchant'].nunique(),
            'unique_customers': self.orders['amplitude_id'].nunique(),
            'date_range': f"{self.orders['date'].min()} to {self.orders['date'].max()}",
            'avg_orders_per_merchant': len(self.orders) / self.orders['merchant'].nunique(),
            'avg_orders_per_customer': len(self.orders) / self.orders['amplitude_id'].nunique()
        }
