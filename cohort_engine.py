"""
Cohort Analysis Engine
======================
Comprehensive cohort analysis with retention, LTV curves, and segment export.

Features:
- Multiple cohort definitions (first order, platform, AOV tier, etc.)
- Retention heatmaps
- Cumulative LTV curves
- Time to Nth order analysis
- Custom segment builder for user exports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import json

logger = logging.getLogger(__name__)


class CohortEngine:
    """Build and analyze user cohorts."""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def build_cohorts(self, df: pd.DataFrame,
                      cohort_type: str = 'first_order_week') -> pd.DataFrame:
        """
        Assign users to cohorts based on specified criteria.

        Args:
            df: Event data
            cohort_type: How to define cohorts
                - 'first_order_week': Week of first order
                - 'first_order_month': Month of first order
                - 'first_visit_week': Week of first homepage view
                - 'platform': iOS vs Android
                - 'first_aov_tier': AOV tier of first order
                - 'first_order_type': delivery/scheduled/takeaway

        Returns:
            DataFrame with user-level cohort assignments
        """
        logger.info(f"Building cohorts by: {cohort_type}")

        if cohort_type == 'first_order_week':
            return self._cohort_by_first_order_week(df)
        elif cohort_type == 'first_order_month':
            return self._cohort_by_first_order_month(df)
        elif cohort_type == 'first_visit_week':
            return self._cohort_by_first_visit_week(df)
        elif cohort_type == 'platform':
            return self._cohort_by_platform(df)
        elif cohort_type == 'first_aov_tier':
            return self._cohort_by_first_aov(df)
        elif cohort_type == 'first_order_type':
            return self._cohort_by_first_order_type(df)
        else:
            raise ValueError(f"Unknown cohort type: {cohort_type}")

    def _cohort_by_first_order_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cohort by week of first order."""
        orders = df[df['event_type'] == 'checkout_completed'].copy()
        if len(orders) == 0:
            return pd.DataFrame()

        # Ensure timezone-naive
        if orders['event_time'].dt.tz is not None:
            orders['event_time'] = orders['event_time'].dt.tz_localize(None)

        first_order = orders.groupby('amplitude_id')['event_time'].min().reset_index()
        first_order.columns = ['amplitude_id', 'first_order_time']
        first_order['cohort'] = first_order['first_order_time'].dt.to_period('W').astype(str)
        first_order['cohort_date'] = first_order['first_order_time'].dt.to_period('W').dt.start_time

        return first_order

    def _cohort_by_first_order_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cohort by month of first order."""
        orders = df[df['event_type'] == 'checkout_completed'].copy()
        if len(orders) == 0:
            return pd.DataFrame()

        first_order = orders.groupby('amplitude_id')['event_time'].min().reset_index()
        first_order.columns = ['amplitude_id', 'first_order_time']
        first_order['cohort'] = first_order['first_order_time'].dt.to_period('M').astype(str)
        first_order['cohort_date'] = first_order['first_order_time'].dt.to_period('M').dt.start_time

        return first_order

    def _cohort_by_first_visit_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cohort by week of first homepage view."""
        visits = df[df['event_type'] == 'homepage_viewed'].copy()
        if len(visits) == 0:
            return pd.DataFrame()

        first_visit = visits.groupby('amplitude_id')['event_time'].min().reset_index()
        first_visit.columns = ['amplitude_id', 'first_visit_time']
        first_visit['cohort'] = first_visit['first_visit_time'].dt.to_period('W').astype(str)
        first_visit['cohort_date'] = first_visit['first_visit_time'].dt.to_period('W').dt.start_time

        return first_visit

    def _cohort_by_platform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cohort by primary platform."""
        # Get first platform used
        first_platform = df.sort_values('event_time').groupby('amplitude_id')['platform'].first().reset_index()
        first_platform.columns = ['amplitude_id', 'cohort']
        first_platform['cohort'] = first_platform['cohort'].str.upper()

        # Get first activity time
        first_time = df.groupby('amplitude_id')['event_time'].min().reset_index()
        first_time.columns = ['amplitude_id', 'first_order_time']

        return first_platform.merge(first_time, on='amplitude_id')

    def _cohort_by_first_aov(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cohort by AOV tier of first order."""
        orders = df[df['event_type'] == 'checkout_completed'].copy()
        if len(orders) == 0:
            return pd.DataFrame()

        # Parse order total
        orders = self._extract_order_total(orders)

        # Get first order per user
        first_orders = orders.sort_values('event_time').groupby('amplitude_id').first().reset_index()

        # Define tiers
        def get_tier(aov):
            if pd.isna(aov) or aov <= 0:
                return 'Unknown'
            elif aov < 50:
                return 'Low (<50)'
            elif aov < 100:
                return 'Medium (50-100)'
            else:
                return 'High (>100)'

        first_orders['cohort'] = first_orders['order_total'].apply(get_tier)
        first_orders['first_order_time'] = first_orders['event_time']

        return first_orders[['amplitude_id', 'cohort', 'first_order_time']]

    def _cohort_by_first_order_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cohort by type of first order."""
        orders = df[df['event_type'] == 'checkout_completed'].copy()
        if len(orders) == 0:
            return pd.DataFrame()

        orders = self._extract_order_type(orders)

        first_orders = orders.sort_values('event_time').groupby('amplitude_id').first().reset_index()
        first_orders['cohort'] = first_orders['order_type'].fillna('unknown').str.title()
        first_orders['first_order_time'] = first_orders['event_time']

        return first_orders[['amplitude_id', 'cohort', 'first_order_time']]

    def _extract_order_total(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract order_total from event properties."""
        if 'order_total' in df.columns:
            return df

        df = df.copy()

        if 'event_properties' in df.columns:
            def get_total(props):
                if pd.isna(props):
                    return 0
                try:
                    if isinstance(props, str):
                        props = json.loads(props)
                    return float(props.get('order_total', 0) or 0)
                except:
                    return 0

            df['order_total'] = df['event_properties'].apply(get_total)
        else:
            df['order_total'] = 0

        return df

    def _extract_order_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract order_type from event properties."""
        if 'order_type' in df.columns:
            return df

        df = df.copy()

        if 'event_properties' in df.columns:
            def get_type(props):
                if pd.isna(props):
                    return 'unknown'
                try:
                    if isinstance(props, str):
                        props = json.loads(props)
                    return props.get('order_type', 'unknown') or 'unknown'
                except:
                    return 'unknown'

            df['order_type'] = df['event_properties'].apply(get_type)
        else:
            df['order_type'] = 'unknown'

        return df

    def calculate_retention(self, df: pd.DataFrame,
                           cohorts: pd.DataFrame,
                           period: str = 'week',
                           max_periods: int = 12) -> pd.DataFrame:
        """
        Calculate retention rates for each cohort over time.

        Args:
            df: Event data
            cohorts: Cohort assignments (from build_cohorts)
            period: 'week' or 'month'
            max_periods: Maximum periods to calculate

        Returns:
            DataFrame with retention heatmap data
        """
        logger.info(f"Calculating {period}ly retention...")

        # Get order events
        orders = df[df['event_type'] == 'checkout_completed'][['amplitude_id', 'event_time']].copy()

        if len(orders) == 0:
            return pd.DataFrame()

        # Ensure timezone-naive
        if orders['event_time'].dt.tz is not None:
            orders['event_time'] = orders['event_time'].dt.tz_localize(None)

        # Merge with cohorts
        orders = orders.merge(cohorts[['amplitude_id', 'cohort', 'cohort_date']], on='amplitude_id', how='inner')

        if 'cohort_date' not in orders.columns:
            # Infer cohort_date from cohort string
            orders['cohort_date'] = pd.to_datetime(orders['cohort'].str.split('/').str[0])

        # Calculate period number
        if period == 'week':
            orders['period_num'] = ((orders['event_time'] - orders['cohort_date']).dt.days // 7).clip(lower=0)
        else:
            orders['period_num'] = ((orders['event_time'] - orders['cohort_date']).dt.days // 30).clip(lower=0)

        # Get cohort sizes
        cohort_sizes = cohorts.groupby('cohort')['amplitude_id'].nunique()

        # Calculate retention
        retention_data = []

        for cohort in cohorts['cohort'].unique():
            cohort_orders = orders[orders['cohort'] == cohort]
            cohort_size = cohort_sizes.get(cohort, 0)

            if cohort_size == 0:
                continue

            for period_num in range(max_periods + 1):
                active_users = cohort_orders[cohort_orders['period_num'] == period_num]['amplitude_id'].nunique()
                retention_rate = active_users / cohort_size * 100

                retention_data.append({
                    'cohort': cohort,
                    'period': period_num,
                    'active_users': active_users,
                    'cohort_size': cohort_size,
                    'retention_rate': retention_rate
                })

        retention_df = pd.DataFrame(retention_data)

        # Pivot for heatmap
        if len(retention_df) > 0:
            heatmap = retention_df.pivot(index='cohort', columns='period', values='retention_rate')
            return heatmap

        return pd.DataFrame()

    def calculate_cumulative_ltv(self, df: pd.DataFrame,
                                  cohorts: pd.DataFrame,
                                  max_periods: int = 12) -> pd.DataFrame:
        """
        Calculate cumulative LTV curves by cohort.

        Returns:
            DataFrame with cumulative revenue per user over time
        """
        logger.info("Calculating cumulative LTV curves...")

        orders = df[df['event_type'] == 'checkout_completed'].copy()
        orders = self._extract_order_total(orders)

        # Ensure timezone-naive
        if orders['event_time'].dt.tz is not None:
            orders['event_time'] = orders['event_time'].dt.tz_localize(None)

        # Merge with cohorts
        orders = orders.merge(cohorts[['amplitude_id', 'cohort', 'cohort_date']], on='amplitude_id', how='inner')

        if 'cohort_date' not in orders.columns:
            orders['cohort_date'] = pd.to_datetime(orders['cohort'].str.split('/').str[0])

        # Calculate week number
        orders['week_num'] = ((orders['event_time'] - orders['cohort_date']).dt.days // 7).clip(lower=0)

        # Get cohort sizes
        cohort_sizes = cohorts.groupby('cohort')['amplitude_id'].nunique()

        # Calculate cumulative LTV
        ltv_data = []

        for cohort in cohorts['cohort'].unique():
            cohort_orders = orders[orders['cohort'] == cohort].copy()
            cohort_size = cohort_sizes.get(cohort, 1)

            cumulative_revenue = 0
            for week in range(max_periods + 1):
                week_revenue = cohort_orders[cohort_orders['week_num'] == week]['order_total'].sum()
                cumulative_revenue += week_revenue

                ltv_data.append({
                    'cohort': cohort,
                    'week': week,
                    'cumulative_revenue': cumulative_revenue,
                    'cumulative_ltv': cumulative_revenue / cohort_size
                })

        return pd.DataFrame(ltv_data)

    def calculate_time_to_nth_order(self, df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        """
        Calculate distribution of days to Nth order.

        Args:
            df: Event data
            n: Which order to measure (2 = second order)

        Returns:
            DataFrame with time to Nth order per user
        """
        logger.info(f"Calculating time to order #{n}...")

        orders = df[df['event_type'] == 'checkout_completed'][['amplitude_id', 'event_time']].copy()
        orders = orders.sort_values(['amplitude_id', 'event_time'])

        # Number each order per user
        orders['order_num'] = orders.groupby('amplitude_id').cumcount() + 1

        # Get first order time
        first_orders = orders[orders['order_num'] == 1][['amplitude_id', 'event_time']].copy()
        first_orders.columns = ['amplitude_id', 'first_order_time']

        # Get Nth order time
        nth_orders = orders[orders['order_num'] == n][['amplitude_id', 'event_time']].copy()
        nth_orders.columns = ['amplitude_id', 'nth_order_time']

        # Merge
        result = first_orders.merge(nth_orders, on='amplitude_id', how='inner')
        result['days_to_nth_order'] = (result['nth_order_time'] - result['first_order_time']).dt.total_seconds() / 86400

        logger.info(f"Users with {n}+ orders: {len(result)}")
        logger.info(f"Median days to order #{n}: {result['days_to_nth_order'].median():.1f}")

        return result

    def calculate_order_frequency(self, df: pd.DataFrame,
                                   cohorts: pd.DataFrame) -> pd.DataFrame:
        """Calculate orders per active user per period by cohort."""
        orders = df[df['event_type'] == 'checkout_completed'][['amplitude_id', 'event_time']].copy()
        orders = orders.merge(cohorts[['amplitude_id', 'cohort', 'cohort_date']], on='amplitude_id', how='inner')

        if 'cohort_date' not in orders.columns:
            orders['cohort_date'] = pd.to_datetime(orders['cohort'].str.split('/').str[0])

        orders['week_num'] = ((orders['event_time'] - orders['cohort_date']).dt.days // 7).clip(lower=0)

        # Calculate frequency per cohort per week
        freq_data = orders.groupby(['cohort', 'week_num']).agg({
            'amplitude_id': ['nunique', 'count']
        }).reset_index()
        freq_data.columns = ['cohort', 'week', 'active_users', 'orders']
        freq_data['orders_per_user'] = freq_data['orders'] / freq_data['active_users']

        return freq_data

    def get_cohort_summary(self, df: pd.DataFrame,
                           cohorts: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics per cohort.

        Returns:
            DataFrame with cohort size, total orders, revenue, avg LTV, etc.
        """
        orders = df[df['event_type'] == 'checkout_completed'].copy()
        orders = self._extract_order_total(orders)
        orders = orders.merge(cohorts[['amplitude_id', 'cohort']], on='amplitude_id', how='inner')

        # Calculate metrics per cohort
        summary = orders.groupby('cohort').agg({
            'amplitude_id': 'nunique',
            'event_time': 'count',
            'order_total': ['sum', 'mean']
        }).reset_index()

        summary.columns = ['cohort', 'users', 'orders', 'total_revenue', 'avg_order_value']
        summary['orders_per_user'] = summary['orders'] / summary['users']
        summary['ltv'] = summary['total_revenue'] / summary['users']

        # Sort by cohort
        summary = summary.sort_values('cohort')

        return summary


class SegmentBuilder:
    """
    Build custom user segments based on behavioral filters.

    Examples:
        - Users who added to cart but didn't checkout
        - Users who ordered once but never again
        - Users who ordered food but never tried grocery
        - High LTV users at risk of churning
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.users = set(df['amplitude_id'].unique())
        self.filters_applied = []

    def did_event(self, event_type: str) -> 'SegmentBuilder':
        """Filter to users who did a specific event."""
        users_with_event = set(self.df[self.df['event_type'] == event_type]['amplitude_id'].unique())
        self.users = self.users & users_with_event
        self.filters_applied.append(f"did '{event_type}'")
        return self

    def did_not_event(self, event_type: str) -> 'SegmentBuilder':
        """Filter to users who did NOT do a specific event."""
        users_with_event = set(self.df[self.df['event_type'] == event_type]['amplitude_id'].unique())
        self.users = self.users - users_with_event
        self.filters_applied.append(f"did NOT do '{event_type}'")
        return self

    def event_count(self, event_type: str, operator: str, value: int) -> 'SegmentBuilder':
        """Filter by number of times user did an event.

        Args:
            event_type: Event to count
            operator: 'eq', 'gt', 'gte', 'lt', 'lte'
            value: Comparison value
        """
        counts = self.df[self.df['event_type'] == event_type].groupby('amplitude_id').size()

        if operator == 'eq':
            matching = counts[counts == value].index
        elif operator == 'gt':
            matching = counts[counts > value].index
        elif operator == 'gte':
            matching = counts[counts >= value].index
        elif operator == 'lt':
            matching = counts[counts < value].index
        elif operator == 'lte':
            matching = counts[counts <= value].index
        else:
            raise ValueError(f"Unknown operator: {operator}")

        self.users = self.users & set(matching)
        self.filters_applied.append(f"'{event_type}' count {operator} {value}")
        return self

    def ordered_exactly(self, n: int) -> 'SegmentBuilder':
        """Filter to users with exactly N orders."""
        order_counts = self.df[self.df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()
        matching = order_counts[order_counts == n].index
        self.users = self.users & set(matching)
        self.filters_applied.append(f"ordered exactly {n} times")
        return self

    def ordered_at_least(self, n: int) -> 'SegmentBuilder':
        """Filter to users with at least N orders."""
        order_counts = self.df[self.df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()
        matching = order_counts[order_counts >= n].index
        self.users = self.users & set(matching)
        self.filters_applied.append(f"ordered at least {n} times")
        return self

    def ordered_less_than(self, n: int) -> 'SegmentBuilder':
        """Filter to users with less than N orders."""
        order_counts = self.df[self.df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()
        # Include users with 0 orders
        all_users = set(self.df['amplitude_id'].unique())
        users_with_orders = set(order_counts[order_counts >= n].index)
        matching = all_users - users_with_orders
        self.users = self.users & matching
        self.filters_applied.append(f"ordered less than {n} times")
        return self

    def never_ordered(self) -> 'SegmentBuilder':
        """Filter to users who never placed an order."""
        orderers = set(self.df[self.df['event_type'] == 'checkout_completed']['amplitude_id'].unique())
        self.users = self.users - orderers
        self.filters_applied.append("never ordered")
        return self

    def platform(self, platform: str) -> 'SegmentBuilder':
        """Filter by platform (ios/android)."""
        platform_users = set(self.df[self.df['platform'].str.lower() == platform.lower()]['amplitude_id'].unique())
        self.users = self.users & platform_users
        self.filters_applied.append(f"platform = {platform}")
        return self

    def active_in_last_n_days(self, n: int) -> 'SegmentBuilder':
        """Filter to users active in last N days."""
        max_date = self.df['event_time'].max()
        cutoff = max_date - timedelta(days=n)
        recent_users = set(self.df[self.df['event_time'] >= cutoff]['amplitude_id'].unique())
        self.users = self.users & recent_users
        self.filters_applied.append(f"active in last {n} days")
        return self

    def inactive_for_n_days(self, n: int) -> 'SegmentBuilder':
        """Filter to users inactive for at least N days."""
        max_date = self.df['event_time'].max()
        cutoff = max_date - timedelta(days=n)

        # Get last activity per user
        last_activity = self.df.groupby('amplitude_id')['event_time'].max()
        inactive_users = set(last_activity[last_activity < cutoff].index)
        self.users = self.users & inactive_users
        self.filters_applied.append(f"inactive for {n}+ days")
        return self

    def first_order_between(self, start_date: str, end_date: str) -> 'SegmentBuilder':
        """Filter by first order date range."""
        orders = self.df[self.df['event_type'] == 'checkout_completed']
        first_orders = orders.groupby('amplitude_id')['event_time'].min()

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        matching = set(first_orders[(first_orders >= start) & (first_orders <= end)].index)
        self.users = self.users & matching
        self.filters_applied.append(f"first order between {start_date} and {end_date}")
        return self

    def with_churn_risk(self, min_probability: float = 0.7) -> 'SegmentBuilder':
        """Filter by churn risk (requires churn predictions)."""
        try:
            churn_df = pd.read_csv('outputs/churn_predictions_ml.csv')
            if 'index' in churn_df.columns:
                churn_df = churn_df.rename(columns={'index': 'amplitude_id'})

            high_risk = set(churn_df[churn_df['churn_probability'] >= min_probability]['amplitude_id'])
            self.users = self.users & high_risk
            self.filters_applied.append(f"churn risk >= {min_probability:.0%}")
        except FileNotFoundError:
            logger.warning("Churn predictions not found. Run predict_churn first.")
        return self

    def with_ltv_tier(self, tier: str) -> 'SegmentBuilder':
        """Filter by LTV tier (requires LTV predictions)."""
        try:
            ltv_df = pd.read_csv('outputs/ltv_predictions.csv')
            matching = set(ltv_df[ltv_df['ltv_tier'] == tier]['amplitude_id'])
            self.users = self.users & matching
            self.filters_applied.append(f"LTV tier = {tier}")
        except FileNotFoundError:
            logger.warning("LTV predictions not found. Run predict_ltv first.")
        return self

    def count(self) -> int:
        """Get count of users in segment."""
        return len(self.users)

    def get_users(self) -> List[int]:
        """Get list of user IDs in segment."""
        return list(self.users)

    def export(self, path: str = None) -> pd.DataFrame:
        """Export segment to DataFrame or CSV."""
        result = pd.DataFrame({'amplitude_id': list(self.users)})
        result['segment_filters'] = ' AND '.join(self.filters_applied)

        if path:
            result.to_csv(path, index=False)
            logger.info(f"Exported {len(result)} users to {path}")

        return result

    def describe(self) -> Dict:
        """Get segment description."""
        return {
            'user_count': len(self.users),
            'filters': self.filters_applied,
            'filter_description': ' AND '.join(self.filters_applied)
        }

    def reset(self) -> 'SegmentBuilder':
        """Reset to all users."""
        self.users = set(self.df['amplitude_id'].unique())
        self.filters_applied = []
        return self


# Predefined segments for common use cases
class PredefinedSegments:
    """Common segment definitions."""

    @staticmethod
    def cart_abandoners(df: pd.DataFrame) -> SegmentBuilder:
        """Users who added to cart but didn't checkout."""
        return (SegmentBuilder(df)
                .did_event('product_added')
                .did_not_event('checkout_completed'))

    @staticmethod
    def one_time_buyers(df: pd.DataFrame) -> SegmentBuilder:
        """Users who ordered exactly once."""
        return SegmentBuilder(df).ordered_exactly(1)

    @staticmethod
    def repeat_buyers(df: pd.DataFrame) -> SegmentBuilder:
        """Users who ordered 2+ times."""
        return SegmentBuilder(df).ordered_at_least(2)

    @staticmethod
    def power_users(df: pd.DataFrame) -> SegmentBuilder:
        """Users who ordered 5+ times."""
        return SegmentBuilder(df).ordered_at_least(5)

    @staticmethod
    def browsers_not_buyers(df: pd.DataFrame) -> SegmentBuilder:
        """Users who viewed products but never ordered."""
        return (SegmentBuilder(df)
                .did_event('product_page_viewed')
                .never_ordered())

    @staticmethod
    def lapsed_customers(df: pd.DataFrame, days: int = 30) -> SegmentBuilder:
        """Users who ordered but haven't been active for N days."""
        return (SegmentBuilder(df)
                .did_event('checkout_completed')
                .inactive_for_n_days(days))

    @staticmethod
    def high_value_at_risk(df: pd.DataFrame) -> SegmentBuilder:
        """High LTV users with high churn risk."""
        return (SegmentBuilder(df)
                .with_ltv_tier('Diamond')
                .with_churn_risk(0.6))

    @staticmethod
    def engaged_non_converters(df: pd.DataFrame) -> SegmentBuilder:
        """Users with high engagement (10+ events) but no orders."""
        return (SegmentBuilder(df)
                .event_count('homepage_viewed', 'gte', 3)
                .never_ordered())

    @staticmethod
    def ios_power_users(df: pd.DataFrame) -> SegmentBuilder:
        """iOS users with 3+ orders."""
        return (SegmentBuilder(df)
                .platform('ios')
                .ordered_at_least(3))


def run_cohort_analysis(df: pd.DataFrame,
                        cohort_type: str = 'first_order_week',
                        output_dir: str = 'outputs') -> Dict:
    """
    Run full cohort analysis and save outputs.

    Args:
        df: Event data
        cohort_type: How to define cohorts
        output_dir: Where to save outputs

    Returns:
        Dictionary with all cohort analysis results
    """
    from pathlib import Path
    Path(output_dir).mkdir(exist_ok=True)

    engine = CohortEngine()

    # Build cohorts
    cohorts = engine.build_cohorts(df, cohort_type=cohort_type)
    logger.info(f"Built {cohorts['cohort'].nunique()} cohorts with {len(cohorts)} users")

    # Calculate retention
    retention = engine.calculate_retention(df, cohorts)

    # Calculate LTV curves
    ltv_curves = engine.calculate_cumulative_ltv(df, cohorts)

    # Time to 2nd order
    time_to_2nd = engine.calculate_time_to_nth_order(df, n=2)

    # Cohort summary
    summary = engine.get_cohort_summary(df, cohorts)

    # Save outputs
    cohorts.to_csv(f'{output_dir}/cohort_assignments.csv', index=False)
    retention.to_csv(f'{output_dir}/cohort_retention.csv')
    ltv_curves.to_csv(f'{output_dir}/cohort_ltv_curves.csv', index=False)
    time_to_2nd.to_csv(f'{output_dir}/time_to_2nd_order.csv', index=False)
    summary.to_csv(f'{output_dir}/cohort_summary.csv', index=False)

    logger.info(f"Cohort analysis outputs saved to {output_dir}/")

    return {
        'cohorts': cohorts,
        'retention': retention,
        'ltv_curves': ltv_curves,
        'time_to_2nd_order': time_to_2nd,
        'summary': summary
    }


if __name__ == '__main__':
    # Test the cohort engine
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '/Users/muhannadsaad/Desktop/investigation/dec_15/dec_15_25.parquet'

    from data_loader import DataLoader

    loader = DataLoader()
    df = loader.load(data_path)

    # Run cohort analysis
    results = run_cohort_analysis(df, cohort_type='first_order_week')

    print("\n=== COHORT SUMMARY ===")
    print(results['summary'].to_string())

    print("\n=== RETENTION HEATMAP ===")
    print(results['retention'].to_string())

    # Test segment builder
    print("\n=== SEGMENT BUILDER DEMO ===")

    # Cart abandoners
    segment = PredefinedSegments.cart_abandoners(df)
    print(f"Cart abandoners: {segment.count()} users")

    # One-time buyers
    segment = PredefinedSegments.one_time_buyers(df)
    print(f"One-time buyers: {segment.count()} users")

    # Custom segment
    segment = (SegmentBuilder(df)
               .did_event('product_added')
               .ordered_less_than(2)
               .platform('ios'))
    print(f"iOS users who added to cart but ordered < 2 times: {segment.count()} users")
    print(f"Filters: {segment.describe()['filter_description']}")
