"""Merchant Intelligence Analytics.

Comprehensive merchant analysis including health scoring, performance trends,
delivery impact, and competitive positioning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class MerchantIntelligence:
    """Comprehensive merchant analytics and intelligence."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Event data DataFrame with checkout_completed events
        """
        self.df = df.copy()
        self._extract_order_data()
        self._compute_merchant_metrics()

    def _extract_order_data(self):
        """Extract order data from checkout events."""
        logger.info("Extracting merchant order data...")

        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        # Handle different data formats
        if 'event_data_merchant_name' in checkouts.columns:
            # Dec 9 format
            checkouts['merchant_name'] = checkouts['event_data_merchant_name']
            checkouts['merchant_id'] = checkouts['event_data_merchant_id']
            checkouts['order_total'] = 0
            checkouts['order_type'] = 'unknown'
            checkouts['delivery_fee'] = 0
            checkouts['products_quantity'] = 1
            checkouts['category'] = checkouts.get('event_data_category_name', 'Unknown')
        else:
            # Dec 15 format - parse from event_properties
            def parse_props(props):
                if isinstance(props, str):
                    try:
                        return json.loads(props)
                    except:
                        return {}
                return props if isinstance(props, dict) else {}

            checkouts['props'] = checkouts['event_properties'].apply(parse_props)
            checkouts['merchant_name'] = checkouts['props'].apply(lambda x: x.get('merchant_name', 'Unknown'))
            checkouts['merchant_id'] = checkouts['props'].apply(lambda x: x.get('merchant_id'))
            checkouts['order_total'] = checkouts['props'].apply(lambda x: float(x.get('order_total', 0) or 0))
            checkouts['order_type'] = checkouts['props'].apply(lambda x: x.get('order_type', 'unknown'))
            checkouts['delivery_fee'] = checkouts['props'].apply(lambda x: float(x.get('delivery_fee', 0) or 0))
            checkouts['products_quantity'] = checkouts['props'].apply(lambda x: int(x.get('products_quantity', 1) or 1))
            checkouts['category'] = checkouts['props'].apply(lambda x: x.get('category_name', 'Unknown'))

        checkouts['event_time'] = pd.to_datetime(checkouts['event_time'])
        checkouts['hour'] = checkouts['event_time'].dt.hour
        checkouts['dayofweek'] = checkouts['event_time'].dt.dayofweek
        checkouts['date'] = checkouts['event_time'].dt.date

        self.orders = checkouts[checkouts['merchant_name'].notna() & (checkouts['merchant_name'] != 'Unknown')].copy()
        logger.info(f"Extracted {len(self.orders):,} orders from {self.orders['merchant_name'].nunique():,} merchants")

    def _compute_merchant_metrics(self):
        """Compute core metrics for each merchant."""
        logger.info("Computing merchant metrics...")

        # Basic metrics
        self.merchant_metrics = self.orders.groupby('merchant_name').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': ['sum', 'mean'],
            'products_quantity': 'mean',
            'delivery_fee': 'mean',
            'date': ['min', 'max', 'nunique']
        }).reset_index()

        self.merchant_metrics.columns = [
            'merchant_name', 'order_count', 'unique_customers',
            'total_revenue', 'avg_order_value', 'avg_items_per_order',
            'avg_delivery_fee', 'first_order_date', 'last_order_date', 'active_days'
        ]

        # Orders per customer (retention indicator)
        self.merchant_metrics['orders_per_customer'] = (
            self.merchant_metrics['order_count'] / self.merchant_metrics['unique_customers']
        ).round(2)

        # Repeat customer rate
        repeat_stats = self.orders.groupby(['merchant_name', 'amplitude_id']).size().reset_index(name='orders')
        repeat_by_merchant = repeat_stats.groupby('merchant_name').apply(
            lambda x: (x['orders'] > 1).sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index(name='repeat_rate')
        self.merchant_metrics = self.merchant_metrics.merge(repeat_by_merchant, on='merchant_name', how='left')

        # Category
        category_mode = self.orders.groupby('merchant_name')['category'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        ).reset_index()
        category_mode.columns = ['merchant_name', 'primary_category']
        self.merchant_metrics = self.merchant_metrics.merge(category_mode, on='merchant_name', how='left')

        logger.info(f"Computed metrics for {len(self.merchant_metrics):,} merchants")

    def get_merchant_health_scores(self, top_n: int = 50) -> pd.DataFrame:
        """Calculate health scores for merchants.

        Health Score Components:
        - Volume Score (25%): Order count relative to category
        - Retention Score (25%): Repeat customer rate
        - Growth Score (25%): Recent trend
        - Value Score (25%): AOV relative to category
        """
        df = self.merchant_metrics.copy()

        # Need minimum orders for reliable scoring
        df = df[df['order_count'] >= 5].copy()

        if len(df) == 0:
            return pd.DataFrame()

        # Volume Score (percentile within category or overall)
        df['volume_percentile'] = df['order_count'].rank(pct=True) * 100

        # Retention Score (based on repeat rate)
        df['retention_percentile'] = df['repeat_rate'].rank(pct=True) * 100

        # Value Score (based on orders per customer)
        df['value_percentile'] = df['orders_per_customer'].rank(pct=True) * 100

        # Growth Score (based on recency - more recent last order = better)
        max_date = df['last_order_date'].max()
        df['days_since_last_order'] = (max_date - df['last_order_date']).apply(lambda x: x.days if hasattr(x, 'days') else 0)
        df['recency_percentile'] = (100 - df['days_since_last_order'].rank(pct=True) * 100)

        # Composite Health Score
        df['health_score'] = (
            df['volume_percentile'] * 0.25 +
            df['retention_percentile'] * 0.25 +
            df['value_percentile'] * 0.25 +
            df['recency_percentile'] * 0.25
        ).round(1)

        # Health tier
        df['health_tier'] = pd.cut(
            df['health_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['At Risk', 'Needs Attention', 'Healthy', 'Thriving']
        )

        result = df.nlargest(top_n, 'health_score')[[
            'merchant_name', 'primary_category', 'health_score', 'health_tier',
            'order_count', 'unique_customers', 'repeat_rate', 'orders_per_customer'
        ]]

        return result

    def get_at_risk_merchants(self, min_orders: int = 10) -> pd.DataFrame:
        """Identify merchants that may be declining or at risk."""
        df = self.merchant_metrics[self.merchant_metrics['order_count'] >= min_orders].copy()

        if len(df) == 0:
            return pd.DataFrame()

        # Risk indicators
        risk_factors = []

        for _, merchant in df.iterrows():
            factors = []
            risk_score = 0

            # Low repeat rate
            if merchant['repeat_rate'] < 10:
                factors.append('Very low repeat rate (<10%)')
                risk_score += 30

            # Low orders per customer
            if merchant['orders_per_customer'] < 1.1:
                factors.append('Almost no repeat orders')
                risk_score += 25

            # Check for declining trend (compare first half vs second half of period)
            merchant_orders = self.orders[self.orders['merchant_name'] == merchant['merchant_name']]
            if len(merchant_orders) > 10:
                mid_date = merchant_orders['event_time'].median()
                first_half = len(merchant_orders[merchant_orders['event_time'] <= mid_date])
                second_half = len(merchant_orders[merchant_orders['event_time'] > mid_date])

                if second_half < first_half * 0.7:
                    factors.append('Declining order volume (-30%+)')
                    risk_score += 30

            # High delivery fee relative to order value (if data available)
            if merchant['avg_delivery_fee'] > 0 and merchant['avg_order_value'] > 0:
                fee_pct = merchant['avg_delivery_fee'] / merchant['avg_order_value'] * 100
                if fee_pct > 20:
                    factors.append(f'High delivery fee ratio ({fee_pct:.0f}%)')
                    risk_score += 15

            if risk_score > 0:
                risk_factors.append({
                    'merchant_name': merchant['merchant_name'],
                    'category': merchant['primary_category'],
                    'risk_score': risk_score,
                    'risk_factors': '; '.join(factors),
                    'order_count': merchant['order_count'],
                    'repeat_rate': round(merchant['repeat_rate'], 1)
                })

        result = pd.DataFrame(risk_factors)
        if len(result) > 0:
            result = result.sort_values('risk_score', ascending=False)

        return result

    def get_merchant_comparison(self, merchant_name: str) -> Dict:
        """Compare a merchant against category and overall benchmarks."""
        merchant = self.merchant_metrics[self.merchant_metrics['merchant_name'] == merchant_name]

        if len(merchant) == 0:
            return {'error': 'Merchant not found'}

        merchant = merchant.iloc[0]
        category = merchant['primary_category']

        # Category benchmarks
        category_merchants = self.merchant_metrics[self.merchant_metrics['primary_category'] == category]

        # Overall benchmarks
        overall = self.merchant_metrics

        comparison = {
            'merchant_name': merchant_name,
            'category': category,
            'metrics': {
                'order_count': {
                    'value': int(merchant['order_count']),
                    'category_avg': round(category_merchants['order_count'].mean(), 1),
                    'category_rank': int((category_merchants['order_count'] <= merchant['order_count']).sum()),
                    'category_total': len(category_merchants),
                    'overall_percentile': round((overall['order_count'] <= merchant['order_count']).mean() * 100, 1)
                },
                'repeat_rate': {
                    'value': round(merchant['repeat_rate'], 1),
                    'category_avg': round(category_merchants['repeat_rate'].mean(), 1),
                    'overall_avg': round(overall['repeat_rate'].mean(), 1),
                    'status': 'Above Average' if merchant['repeat_rate'] > category_merchants['repeat_rate'].mean() else 'Below Average'
                },
                'orders_per_customer': {
                    'value': round(merchant['orders_per_customer'], 2),
                    'category_avg': round(category_merchants['orders_per_customer'].mean(), 2),
                    'overall_avg': round(overall['orders_per_customer'].mean(), 2)
                },
                'avg_order_value': {
                    'value': round(merchant['avg_order_value'], 2),
                    'category_avg': round(category_merchants['avg_order_value'].mean(), 2),
                    'overall_avg': round(overall['avg_order_value'].mean(), 2)
                }
            }
        }

        return comparison

    def get_category_performance(self) -> pd.DataFrame:
        """Get performance metrics by category."""
        category_stats = self.merchant_metrics.groupby('primary_category').agg({
            'merchant_name': 'count',
            'order_count': ['sum', 'mean'],
            'unique_customers': 'sum',
            'total_revenue': 'sum',
            'repeat_rate': 'mean',
            'orders_per_customer': 'mean'
        }).reset_index()

        category_stats.columns = [
            'category', 'merchant_count', 'total_orders', 'avg_orders_per_merchant',
            'total_customers', 'total_revenue', 'avg_repeat_rate', 'avg_orders_per_customer'
        ]

        category_stats['market_share'] = (
            category_stats['total_orders'] / category_stats['total_orders'].sum() * 100
        ).round(1)

        category_stats = category_stats.sort_values('total_orders', ascending=False)

        return category_stats

    def get_time_performance(self, merchant_name: Optional[str] = None) -> pd.DataFrame:
        """Get order performance by time slot."""
        if merchant_name:
            orders = self.orders[self.orders['merchant_name'] == merchant_name]
        else:
            orders = self.orders

        time_slots = {
            'breakfast': (6, 11),
            'lunch': (11, 15),
            'afternoon': (15, 18),
            'dinner': (18, 22),
            'late_night': (22, 6)
        }

        def get_slot(hour):
            for slot, (start, end) in time_slots.items():
                if start < end:
                    if start <= hour < end:
                        return slot
                else:
                    if hour >= start or hour < end:
                        return slot
            return 'other'

        orders = orders.copy()
        orders['time_slot'] = orders['hour'].apply(get_slot)

        time_stats = orders.groupby('time_slot').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': 'mean'
        }).reset_index()
        time_stats.columns = ['time_slot', 'orders', 'unique_customers', 'avg_order_value']

        # Order the slots
        slot_order = ['breakfast', 'lunch', 'afternoon', 'dinner', 'late_night']
        time_stats['slot_order'] = time_stats['time_slot'].apply(lambda x: slot_order.index(x) if x in slot_order else 99)
        time_stats = time_stats.sort_values('slot_order').drop('slot_order', axis=1)

        time_stats['pct_of_orders'] = (time_stats['orders'] / time_stats['orders'].sum() * 100).round(1)

        return time_stats

    def get_top_merchants_by_category(self, n_per_category: int = 5) -> pd.DataFrame:
        """Get top merchants in each category."""
        def get_top_n(group):
            return group.nlargest(n_per_category, 'order_count')

        top_merchants = self.merchant_metrics.groupby('primary_category').apply(get_top_n).reset_index(drop=True)

        return top_merchants[[
            'primary_category', 'merchant_name', 'order_count',
            'unique_customers', 'repeat_rate', 'orders_per_customer'
        ]]

    def get_new_merchants(self, days_threshold: int = 7) -> pd.DataFrame:
        """Identify new/emerging merchants."""
        if len(self.orders) == 0:
            return pd.DataFrame()

        max_date = self.orders['date'].max()
        threshold_date = max_date - timedelta(days=days_threshold)

        # Merchants whose first order is recent
        new_merchants = self.merchant_metrics[
            self.merchant_metrics['first_order_date'] >= threshold_date
        ].copy()

        if len(new_merchants) == 0:
            return pd.DataFrame()

        new_merchants = new_merchants.sort_values('order_count', ascending=False)

        return new_merchants[[
            'merchant_name', 'primary_category', 'first_order_date',
            'order_count', 'unique_customers', 'orders_per_customer'
        ]]

    def get_merchant_customer_overlap(self, merchant_a: str, merchant_b: str) -> Dict:
        """Analyze customer overlap between two merchants."""
        customers_a = set(self.orders[self.orders['merchant_name'] == merchant_a]['amplitude_id'])
        customers_b = set(self.orders[self.orders['merchant_name'] == merchant_b]['amplitude_id'])

        overlap = customers_a & customers_b
        only_a = customers_a - customers_b
        only_b = customers_b - customers_a

        return {
            'merchant_a': merchant_a,
            'merchant_b': merchant_b,
            'customers_a': len(customers_a),
            'customers_b': len(customers_b),
            'overlap': len(overlap),
            'overlap_pct_of_a': round(len(overlap) / len(customers_a) * 100, 1) if customers_a else 0,
            'overlap_pct_of_b': round(len(overlap) / len(customers_b) * 100, 1) if customers_b else 0,
            'only_a': len(only_a),
            'only_b': len(only_b)
        }

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_merchants': len(self.merchant_metrics),
            'total_orders': int(self.merchant_metrics['order_count'].sum()),
            'total_customers': int(self.orders['amplitude_id'].nunique()),
            'total_revenue': float(self.merchant_metrics['total_revenue'].sum()),
            'avg_repeat_rate': round(self.merchant_metrics['repeat_rate'].mean(), 1),
            'avg_orders_per_customer': round(self.merchant_metrics['orders_per_customer'].mean(), 2),
            'top_category': self.get_category_performance().iloc[0]['category'] if len(self.get_category_performance()) > 0 else 'Unknown'
        }
