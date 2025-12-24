"""Promo & Marketing Analytics.

Analysis of promotional campaigns, discount effectiveness, and marketing ROI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class PromoAnalyzer:
    """Comprehensive promo code and marketing analytics."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Event data DataFrame with checkout_completed events
        """
        self.df = df.copy()
        self._extract_order_data()

    def _extract_order_data(self):
        """Extract order and promo data from checkout events."""
        logger.info("Extracting promo data...")

        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        # Handle different data formats
        if 'event_data_merchant_name' in checkouts.columns:
            # Dec 9 format - limited promo data
            checkouts['merchant_name'] = checkouts['event_data_merchant_name']
            checkouts['order_total'] = 0
            checkouts['promo_code'] = None
            checkouts['discount_amount'] = 0
            checkouts['payment_method'] = 'unknown'
            checkouts['order_type'] = 'unknown'
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
            checkouts['order_total'] = checkouts['props'].apply(lambda x: float(x.get('order_total', 0) or 0))
            checkouts['promo_code'] = checkouts['props'].apply(lambda x: x.get('promo_code') or x.get('coupon_code'))
            checkouts['discount_amount'] = checkouts['props'].apply(lambda x: float(x.get('discount_amount', 0) or x.get('promo_discount', 0) or 0))
            checkouts['payment_method'] = checkouts['props'].apply(lambda x: x.get('payment_method', 'unknown'))
            checkouts['order_type'] = checkouts['props'].apply(lambda x: x.get('order_type', 'unknown'))
            checkouts['delivery_fee'] = checkouts['props'].apply(lambda x: float(x.get('delivery_fee', 0) or 0))

        checkouts['event_time'] = pd.to_datetime(checkouts['event_time'])
        checkouts['date'] = checkouts['event_time'].dt.date
        checkouts['hour'] = checkouts['event_time'].dt.hour

        # Flag promo orders
        checkouts['has_promo'] = checkouts['promo_code'].notna() & (checkouts['promo_code'] != '')

        self.orders = checkouts.copy()

        # User order sequence
        self.orders = self.orders.sort_values(['amplitude_id', 'event_time'])
        self.orders['order_number'] = self.orders.groupby('amplitude_id').cumcount() + 1
        self.orders['is_first_order'] = self.orders['order_number'] == 1

        promo_count = self.orders['has_promo'].sum()
        logger.info(f"Extracted {len(self.orders):,} orders, {promo_count:,} with promo codes")

    def get_promo_summary(self) -> Dict:
        """Get overall promo usage summary."""
        total_orders = len(self.orders)
        promo_orders = self.orders['has_promo'].sum()
        promo_rate = promo_orders / total_orders * 100 if total_orders > 0 else 0

        promo_users = self.orders[self.orders['has_promo']]['amplitude_id'].nunique()
        total_users = self.orders['amplitude_id'].nunique()

        return {
            'total_orders': total_orders,
            'promo_orders': int(promo_orders),
            'promo_rate': round(promo_rate, 1),
            'unique_promo_codes': self.orders[self.orders['has_promo']]['promo_code'].nunique(),
            'promo_users': promo_users,
            'total_users': total_users,
            'user_promo_rate': round(promo_users / total_users * 100, 1) if total_users > 0 else 0,
            'total_discount_given': float(self.orders['discount_amount'].sum()),
            'avg_discount': float(self.orders[self.orders['has_promo']]['discount_amount'].mean()) if promo_orders > 0 else 0
        }

    def get_promo_performance(self, min_uses: int = 5) -> pd.DataFrame:
        """Get performance metrics for each promo code."""
        if self.orders['has_promo'].sum() == 0:
            return pd.DataFrame({'message': ['No promo codes found in data']})

        promo_stats = self.orders[self.orders['has_promo']].groupby('promo_code').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': ['sum', 'mean'],
            'discount_amount': ['sum', 'mean'],
            'is_first_order': 'mean'
        }).reset_index()

        promo_stats.columns = [
            'promo_code', 'uses', 'unique_users', 'total_revenue',
            'avg_order_value', 'total_discount', 'avg_discount', 'first_order_rate'
        ]

        # Filter by minimum uses
        promo_stats = promo_stats[promo_stats['uses'] >= min_uses]

        if len(promo_stats) == 0:
            return pd.DataFrame({'message': [f'No promo codes with {min_uses}+ uses']})

        # Calculate metrics
        promo_stats['discount_rate'] = (
            promo_stats['avg_discount'] / promo_stats['avg_order_value'] * 100
        ).round(1)
        promo_stats['first_order_rate'] = (promo_stats['first_order_rate'] * 100).round(1)
        promo_stats['uses_per_user'] = (promo_stats['uses'] / promo_stats['unique_users']).round(2)

        # ROI proxy (revenue per discount dollar)
        promo_stats['revenue_per_discount'] = (
            promo_stats['total_revenue'] / promo_stats['total_discount']
        ).round(2)
        promo_stats['revenue_per_discount'] = promo_stats['revenue_per_discount'].replace([np.inf, -np.inf], 0)

        promo_stats = promo_stats.sort_values('uses', ascending=False)

        return promo_stats[[
            'promo_code', 'uses', 'unique_users', 'uses_per_user',
            'total_revenue', 'avg_order_value', 'total_discount', 'avg_discount',
            'discount_rate', 'first_order_rate', 'revenue_per_discount'
        ]]

    def get_first_order_promos(self) -> pd.DataFrame:
        """Analyze which promos are most effective for first orders."""
        first_orders = self.orders[self.orders['is_first_order']]

        if first_orders['has_promo'].sum() == 0:
            return pd.DataFrame({'message': ['No promo codes on first orders']})

        promo_first = first_orders[first_orders['has_promo']].groupby('promo_code').agg({
            'amplitude_id': 'count',
            'order_total': 'mean',
            'discount_amount': 'mean'
        }).reset_index()
        promo_first.columns = ['promo_code', 'first_orders', 'avg_order_value', 'avg_discount']

        # Check if these users came back
        first_order_users_by_promo = first_orders[first_orders['has_promo']].groupby('promo_code')['amplitude_id'].apply(set)

        retention_data = []
        for promo, users in first_order_users_by_promo.items():
            repeat_orders = self.orders[
                (self.orders['amplitude_id'].isin(users)) &
                (self.orders['order_number'] > 1)
            ]
            users_who_returned = repeat_orders['amplitude_id'].nunique()
            retention_data.append({
                'promo_code': promo,
                'users_acquired': len(users),
                'users_returned': users_who_returned,
                'retention_rate': round(users_who_returned / len(users) * 100, 1) if users else 0
            })

        retention_df = pd.DataFrame(retention_data)
        result = promo_first.merge(retention_df, on='promo_code')

        # Filter to promos with meaningful volume
        result = result[result['first_orders'] >= 3]
        result = result.sort_values('first_orders', ascending=False)

        return result

    def get_promo_vs_organic(self) -> pd.DataFrame:
        """Compare promo orders vs organic orders."""
        comparison = self.orders.groupby('has_promo').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': 'mean',
            'merchant_name': 'nunique'
        }).reset_index()

        comparison.columns = ['has_promo', 'orders', 'unique_users', 'avg_order_value', 'unique_merchants']
        comparison['has_promo'] = comparison['has_promo'].map({True: 'Promo', False: 'Organic'})

        comparison['orders_per_user'] = (comparison['orders'] / comparison['unique_users']).round(2)

        return comparison

    def get_promo_by_merchant(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze promo usage by merchant."""
        merchant_promo = self.orders.groupby('merchant_name').agg({
            'amplitude_id': 'count',
            'has_promo': ['sum', 'mean'],
            'discount_amount': 'sum',
            'order_total': 'sum'
        }).reset_index()

        merchant_promo.columns = [
            'merchant_name', 'total_orders', 'promo_orders', 'promo_rate',
            'total_discount', 'total_revenue'
        ]

        merchant_promo['promo_rate'] = (merchant_promo['promo_rate'] * 100).round(1)

        # Discount as % of revenue
        merchant_promo['discount_pct_revenue'] = (
            merchant_promo['total_discount'] / merchant_promo['total_revenue'] * 100
        ).round(2)
        merchant_promo['discount_pct_revenue'] = merchant_promo['discount_pct_revenue'].replace([np.inf, -np.inf], 0)

        merchant_promo = merchant_promo[merchant_promo['total_orders'] >= 10]
        merchant_promo = merchant_promo.sort_values('promo_orders', ascending=False).head(top_n)

        return merchant_promo

    def get_promo_cannibalization(self) -> pd.DataFrame:
        """Detect potential promo cannibalization.

        Identifies users who:
        1. Only order with promos (promo dependent)
        2. Reduced organic ordering after using promos
        """
        user_behavior = self.orders.groupby('amplitude_id').agg({
            'order_number': 'max',
            'has_promo': ['sum', 'mean'],
            'order_total': 'mean'
        }).reset_index()

        user_behavior.columns = ['amplitude_id', 'total_orders', 'promo_orders', 'promo_rate', 'avg_order_value']

        # Categorize users
        def categorize(row):
            if row['total_orders'] == 1:
                return 'Single Order'
            elif row['promo_rate'] == 1:
                return 'Promo Dependent (100% promo)'
            elif row['promo_rate'] >= 0.7:
                return 'High Promo (70%+)'
            elif row['promo_rate'] >= 0.3:
                return 'Mixed (30-70% promo)'
            elif row['promo_rate'] > 0:
                return 'Occasional Promo (<30%)'
            else:
                return 'Organic Only'

        user_behavior['user_type'] = user_behavior.apply(categorize, axis=1)

        summary = user_behavior.groupby('user_type').agg({
            'amplitude_id': 'count',
            'total_orders': 'mean',
            'avg_order_value': 'mean'
        }).reset_index()

        summary.columns = ['user_type', 'users', 'avg_orders', 'avg_order_value']
        summary['pct_of_users'] = (summary['users'] / summary['users'].sum() * 100).round(1)

        # Order for display
        type_order = ['Promo Dependent (100% promo)', 'High Promo (70%+)', 'Mixed (30-70% promo)',
                      'Occasional Promo (<30%)', 'Organic Only', 'Single Order']
        summary['sort_order'] = summary['user_type'].apply(lambda x: type_order.index(x) if x in type_order else 99)
        summary = summary.sort_values('sort_order').drop('sort_order', axis=1)

        return summary

    def get_promo_timing(self) -> pd.DataFrame:
        """Analyze promo usage by time of day and day of week."""
        self.orders['dayofweek'] = pd.to_datetime(self.orders['event_time']).dt.dayofweek

        time_analysis = self.orders.groupby('hour').agg({
            'amplitude_id': 'count',
            'has_promo': 'mean'
        }).reset_index()

        time_analysis.columns = ['hour', 'orders', 'promo_rate']
        time_analysis['promo_rate'] = (time_analysis['promo_rate'] * 100).round(1)

        return time_analysis

    def get_payment_method_analysis(self) -> pd.DataFrame:
        """Analyze promo usage by payment method."""
        payment_promo = self.orders.groupby('payment_method').agg({
            'amplitude_id': 'count',
            'has_promo': ['sum', 'mean'],
            'order_total': 'mean'
        }).reset_index()

        payment_promo.columns = ['payment_method', 'orders', 'promo_orders', 'promo_rate', 'avg_order_value']
        payment_promo['promo_rate'] = (payment_promo['promo_rate'] * 100).round(1)
        payment_promo = payment_promo.sort_values('orders', ascending=False)

        return payment_promo

    def get_promo_effectiveness_score(self) -> pd.DataFrame:
        """Score promo codes by overall effectiveness.

        Scoring:
        - Acquisition power (% first orders): 30%
        - Retention (users who returned): 30%
        - Revenue efficiency (revenue per discount): 25%
        - Reach (unique users): 15%
        """
        if self.orders['has_promo'].sum() == 0:
            return pd.DataFrame({'message': ['No promo codes found']})

        promo_perf = self.get_promo_performance(min_uses=3)

        if 'message' in promo_perf.columns:
            return promo_perf

        first_order_data = self.get_first_order_promos()

        if 'message' not in first_order_data.columns:
            promo_perf = promo_perf.merge(
                first_order_data[['promo_code', 'retention_rate']],
                on='promo_code',
                how='left'
            )
            promo_perf['retention_rate'] = promo_perf['retention_rate'].fillna(0)
        else:
            promo_perf['retention_rate'] = 0

        # Normalize scores (percentile ranks)
        promo_perf['acquisition_score'] = promo_perf['first_order_rate'].rank(pct=True) * 100
        promo_perf['retention_score'] = promo_perf['retention_rate'].rank(pct=True) * 100
        promo_perf['efficiency_score'] = promo_perf['revenue_per_discount'].rank(pct=True) * 100
        promo_perf['reach_score'] = promo_perf['unique_users'].rank(pct=True) * 100

        # Weighted score
        promo_perf['effectiveness_score'] = (
            promo_perf['acquisition_score'] * 0.30 +
            promo_perf['retention_score'] * 0.30 +
            promo_perf['efficiency_score'] * 0.25 +
            promo_perf['reach_score'] * 0.15
        ).round(1)

        promo_perf = promo_perf.sort_values('effectiveness_score', ascending=False)

        return promo_perf[[
            'promo_code', 'uses', 'unique_users', 'effectiveness_score',
            'first_order_rate', 'retention_rate', 'revenue_per_discount',
            'avg_order_value', 'avg_discount'
        ]]

    def get_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations based on promo analysis."""
        recommendations = []

        summary = self.get_promo_summary()

        # Check promo dependency
        cannibalization = self.get_promo_cannibalization()
        promo_dependent = cannibalization[cannibalization['user_type'] == 'Promo Dependent (100% promo)']
        if len(promo_dependent) > 0 and promo_dependent.iloc[0]['pct_of_users'] > 10:
            recommendations.append({
                'type': 'warning',
                'title': 'High Promo Dependency',
                'message': f"{promo_dependent.iloc[0]['pct_of_users']:.1f}% of repeat users only order with promos",
                'action': 'Consider limiting promo frequency for heavy promo users'
            })

        # Check if promos are driving first orders
        if summary['promo_rate'] < 5:
            recommendations.append({
                'type': 'opportunity',
                'title': 'Low Promo Utilization',
                'message': f"Only {summary['promo_rate']:.1f}% of orders use promos",
                'action': 'Consider targeted promo campaigns to drive acquisition'
            })

        # Check high discount rates
        promo_perf = self.get_promo_performance(min_uses=3)
        if 'discount_rate' in promo_perf.columns:
            high_discount = promo_perf[promo_perf['discount_rate'] > 30]
            if len(high_discount) > 0:
                recommendations.append({
                    'type': 'warning',
                    'title': 'High Discount Promos',
                    'message': f"{len(high_discount)} promo codes have >30% discount rate",
                    'action': 'Review profitability of these high-discount promos'
                })

        return recommendations
