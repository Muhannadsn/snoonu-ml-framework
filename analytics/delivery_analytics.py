"""Delivery & Fulfillment Analytics.

Analysis of delivery performance, timing impact, and fulfillment optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class DeliveryAnalyzer:
    """Comprehensive delivery and fulfillment analytics."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Event data DataFrame with order events
        """
        self.df = df.copy()
        self._extract_delivery_data()

    def _extract_delivery_data(self):
        """Extract delivery and fulfillment data."""
        logger.info("Extracting delivery data...")

        # Get checkout and delivery events
        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()
        deliveries = self.df[self.df['event_type'] == 'order_delivered'].copy()

        def parse_props(props):
            if isinstance(props, str):
                try:
                    return json.loads(props)
                except:
                    return {}
            return props if isinstance(props, dict) else {}

        # Parse checkout data
        if 'event_properties' in checkouts.columns:
            checkouts['props'] = checkouts['event_properties'].apply(parse_props)
            checkouts['order_id'] = checkouts['props'].apply(lambda x: x.get('order_id'))
            checkouts['merchant_name'] = checkouts['props'].apply(lambda x: x.get('merchant_name', 'Unknown'))
            checkouts['order_total'] = checkouts['props'].apply(lambda x: float(x.get('order_total', 0) or 0))
            checkouts['order_type'] = checkouts['props'].apply(lambda x: x.get('order_type', 'delivery'))
            checkouts['delivery_fee'] = checkouts['props'].apply(lambda x: float(x.get('delivery_fee', 0) or 0))
            checkouts['estimated_time'] = checkouts['props'].apply(lambda x: x.get('estimated_delivery_time') or x.get('eta_minutes'))
        elif 'event_data_merchant_name' in checkouts.columns:
            checkouts['order_id'] = checkouts.get('event_data_order_id', range(len(checkouts)))
            checkouts['merchant_name'] = checkouts['event_data_merchant_name']
            checkouts['order_total'] = 0
            checkouts['order_type'] = 'delivery'
            checkouts['delivery_fee'] = 0
            checkouts['estimated_time'] = None

        checkouts['checkout_time'] = pd.to_datetime(checkouts['event_time'])
        checkouts['hour'] = checkouts['checkout_time'].dt.hour
        checkouts['dayofweek'] = checkouts['checkout_time'].dt.dayofweek
        checkouts['date'] = checkouts['checkout_time'].dt.date

        # Parse delivery data
        if len(deliveries) > 0:
            if 'event_properties' in deliveries.columns:
                deliveries['props'] = deliveries['event_properties'].apply(parse_props)
                deliveries['order_id'] = deliveries['props'].apply(lambda x: x.get('order_id'))
                deliveries['fulfillment_time'] = deliveries['props'].apply(
                    lambda x: float(x.get('fulfillment_time', 0) or x.get('delivery_time_minutes', 0) or 0)
                )
                deliveries['total_delivered_orders'] = deliveries['props'].apply(
                    lambda x: int(x.get('total_delivered_orders', 0) or 0)
                )
            else:
                deliveries['order_id'] = deliveries.get('event_data_order_id')
                deliveries['fulfillment_time'] = 0
                deliveries['total_delivered_orders'] = 0

            deliveries['delivery_time'] = pd.to_datetime(deliveries['event_time'])
            self.deliveries = deliveries
            self.has_delivery_events = True
        else:
            self.deliveries = pd.DataFrame()
            self.has_delivery_events = False

        self.orders = checkouts
        self.has_data = len(self.orders) > 0

        # Merge checkout with delivery if possible
        if self.has_delivery_events and 'order_id' in checkouts.columns and 'order_id' in deliveries.columns:
            self.order_fulfillment = checkouts.merge(
                deliveries[['order_id', 'delivery_time', 'fulfillment_time']],
                on='order_id',
                how='left'
            )
            self.order_fulfillment['actual_time_minutes'] = (
                (self.order_fulfillment['delivery_time'] - self.order_fulfillment['checkout_time']).dt.total_seconds() / 60
            )
        else:
            self.order_fulfillment = checkouts.copy()
            self.order_fulfillment['delivery_time'] = None
            self.order_fulfillment['fulfillment_time'] = None
            self.order_fulfillment['actual_time_minutes'] = None

        logger.info(f"Extracted {len(self.orders):,} orders, {len(self.deliveries):,} deliveries")

    def get_summary(self) -> Dict:
        """Get delivery analytics summary."""
        if not self.has_data:
            return {'error': 'No order data available'}

        total_orders = len(self.orders)
        unique_customers = self.orders['amplitude_id'].nunique()

        # Order type breakdown
        order_types = self.orders['order_type'].value_counts().to_dict()

        # Delivery stats
        if self.has_delivery_events:
            fulfilled_orders = len(self.deliveries)
            fulfillment_rate = fulfilled_orders / total_orders * 100 if total_orders > 0 else 0

            valid_times = self.order_fulfillment['actual_time_minutes'].dropna()
            valid_times = valid_times[(valid_times > 0) & (valid_times < 180)]  # Reasonable range
            avg_delivery_time = valid_times.mean() if len(valid_times) > 0 else 0
            median_delivery_time = valid_times.median() if len(valid_times) > 0 else 0
        else:
            fulfillment_rate = 0
            avg_delivery_time = 0
            median_delivery_time = 0

        return {
            'total_orders': total_orders,
            'unique_customers': unique_customers,
            'order_types': order_types,
            'fulfillment_rate': round(fulfillment_rate, 1),
            'avg_delivery_time_min': round(avg_delivery_time, 1),
            'median_delivery_time_min': round(median_delivery_time, 1),
            'has_delivery_data': self.has_delivery_events
        }

    def get_delivery_time_distribution(self) -> pd.DataFrame:
        """Get delivery time distribution."""
        if not self.has_delivery_events:
            return pd.DataFrame({'message': ['No delivery data available']})

        valid_times = self.order_fulfillment[['actual_time_minutes', 'amplitude_id']].dropna()
        valid_times = valid_times[(valid_times['actual_time_minutes'] > 0) & (valid_times['actual_time_minutes'] < 180)]

        if len(valid_times) == 0:
            return pd.DataFrame({'message': ['No valid delivery times found']})

        # Bucket delivery times
        bins = [0, 15, 30, 45, 60, 90, 120, float('inf')]
        labels = ['<15 min', '15-30 min', '30-45 min', '45-60 min', '60-90 min', '90-120 min', '120+ min']

        valid_times['time_bucket'] = pd.cut(
            valid_times['actual_time_minutes'],
            bins=bins,
            labels=labels
        )

        dist = valid_times.groupby('time_bucket').agg({
            'amplitude_id': 'count'
        }).reset_index()
        dist.columns = ['delivery_time', 'orders']
        dist['pct'] = (dist['orders'] / dist['orders'].sum() * 100).round(1)

        return dist

    def get_delivery_by_hour(self) -> pd.DataFrame:
        """Get delivery performance by order hour."""
        if not self.has_data:
            return pd.DataFrame()

        hourly = self.orders.groupby('hour').agg({
            'amplitude_id': 'count'
        }).reset_index()
        hourly.columns = ['hour', 'orders']

        if self.has_delivery_events:
            # Add delivery times by hour
            valid_fulfillment = self.order_fulfillment[
                (self.order_fulfillment['actual_time_minutes'].notna()) &
                (self.order_fulfillment['actual_time_minutes'] > 0) &
                (self.order_fulfillment['actual_time_minutes'] < 180)
            ]

            hourly_times = valid_fulfillment.groupby('hour')['actual_time_minutes'].agg(['mean', 'median']).reset_index()
            hourly_times.columns = ['hour', 'avg_delivery_time', 'median_delivery_time']
            hourly = hourly.merge(hourly_times, on='hour', how='left')
        else:
            hourly['avg_delivery_time'] = None
            hourly['median_delivery_time'] = None

        # Time slot labels
        def get_time_slot(hour):
            if 6 <= hour < 11:
                return 'Breakfast'
            elif 11 <= hour < 15:
                return 'Lunch'
            elif 15 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Dinner'
            else:
                return 'Late Night'

        hourly['time_slot'] = hourly['hour'].apply(get_time_slot)

        return hourly

    def get_peak_hours(self) -> pd.DataFrame:
        """Identify peak order hours and capacity constraints."""
        if not self.has_data:
            return pd.DataFrame()

        # Orders by hour and day
        self.orders['hour_day'] = self.orders['checkout_time'].dt.floor('H')

        hourly_volume = self.orders.groupby('hour_day').agg({
            'amplitude_id': 'count'
        }).reset_index()
        hourly_volume.columns = ['datetime', 'orders']
        hourly_volume['hour'] = hourly_volume['datetime'].dt.hour

        # Calculate percentiles for each hour
        peak_stats = hourly_volume.groupby('hour').agg({
            'orders': ['mean', 'median', 'max', lambda x: x.quantile(0.95)]
        }).reset_index()
        peak_stats.columns = ['hour', 'avg_orders', 'median_orders', 'max_orders', 'p95_orders']

        # Identify peak hours (above 75th percentile of average)
        peak_threshold = peak_stats['avg_orders'].quantile(0.75)
        peak_stats['is_peak'] = peak_stats['avg_orders'] >= peak_threshold

        return peak_stats

    def get_merchant_delivery_performance(self, min_orders: int = 10) -> pd.DataFrame:
        """Get delivery performance by merchant."""
        if not self.has_data:
            return pd.DataFrame()

        merchant_stats = self.orders.groupby('merchant_name').agg({
            'amplitude_id': 'count',
            'order_total': 'mean',
            'delivery_fee': 'mean'
        }).reset_index()
        merchant_stats.columns = ['merchant_name', 'orders', 'avg_order_value', 'avg_delivery_fee']

        if self.has_delivery_events:
            valid_fulfillment = self.order_fulfillment[
                (self.order_fulfillment['actual_time_minutes'].notna()) &
                (self.order_fulfillment['actual_time_minutes'] > 0) &
                (self.order_fulfillment['actual_time_minutes'] < 180)
            ]

            merchant_times = valid_fulfillment.groupby('merchant_name').agg({
                'actual_time_minutes': ['mean', 'median', 'count']
            }).reset_index()
            merchant_times.columns = ['merchant_name', 'avg_delivery_time', 'median_delivery_time', 'delivered_orders']

            merchant_stats = merchant_stats.merge(merchant_times, on='merchant_name', how='left')

        merchant_stats = merchant_stats[merchant_stats['orders'] >= min_orders]
        merchant_stats = merchant_stats.sort_values('orders', ascending=False)

        return merchant_stats

    def get_delivery_impact_on_reorders(self) -> pd.DataFrame:
        """Analyze how delivery time affects likelihood to reorder."""
        if not self.has_delivery_events:
            return pd.DataFrame({'message': ['Delivery data required for this analysis']})

        # Get users with delivery time data
        valid_fulfillment = self.order_fulfillment[
            (self.order_fulfillment['actual_time_minutes'].notna()) &
            (self.order_fulfillment['actual_time_minutes'] > 0) &
            (self.order_fulfillment['actual_time_minutes'] < 180)
        ].copy()

        if len(valid_fulfillment) == 0:
            return pd.DataFrame({'message': ['No valid delivery time data']})

        # Bucket by delivery time
        bins = [0, 30, 45, 60, 90, float('inf')]
        labels = ['<30 min', '30-45 min', '45-60 min', '60-90 min', '90+ min']

        valid_fulfillment['delivery_bucket'] = pd.cut(
            valid_fulfillment['actual_time_minutes'],
            bins=bins,
            labels=labels
        )

        # Check if user reordered (has more than 1 order)
        user_order_counts = self.orders.groupby('amplitude_id').size()
        valid_fulfillment['user_reordered'] = valid_fulfillment['amplitude_id'].map(
            lambda x: user_order_counts.get(x, 1) > 1
        )

        # Analyze by delivery time bucket
        impact = valid_fulfillment.groupby('delivery_bucket').agg({
            'amplitude_id': ['count', 'nunique'],
            'user_reordered': 'mean'
        }).reset_index()
        impact.columns = ['delivery_time', 'orders', 'unique_users', 'reorder_rate']
        impact['reorder_rate'] = (impact['reorder_rate'] * 100).round(1)

        return impact

    def get_order_type_analysis(self) -> pd.DataFrame:
        """Analyze performance by order type (delivery, pickup, etc.)."""
        if not self.has_data:
            return pd.DataFrame()

        order_type_stats = self.orders.groupby('order_type').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': 'mean',
            'delivery_fee': 'mean'
        }).reset_index()
        order_type_stats.columns = ['order_type', 'orders', 'unique_customers', 'avg_order_value', 'avg_delivery_fee']

        order_type_stats['pct_of_orders'] = (order_type_stats['orders'] / order_type_stats['orders'].sum() * 100).round(1)
        order_type_stats = order_type_stats.sort_values('orders', ascending=False)

        return order_type_stats

    def get_delivery_fee_impact(self) -> pd.DataFrame:
        """Analyze delivery fee impact on order behavior."""
        if not self.has_data:
            return pd.DataFrame()

        # Only analyze orders with delivery fees
        delivery_orders = self.orders[self.orders['delivery_fee'] > 0].copy()

        if len(delivery_orders) == 0:
            return pd.DataFrame({'message': ['No delivery fee data available']})

        # Bucket by delivery fee
        bins = [0, 5, 10, 15, 20, float('inf')]
        labels = ['0-5 QAR', '5-10 QAR', '10-15 QAR', '15-20 QAR', '20+ QAR']

        delivery_orders['fee_bucket'] = pd.cut(
            delivery_orders['delivery_fee'],
            bins=bins,
            labels=labels
        )

        fee_analysis = delivery_orders.groupby('fee_bucket').agg({
            'amplitude_id': ['count', 'nunique'],
            'order_total': 'mean'
        }).reset_index()
        fee_analysis.columns = ['delivery_fee', 'orders', 'unique_customers', 'avg_order_value']
        fee_analysis['pct_of_orders'] = (fee_analysis['orders'] / fee_analysis['orders'].sum() * 100).round(1)

        return fee_analysis

    def get_recommendations(self) -> List[Dict]:
        """Generate actionable delivery recommendations."""
        recommendations = []

        if not self.has_data:
            return [{'type': 'info', 'message': 'No data available'}]

        summary = self.get_summary()

        # Check fulfillment rate
        if self.has_delivery_events and summary['fulfillment_rate'] < 90:
            recommendations.append({
                'type': 'warning',
                'title': f"Low Fulfillment Rate ({summary['fulfillment_rate']:.1f}%)",
                'message': 'Some orders are not being marked as delivered',
                'action': 'Investigate order tracking and driver app compliance'
            })

        # Check delivery times
        if summary['avg_delivery_time_min'] > 45:
            recommendations.append({
                'type': 'warning',
                'title': f"High Average Delivery Time ({summary['avg_delivery_time_min']:.0f} min)",
                'message': 'Delivery times are above optimal threshold',
                'action': 'Review logistics, driver allocation, and restaurant prep times'
            })

        # Peak hour analysis
        peak_hours = self.get_peak_hours()
        if len(peak_hours) > 0:
            peak_count = peak_hours['is_peak'].sum()
            if peak_count > 0:
                peak_hour_list = peak_hours[peak_hours['is_peak']]['hour'].tolist()
                recommendations.append({
                    'type': 'info',
                    'title': f"{peak_count} Peak Hours Identified",
                    'message': f"Peak hours: {', '.join([f'{h}:00' for h in peak_hour_list[:5]])}",
                    'action': 'Ensure adequate driver coverage during peak hours'
                })

        # Delivery time impact on reorders
        if self.has_delivery_events:
            impact = self.get_delivery_impact_on_reorders()
            if 'reorder_rate' in impact.columns:
                fast_reorder = impact[impact['delivery_time'] == '<30 min']['reorder_rate'].values
                slow_reorder = impact[impact['delivery_time'] == '90+ min']['reorder_rate'].values

                if len(fast_reorder) > 0 and len(slow_reorder) > 0:
                    diff = fast_reorder[0] - slow_reorder[0]
                    if diff > 10:
                        recommendations.append({
                            'type': 'insight',
                            'title': 'Fast Delivery Drives Retention',
                            'message': f"Users with <30 min delivery have {diff:.1f}% higher reorder rate",
                            'action': 'Prioritize delivery speed for first-time customers'
                        })

        return recommendations
