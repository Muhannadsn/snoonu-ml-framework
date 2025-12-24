"""Reactivation Targeting for Snoonu ML Framework.

Identifies dormant users most likely to return and recommends optimal
incentives and channels for reactivation campaigns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ReactivationTargeter:
    """Identify and score dormant users for reactivation campaigns."""

    def __init__(self, df: pd.DataFrame, dormancy_days: int = 14):
        """
        Initialize with event data.

        Args:
            df: Event dataframe
            dormancy_days: Days without activity to consider user dormant
        """
        self.df = df.copy()
        self.dormancy_days = dormancy_days
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for analysis."""
        if 'event_time' in self.df.columns:
            self.df['event_time'] = pd.to_datetime(self.df['event_time'])

        self.max_date = self.df['event_time'].max()
        self.min_date = self.df['event_time'].min()

    def identify_dormant_users(self) -> pd.DataFrame:
        """Identify users who haven't been active recently."""
        # Get last activity per user
        user_last_activity = self.df.groupby('amplitude_id').agg({
            'event_time': 'max'
        }).reset_index()
        user_last_activity.columns = ['user_id', 'last_activity']

        # Calculate days since last activity
        user_last_activity['days_inactive'] = (
            self.max_date - user_last_activity['last_activity']
        ).dt.total_seconds() / 86400

        # Filter to dormant users
        dormant = user_last_activity[
            user_last_activity['days_inactive'] >= self.dormancy_days
        ].copy()

        dormant['dormancy_tier'] = pd.cut(
            dormant['days_inactive'],
            bins=[0, 14, 30, 60, 90, float('inf')],
            labels=['2_weeks', '1_month', '2_months', '3_months', '3_months+']
        )

        return dormant

    def _build_user_features(self, user_ids: List) -> pd.DataFrame:
        """Build features for dormant users based on historical behavior."""
        user_df = self.df[self.df['amplitude_id'].isin(user_ids)].copy()

        features = []

        for user_id in user_ids:
            user_events = user_df[user_df['amplitude_id'] == user_id]

            if user_events.empty:
                continue

            # Parse order values
            orders = user_events[user_events['event_type'] == 'checkout_completed']
            order_values = []
            for _, order in orders.iterrows():
                try:
                    props = order.get('event_properties', '{}')
                    if isinstance(props, str):
                        props = json.loads(props)
                    if 'order_total' in props:
                        order_values.append(float(props['order_total']))
                except:
                    pass

            # Calculate features
            feature_dict = {
                'user_id': user_id,

                # Activity metrics
                'total_events': len(user_events),
                'total_sessions': user_events.groupby(
                    user_events['event_time'].dt.date
                ).ngroups,

                # Order history
                'total_orders': len(orders),
                'total_revenue': sum(order_values) if order_values else 0,
                'avg_order_value': np.mean(order_values) if order_values else 0,

                # Engagement depth
                'unique_event_types': user_events['event_type'].nunique(),
                'cart_events': len(user_events[user_events['event_type'].isin([
                    'product_added', 'cart_created', 'cart_page_viewed'
                ])]),
                'search_events': len(user_events[user_events['event_type'] == 'search_made']),

                # Platform
                'primary_platform': user_events['platform'].mode().iloc[0] if 'platform' in user_events.columns and not user_events['platform'].mode().empty else 'unknown',

                # Time patterns
                'first_seen': user_events['event_time'].min(),
                'last_seen': user_events['event_time'].max(),
            }

            # Calculate tenure (days as customer)
            feature_dict['tenure_days'] = (
                feature_dict['last_seen'] - feature_dict['first_seen']
            ).total_seconds() / 86400

            # Average days between orders
            if len(orders) > 1:
                order_times = orders['event_time'].sort_values()
                gaps = order_times.diff().dropna().dt.total_seconds() / 86400
                feature_dict['avg_order_gap_days'] = gaps.mean()
            else:
                feature_dict['avg_order_gap_days'] = 0

            features.append(feature_dict)

        return pd.DataFrame(features)

    def score_reactivation_potential(self) -> pd.DataFrame:
        """Score dormant users by likelihood to reactivate."""
        dormant = self.identify_dormant_users()

        if dormant.empty:
            return pd.DataFrame()

        # Build features for dormant users
        features_df = self._build_user_features(dormant['user_id'].tolist())

        if features_df.empty:
            return pd.DataFrame()

        # Merge dormancy info
        scored = features_df.merge(
            dormant[['user_id', 'days_inactive', 'dormancy_tier']],
            on='user_id',
            how='left'
        )

        # Calculate reactivation score (heuristic-based)
        # Higher score = more likely to reactivate

        # Normalize factors
        scored['order_score'] = np.clip(scored['total_orders'] / 10, 0, 1) * 30
        scored['revenue_score'] = np.clip(scored['total_revenue'] / 500, 0, 1) * 25
        scored['engagement_score'] = np.clip(scored['unique_event_types'] / 10, 0, 1) * 15
        scored['recency_score'] = np.clip(1 - (scored['days_inactive'] / 90), 0, 1) * 20
        scored['tenure_score'] = np.clip(scored['tenure_days'] / 60, 0, 1) * 10

        scored['reactivation_score'] = (
            scored['order_score'] +
            scored['revenue_score'] +
            scored['engagement_score'] +
            scored['recency_score'] +
            scored['tenure_score']
        )

        # Tier assignment
        scored['reactivation_tier'] = pd.cut(
            scored['reactivation_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['low', 'medium', 'high', 'very_high']
        )

        return scored.sort_values('reactivation_score', ascending=False)

    def recommend_incentive(self, user_features: pd.Series) -> Dict:
        """Recommend optimal incentive for a dormant user."""
        aov = user_features.get('avg_order_value', 0)
        total_orders = user_features.get('total_orders', 0)
        days_inactive = user_features.get('days_inactive', 0)
        reactivation_score = user_features.get('reactivation_score', 0)

        # Base incentive logic
        if reactivation_score >= 75:
            # High value, likely to return - minimal incentive needed
            incentive_type = 'reminder'
            incentive_value = 0
            message = "We miss you! Your favorites are waiting."
        elif reactivation_score >= 50:
            # Medium potential - small incentive
            if aov > 100:
                incentive_type = 'percentage_discount'
                incentive_value = 10
            else:
                incentive_type = 'free_delivery'
                incentive_value = 0
            message = "Come back for free delivery on your next order!"
        elif reactivation_score >= 25:
            # Lower potential - bigger incentive
            incentive_type = 'percentage_discount'
            incentive_value = 15 if aov > 50 else 20
            message = f"We want you back! Enjoy {incentive_value}% off your next order."
        else:
            # Low potential - aggressive incentive or skip
            if total_orders > 0:
                incentive_type = 'fixed_discount'
                incentive_value = 20
                message = "Special offer: 20 QAR off your next order!"
            else:
                incentive_type = 'skip'
                incentive_value = 0
                message = "User may not be worth reactivation cost"

        # Adjust for dormancy duration
        if days_inactive > 60 and incentive_type != 'skip':
            incentive_value = min(incentive_value * 1.5, 30)

        return {
            'incentive_type': incentive_type,
            'incentive_value': incentive_value,
            'suggested_message': message,
            'priority': 'high' if reactivation_score >= 50 else 'medium' if reactivation_score >= 25 else 'low'
        }

    def get_reactivation_campaign_segments(self) -> pd.DataFrame:
        """Segment dormant users for targeted campaigns."""
        scored = self.score_reactivation_potential()

        if scored.empty:
            return pd.DataFrame()

        # Create campaign segments
        segments = []

        # Segment 1: High-value recently dormant
        hv_recent = scored[
            (scored['total_revenue'] > scored['total_revenue'].median()) &
            (scored['days_inactive'] <= 30)
        ]
        if not hv_recent.empty:
            segments.append({
                'segment': 'high_value_recently_dormant',
                'user_count': len(hv_recent),
                'avg_revenue': hv_recent['total_revenue'].mean(),
                'avg_orders': hv_recent['total_orders'].mean(),
                'recommended_channel': 'push_notification',
                'recommended_incentive': 'free_delivery',
                'priority': 1
            })

        # Segment 2: Frequent orderers gone quiet
        frequent = scored[
            (scored['total_orders'] >= 3) &
            (scored['days_inactive'] > 30)
        ]
        if not frequent.empty:
            segments.append({
                'segment': 'frequent_orderers_churned',
                'user_count': len(frequent),
                'avg_revenue': frequent['total_revenue'].mean(),
                'avg_orders': frequent['total_orders'].mean(),
                'recommended_channel': 'email',
                'recommended_incentive': '15%_discount',
                'priority': 2
            })

        # Segment 3: One-time buyers
        one_time = scored[
            (scored['total_orders'] == 1) &
            (scored['days_inactive'] <= 45)
        ]
        if not one_time.empty:
            segments.append({
                'segment': 'one_time_buyers',
                'user_count': len(one_time),
                'avg_revenue': one_time['total_revenue'].mean(),
                'avg_orders': one_time['total_orders'].mean(),
                'recommended_channel': 'sms',
                'recommended_incentive': '20%_first_reorder',
                'priority': 3
            })

        # Segment 4: Long-dormant high spenders
        long_dormant_hv = scored[
            (scored['total_revenue'] > scored['total_revenue'].quantile(0.75)) &
            (scored['days_inactive'] > 60)
        ]
        if not long_dormant_hv.empty:
            segments.append({
                'segment': 'long_dormant_whales',
                'user_count': len(long_dormant_hv),
                'avg_revenue': long_dormant_hv['total_revenue'].mean(),
                'avg_orders': long_dormant_hv['total_orders'].mean(),
                'recommended_channel': 'personal_call',
                'recommended_incentive': 'vip_offer',
                'priority': 4
            })

        # Segment 5: Browsers who never converted
        browsers = scored[scored['total_orders'] == 0]
        if not browsers.empty:
            segments.append({
                'segment': 'never_converted',
                'user_count': len(browsers),
                'avg_revenue': 0,
                'avg_orders': 0,
                'recommended_channel': 'retargeting_ads',
                'recommended_incentive': 'first_order_discount',
                'priority': 5
            })

        return pd.DataFrame(segments)

    def export_for_campaign(self, segment: str = None, limit: int = 1000) -> pd.DataFrame:
        """Export user list for campaign execution."""
        scored = self.score_reactivation_potential()

        if scored.empty:
            return pd.DataFrame()

        # Add incentive recommendations
        incentives = scored.apply(
            lambda row: self.recommend_incentive(row),
            axis=1
        )
        incentive_df = pd.DataFrame(incentives.tolist())
        scored = pd.concat([scored.reset_index(drop=True), incentive_df], axis=1)

        # Filter by segment if specified
        if segment:
            if segment == 'high_value_recently_dormant':
                scored = scored[
                    (scored['total_revenue'] > scored['total_revenue'].median()) &
                    (scored['days_inactive'] <= 30)
                ]
            elif segment == 'frequent_orderers_churned':
                scored = scored[
                    (scored['total_orders'] >= 3) &
                    (scored['days_inactive'] > 30)
                ]
            elif segment == 'one_time_buyers':
                scored = scored[
                    (scored['total_orders'] == 1) &
                    (scored['days_inactive'] <= 45)
                ]

        # Select export columns
        export_cols = [
            'user_id', 'days_inactive', 'dormancy_tier',
            'total_orders', 'total_revenue', 'avg_order_value',
            'reactivation_score', 'reactivation_tier',
            'incentive_type', 'incentive_value', 'suggested_message', 'priority',
            'primary_platform'
        ]

        available_cols = [c for c in export_cols if c in scored.columns]

        return scored[available_cols].head(limit)

    def get_reactivation_summary(self) -> Dict:
        """Get summary statistics for reactivation analysis."""
        dormant = self.identify_dormant_users()
        scored = self.score_reactivation_potential()

        if dormant.empty:
            return {
                'total_dormant_users': 0,
                'dormancy_distribution': {},
                'reactivation_potential': {}
            }

        # Dormancy distribution
        dormancy_dist = dormant['dormancy_tier'].value_counts().to_dict()

        # Reactivation potential
        if not scored.empty:
            reactivation_dist = scored['reactivation_tier'].value_counts().to_dict()
            potential_revenue = scored[scored['reactivation_tier'].isin(['high', 'very_high'])]['total_revenue'].sum()
        else:
            reactivation_dist = {}
            potential_revenue = 0

        return {
            'total_dormant_users': len(dormant),
            'avg_days_inactive': dormant['days_inactive'].mean(),
            'dormancy_distribution': dormancy_dist,
            'reactivation_distribution': reactivation_dist,
            'high_potential_users': len(scored[scored['reactivation_tier'].isin(['high', 'very_high'])]) if not scored.empty else 0,
            'potential_recoverable_revenue': potential_revenue
        }
