"""Customer Journey Scoring.

Comprehensive customer scoring including engagement, health, lifecycle stage,
and next-best-action recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class CustomerScorer:
    """Customer journey scoring and next-best-action engine."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Event data DataFrame
        """
        self.df = df.copy()
        self._prepare_data()
        self._build_customer_profiles()

    def _prepare_data(self):
        """Prepare data for scoring."""
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.max_date = self.df['event_time'].max()
        logger.info(f"Preparing customer scoring for {self.df['amplitude_id'].nunique():,} users")

    def _build_customer_profiles(self):
        """Build comprehensive customer profiles."""
        logger.info("Building customer profiles...")

        # Base aggregations
        user_events = self.df.groupby('amplitude_id').agg({
            'event_time': ['min', 'max', 'count'],
            'event_type': lambda x: list(x)
        }).reset_index()
        user_events.columns = ['amplitude_id', 'first_seen', 'last_seen', 'total_events', 'event_sequence']

        # Calculate recency
        user_events['days_since_last'] = (self.max_date - user_events['last_seen']).dt.days
        user_events['days_since_first'] = (self.max_date - user_events['first_seen']).dt.days
        user_events['active_days'] = (user_events['last_seen'] - user_events['first_seen']).dt.days + 1

        # Order data
        checkouts = self.df[self.df['event_type'] == 'checkout_completed']
        order_stats = checkouts.groupby('amplitude_id').agg({
            'event_time': 'count'
        }).reset_index()
        order_stats.columns = ['amplitude_id', 'order_count']

        user_events = user_events.merge(order_stats, on='amplitude_id', how='left')
        user_events['order_count'] = user_events['order_count'].fillna(0).astype(int)

        # Engagement events
        engagement_events = ['product_page_viewed', 'product_added', 'search_made', 'merchant_page_viewed']
        for event in engagement_events:
            event_counts = self.df[self.df['event_type'] == event].groupby('amplitude_id').size()
            user_events[f'{event}_count'] = user_events['amplitude_id'].map(event_counts).fillna(0).astype(int)

        # Session proxy (count of homepage views)
        homepage_counts = self.df[self.df['event_type'] == 'homepage_viewed'].groupby('amplitude_id').size()
        user_events['sessions'] = user_events['amplitude_id'].map(homepage_counts).fillna(1).astype(int)

        # Funnel progress
        funnel_events = [
            'homepage_viewed', 'product_page_viewed', 'product_added',
            'cart_page_viewed', 'checkout_button_pressed', 'checkout_completed'
        ]

        def get_max_funnel_stage(events):
            for i, stage in enumerate(reversed(funnel_events)):
                if stage in events:
                    return len(funnel_events) - i
            return 0

        user_events['max_funnel_stage'] = user_events['event_sequence'].apply(get_max_funnel_stage)

        # Platform
        platform_mode = self.df.groupby('amplitude_id')['platform'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        )
        user_events['platform'] = user_events['amplitude_id'].map(platform_mode)

        self.profiles = user_events
        logger.info(f"Built profiles for {len(self.profiles):,} customers")

    def calculate_engagement_score(self) -> pd.DataFrame:
        """Calculate engagement score (0-100) for each customer.

        Components:
        - Recency (25%): How recently active
        - Frequency (25%): How often they visit
        - Depth (25%): How deep in funnel they go
        - Breadth (25%): Variety of actions
        """
        scores = self.profiles.copy()

        # Recency score (more recent = higher)
        # 0 days = 100, 30+ days = 0
        scores['recency_score'] = np.clip(100 - (scores['days_since_last'] * 100 / 30), 0, 100)

        # Frequency score (based on sessions per active day)
        scores['sessions_per_day'] = scores['sessions'] / np.maximum(scores['active_days'], 1)
        scores['frequency_score'] = np.clip(scores['sessions_per_day'] * 50, 0, 100)

        # Depth score (funnel progress)
        scores['depth_score'] = (scores['max_funnel_stage'] / 6) * 100

        # Breadth score (variety of actions)
        engagement_cols = ['product_page_viewed_count', 'product_added_count', 'search_made_count', 'merchant_page_viewed_count']
        scores['actions_variety'] = (scores[engagement_cols] > 0).sum(axis=1)
        scores['breadth_score'] = (scores['actions_variety'] / 4) * 100

        # Composite engagement score
        scores['engagement_score'] = (
            scores['recency_score'] * 0.25 +
            scores['frequency_score'] * 0.25 +
            scores['depth_score'] * 0.25 +
            scores['breadth_score'] * 0.25
        ).round(1)

        # Engagement tier
        scores['engagement_tier'] = pd.cut(
            scores['engagement_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Dormant', 'Low', 'Medium', 'High']
        )

        return scores[['amplitude_id', 'engagement_score', 'engagement_tier',
                       'recency_score', 'frequency_score', 'depth_score', 'breadth_score']]

    def calculate_health_score(self) -> pd.DataFrame:
        """Calculate customer health score combining multiple signals.

        Components:
        - Engagement (30%)
        - Purchase behavior (30%)
        - Lifecycle stage (20%)
        - Trajectory (20%): improving or declining
        """
        engagement = self.calculate_engagement_score()
        scores = self.profiles.merge(engagement[['amplitude_id', 'engagement_score']], on='amplitude_id')

        # Purchase score
        # Based on order count and recency of last order
        scores['purchase_score'] = np.clip(
            (scores['order_count'] * 15) +  # Up to 60 for 4+ orders
            (40 - scores['days_since_last']),  # Recency bonus
            0, 100
        )

        # Lifecycle score (longer relationship = higher, but with diminishing returns)
        scores['lifecycle_score'] = np.clip(
            np.log1p(scores['days_since_first']) * 20,
            0, 100
        )

        # Trajectory score (are they engaging more or less recently?)
        # Compare first half vs second half of their events
        def calculate_trajectory(row):
            events = row['event_sequence']
            if len(events) < 4:
                return 50  # Neutral for new users

            mid = len(events) // 2
            first_half_engagement = sum(1 for e in events[:mid] if e in ['product_added', 'checkout_completed'])
            second_half_engagement = sum(1 for e in events[mid:] if e in ['product_added', 'checkout_completed'])

            if first_half_engagement == 0:
                return 70 if second_half_engagement > 0 else 50

            ratio = second_half_engagement / first_half_engagement
            return np.clip(50 + (ratio - 1) * 30, 0, 100)

        scores['trajectory_score'] = scores.apply(calculate_trajectory, axis=1)

        # Composite health score
        scores['health_score'] = (
            scores['engagement_score'] * 0.30 +
            scores['purchase_score'] * 0.30 +
            scores['lifecycle_score'] * 0.20 +
            scores['trajectory_score'] * 0.20
        ).round(1)

        # Health tier
        scores['health_tier'] = pd.cut(
            scores['health_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Critical', 'At Risk', 'Healthy', 'Champion']
        )

        return scores[['amplitude_id', 'health_score', 'health_tier',
                       'engagement_score', 'purchase_score', 'lifecycle_score', 'trajectory_score',
                       'order_count', 'days_since_last']]

    def identify_lifecycle_stage(self) -> pd.DataFrame:
        """Identify customer lifecycle stage."""
        stages = self.profiles.copy()

        def get_stage(row):
            orders = row['order_count']
            days_since_first = row['days_since_first']
            days_since_last = row['days_since_last']
            max_funnel = row['max_funnel_stage']

            # New visitor (first seen recently, no order)
            if days_since_first <= 7 and orders == 0:
                if max_funnel >= 3:  # Got to cart
                    return 'New - High Intent'
                return 'New - Exploring'

            # First-time buyer
            if orders == 1:
                if days_since_last <= 7:
                    return 'First Purchase - Recent'
                elif days_since_last <= 30:
                    return 'First Purchase - Waiting'
                else:
                    return 'First Purchase - At Risk'

            # Repeat buyer
            if orders >= 2:
                if days_since_last <= 14:
                    return 'Repeat - Active'
                elif days_since_last <= 30:
                    return 'Repeat - Cooling'
                else:
                    return 'Repeat - Lapsed'

            # Non-buyer
            if orders == 0:
                if days_since_last <= 7:
                    return 'Browser - Recent'
                elif days_since_last <= 30:
                    return 'Browser - Inactive'
                else:
                    return 'Browser - Lost'

            return 'Unknown'

        stages['lifecycle_stage'] = stages.apply(get_stage, axis=1)

        # Lifecycle value (higher = more valuable stage)
        stage_values = {
            'New - Exploring': 20,
            'New - High Intent': 40,
            'Browser - Recent': 15,
            'Browser - Inactive': 5,
            'Browser - Lost': 0,
            'First Purchase - Recent': 60,
            'First Purchase - Waiting': 45,
            'First Purchase - At Risk': 25,
            'Repeat - Active': 100,
            'Repeat - Cooling': 70,
            'Repeat - Lapsed': 30,
            'Unknown': 10
        }
        stages['lifecycle_value'] = stages['lifecycle_stage'].map(stage_values)

        return stages[['amplitude_id', 'lifecycle_stage', 'lifecycle_value',
                       'order_count', 'days_since_first', 'days_since_last']]

    def get_next_best_action(self) -> pd.DataFrame:
        """Determine next best action for each customer."""
        health = self.calculate_health_score()
        lifecycle = self.identify_lifecycle_stage()

        nba = health.merge(lifecycle[['amplitude_id', 'lifecycle_stage']], on='amplitude_id')

        def determine_action(row):
            stage = row['lifecycle_stage']
            health = row['health_tier']
            orders = row['order_count']
            days_inactive = row['days_since_last']

            # New high intent - push to convert
            if stage == 'New - High Intent':
                return {
                    'action': 'First Order Incentive',
                    'channel': 'Push/Email',
                    'message': 'Complete your first order with 20% off',
                    'priority': 'High',
                    'expected_impact': 'Conversion'
                }

            # First purchase recent - nurture for 2nd order
            if stage == 'First Purchase - Recent':
                return {
                    'action': 'Second Order Campaign',
                    'channel': 'Push/Email',
                    'message': 'Welcome back! Try something new',
                    'priority': 'High',
                    'expected_impact': 'Retention'
                }

            # First purchase at risk - win back
            if stage == 'First Purchase - At Risk':
                return {
                    'action': 'Win Back Offer',
                    'channel': 'Email/SMS',
                    'message': 'We miss you! Here\'s a special offer',
                    'priority': 'Medium',
                    'expected_impact': 'Reactivation'
                }

            # Repeat active - loyalty program
            if stage == 'Repeat - Active':
                return {
                    'action': 'Loyalty Rewards',
                    'channel': 'In-App',
                    'message': 'Earn points on every order',
                    'priority': 'Low',
                    'expected_impact': 'LTV Increase'
                }

            # Repeat lapsed - re-engagement
            if stage == 'Repeat - Lapsed':
                return {
                    'action': 'Re-engagement Campaign',
                    'channel': 'Email/SMS',
                    'message': 'Come back! Your favorites are waiting',
                    'priority': 'High',
                    'expected_impact': 'Reactivation'
                }

            # Browser - personalized recommendations
            if 'Browser' in stage:
                return {
                    'action': 'Personalized Recommendations',
                    'channel': 'Push/Email',
                    'message': 'Discover restaurants you\'ll love',
                    'priority': 'Medium',
                    'expected_impact': 'Conversion'
                }

            # Default
            return {
                'action': 'General Engagement',
                'channel': 'Push',
                'message': 'Check out what\'s new',
                'priority': 'Low',
                'expected_impact': 'Engagement'
            }

        actions = nba.apply(determine_action, axis=1, result_type='expand')
        nba = pd.concat([nba, actions], axis=1)

        return nba[['amplitude_id', 'lifecycle_stage', 'health_tier', 'health_score',
                    'action', 'channel', 'message', 'priority', 'expected_impact']]

    def get_segment_summary(self) -> pd.DataFrame:
        """Get summary by lifecycle stage."""
        lifecycle = self.identify_lifecycle_stage()
        health = self.calculate_health_score()

        merged = lifecycle.merge(health[['amplitude_id', 'health_score']], on='amplitude_id')

        summary = merged.groupby('lifecycle_stage').agg({
            'amplitude_id': 'count',
            'lifecycle_value': 'first',
            'health_score': 'mean',
            'order_count': 'mean',
            'days_since_last': 'mean'
        }).reset_index()

        summary.columns = ['lifecycle_stage', 'users', 'stage_value', 'avg_health_score',
                          'avg_orders', 'avg_days_inactive']

        summary['pct_of_users'] = (summary['users'] / summary['users'].sum() * 100).round(1)
        summary = summary.sort_values('stage_value', ascending=False)

        return summary

    def get_high_value_at_risk(self, n: int = 100) -> pd.DataFrame:
        """Identify high-value customers at risk of churning."""
        health = self.calculate_health_score()
        lifecycle = self.identify_lifecycle_stage()

        merged = health.merge(lifecycle[['amplitude_id', 'lifecycle_stage']], on='amplitude_id')

        # High value = has ordered + not in champion tier
        at_risk = merged[
            (merged['order_count'] >= 2) &
            (merged['health_tier'].isin(['At Risk', 'Critical']))
        ].copy()

        at_risk = at_risk.nlargest(n, 'order_count')

        return at_risk[['amplitude_id', 'lifecycle_stage', 'health_score', 'health_tier',
                        'order_count', 'days_since_last']]

    def get_conversion_opportunities(self, n: int = 100) -> pd.DataFrame:
        """Identify users most likely to convert (browsers with high engagement)."""
        engagement = self.calculate_engagement_score()
        lifecycle = self.identify_lifecycle_stage()

        merged = engagement.merge(lifecycle[['amplitude_id', 'lifecycle_stage', 'order_count']], on='amplitude_id')

        # High engagement, no orders yet
        opportunities = merged[
            (merged['order_count'] == 0) &
            (merged['engagement_tier'].isin(['Medium', 'High']))
        ].copy()

        opportunities = opportunities.nlargest(n, 'engagement_score')

        return opportunities[['amplitude_id', 'lifecycle_stage', 'engagement_score', 'engagement_tier',
                              'depth_score', 'recency_score']]

    def get_summary(self) -> Dict:
        """Get overall customer scoring summary."""
        health = self.calculate_health_score()
        lifecycle = self.identify_lifecycle_stage()

        return {
            'total_customers': len(self.profiles),
            'with_orders': int((self.profiles['order_count'] > 0).sum()),
            'avg_health_score': round(health['health_score'].mean(), 1),
            'champions': int((health['health_tier'] == 'Champion').sum()),
            'at_risk': int((health['health_tier'].isin(['At Risk', 'Critical'])).sum()),
            'active_repeaters': int((lifecycle['lifecycle_stage'] == 'Repeat - Active').sum()),
            'lapsed_repeaters': int((lifecycle['lifecycle_stage'] == 'Repeat - Lapsed').sum())
        }

    def export_for_crm(self) -> pd.DataFrame:
        """Export comprehensive customer data for CRM integration."""
        health = self.calculate_health_score()
        lifecycle = self.identify_lifecycle_stage()
        nba = self.get_next_best_action()

        export = health.merge(
            lifecycle[['amplitude_id', 'lifecycle_stage', 'lifecycle_value']],
            on='amplitude_id'
        ).merge(
            nba[['amplitude_id', 'action', 'channel', 'priority']],
            on='amplitude_id'
        )

        export = export.merge(
            self.profiles[['amplitude_id', 'platform', 'first_seen', 'last_seen']],
            on='amplitude_id'
        )

        return export
