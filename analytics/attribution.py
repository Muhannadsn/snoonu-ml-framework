"""Attribution Modeling for Snoonu ML Framework.

Analyzes which touchpoints drive conversions using multiple attribution models:
- First-touch: Credit to first interaction
- Last-touch: Credit to last interaction before conversion
- Linear: Equal credit to all touchpoints
- Time-decay: More credit to recent touchpoints
- Position-based: 40% first, 40% last, 20% distributed to middle
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


class AttributionModeler:
    """Multi-touch attribution modeling for conversion analysis."""

    # Define touchpoint events (ordered by typical funnel position)
    TOUCHPOINT_EVENTS = [
        'homepage_viewed',
        'category_page_viewed',
        'search_made',
        'merchant_page_viewed',
        'product_page_viewed',
        'product_added',
        'cart_created',
        'cart_page_viewed',
        'checkout_button_pressed',
        'payment_initiated'
    ]

    CONVERSION_EVENT = 'checkout_completed'

    def __init__(self, df: pd.DataFrame):
        """Initialize with event data."""
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for attribution analysis."""
        # Ensure event_time is datetime
        if 'event_time' in self.df.columns:
            self.df['event_time'] = pd.to_datetime(self.df['event_time'])

        # Sort by user and time
        self.df = self.df.sort_values(['amplitude_id', 'event_time'])

    def _get_user_journeys(self) -> pd.DataFrame:
        """Extract user journeys leading to conversions."""
        journeys = []

        # Get all conversions
        conversions = self.df[self.df['event_type'] == self.CONVERSION_EVENT].copy()

        if conversions.empty:
            return pd.DataFrame()

        # For each conversion, get the touchpoints leading to it
        for _, conversion in conversions.iterrows():
            user_id = conversion['amplitude_id']
            conversion_time = conversion['event_time']

            # Get user's events before this conversion (within 24 hours)
            lookback_start = conversion_time - timedelta(hours=24)

            user_events = self.df[
                (self.df['amplitude_id'] == user_id) &
                (self.df['event_time'] >= lookback_start) &
                (self.df['event_time'] < conversion_time) &
                (self.df['event_type'].isin(self.TOUCHPOINT_EVENTS))
            ].copy()

            if not user_events.empty:
                touchpoints = user_events['event_type'].tolist()
                touchpoint_times = user_events['event_time'].tolist()

                journeys.append({
                    'user_id': user_id,
                    'conversion_time': conversion_time,
                    'touchpoints': touchpoints,
                    'touchpoint_times': touchpoint_times,
                    'journey_length': len(touchpoints),
                    'journey_duration_mins': (conversion_time - user_events['event_time'].min()).total_seconds() / 60
                })

        return pd.DataFrame(journeys)

    def first_touch_attribution(self) -> pd.DataFrame:
        """First-touch attribution: 100% credit to first touchpoint."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return pd.DataFrame(columns=['touchpoint', 'conversions', 'attribution_pct'])

        # Get first touchpoint from each journey
        first_touches = journeys['touchpoints'].apply(lambda x: x[0] if x else None)

        # Count attributions
        attribution = first_touches.value_counts().reset_index()
        attribution.columns = ['touchpoint', 'conversions']
        attribution['attribution_pct'] = attribution['conversions'] / attribution['conversions'].sum() * 100
        attribution['model'] = 'first_touch'

        return attribution

    def last_touch_attribution(self) -> pd.DataFrame:
        """Last-touch attribution: 100% credit to last touchpoint before conversion."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return pd.DataFrame(columns=['touchpoint', 'conversions', 'attribution_pct'])

        # Get last touchpoint from each journey
        last_touches = journeys['touchpoints'].apply(lambda x: x[-1] if x else None)

        # Count attributions
        attribution = last_touches.value_counts().reset_index()
        attribution.columns = ['touchpoint', 'conversions']
        attribution['attribution_pct'] = attribution['conversions'] / attribution['conversions'].sum() * 100
        attribution['model'] = 'last_touch'

        return attribution

    def linear_attribution(self) -> pd.DataFrame:
        """Linear attribution: Equal credit to all touchpoints in journey."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return pd.DataFrame(columns=['touchpoint', 'conversions', 'attribution_pct'])

        # Distribute credit equally across all touchpoints
        attribution_credits = {}

        for _, journey in journeys.iterrows():
            touchpoints = journey['touchpoints']
            if touchpoints:
                credit_per_touch = 1.0 / len(touchpoints)
                for tp in touchpoints:
                    attribution_credits[tp] = attribution_credits.get(tp, 0) + credit_per_touch

        attribution = pd.DataFrame([
            {'touchpoint': k, 'conversions': v}
            for k, v in attribution_credits.items()
        ])

        if not attribution.empty:
            attribution['attribution_pct'] = attribution['conversions'] / attribution['conversions'].sum() * 100
            attribution = attribution.sort_values('conversions', ascending=False)
        attribution['model'] = 'linear'

        return attribution

    def time_decay_attribution(self, half_life_hours: float = 6.0) -> pd.DataFrame:
        """Time-decay attribution: More credit to touchpoints closer to conversion."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return pd.DataFrame(columns=['touchpoint', 'conversions', 'attribution_pct'])

        attribution_credits = {}

        for _, journey in journeys.iterrows():
            touchpoints = journey['touchpoints']
            touchpoint_times = journey['touchpoint_times']
            conversion_time = journey['conversion_time']

            if touchpoints and touchpoint_times:
                # Calculate decay weights
                weights = []
                for tp_time in touchpoint_times:
                    hours_before = (conversion_time - tp_time).total_seconds() / 3600
                    # Exponential decay: weight = 2^(-hours/half_life)
                    weight = 2 ** (-hours_before / half_life_hours)
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    for tp, weight in zip(touchpoints, weights):
                        credit = weight / total_weight
                        attribution_credits[tp] = attribution_credits.get(tp, 0) + credit

        attribution = pd.DataFrame([
            {'touchpoint': k, 'conversions': v}
            for k, v in attribution_credits.items()
        ])

        if not attribution.empty:
            attribution['attribution_pct'] = attribution['conversions'] / attribution['conversions'].sum() * 100
            attribution = attribution.sort_values('conversions', ascending=False)
        attribution['model'] = 'time_decay'

        return attribution

    def position_based_attribution(self, first_weight: float = 0.4, last_weight: float = 0.4) -> pd.DataFrame:
        """Position-based attribution: 40% first, 40% last, 20% distributed to middle."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return pd.DataFrame(columns=['touchpoint', 'conversions', 'attribution_pct'])

        middle_weight = 1.0 - first_weight - last_weight
        attribution_credits = {}

        for _, journey in journeys.iterrows():
            touchpoints = journey['touchpoints']

            if not touchpoints:
                continue

            if len(touchpoints) == 1:
                # Single touchpoint gets all credit
                attribution_credits[touchpoints[0]] = attribution_credits.get(touchpoints[0], 0) + 1.0
            elif len(touchpoints) == 2:
                # Two touchpoints: split first/last weight
                attribution_credits[touchpoints[0]] = attribution_credits.get(touchpoints[0], 0) + first_weight + middle_weight/2
                attribution_credits[touchpoints[-1]] = attribution_credits.get(touchpoints[-1], 0) + last_weight + middle_weight/2
            else:
                # Multiple touchpoints
                attribution_credits[touchpoints[0]] = attribution_credits.get(touchpoints[0], 0) + first_weight
                attribution_credits[touchpoints[-1]] = attribution_credits.get(touchpoints[-1], 0) + last_weight

                # Distribute middle weight equally
                middle_touchpoints = touchpoints[1:-1]
                if middle_touchpoints:
                    per_middle = middle_weight / len(middle_touchpoints)
                    for tp in middle_touchpoints:
                        attribution_credits[tp] = attribution_credits.get(tp, 0) + per_middle

        attribution = pd.DataFrame([
            {'touchpoint': k, 'conversions': v}
            for k, v in attribution_credits.items()
        ])

        if not attribution.empty:
            attribution['attribution_pct'] = attribution['conversions'] / attribution['conversions'].sum() * 100
            attribution = attribution.sort_values('conversions', ascending=False)
        attribution['model'] = 'position_based'

        return attribution

    def compare_models(self) -> pd.DataFrame:
        """Compare all attribution models side by side."""
        models = {
            'first_touch': self.first_touch_attribution(),
            'last_touch': self.last_touch_attribution(),
            'linear': self.linear_attribution(),
            'time_decay': self.time_decay_attribution(),
            'position_based': self.position_based_attribution()
        }

        # Combine all models
        comparison = pd.DataFrame({'touchpoint': self.TOUCHPOINT_EVENTS})

        for model_name, model_df in models.items():
            if not model_df.empty:
                model_df = model_df[['touchpoint', 'attribution_pct']].copy()
                model_df.columns = ['touchpoint', f'{model_name}_pct']
                comparison = comparison.merge(model_df, on='touchpoint', how='left')

        # Fill NaN with 0
        comparison = comparison.fillna(0)

        return comparison

    def get_journey_stats(self) -> Dict:
        """Get statistics about user journeys."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return {
                'total_conversions': 0,
                'avg_journey_length': 0,
                'avg_journey_duration_mins': 0,
                'most_common_paths': []
            }

        # Most common paths (top 10)
        path_counts = journeys['touchpoints'].apply(lambda x: ' -> '.join(x)).value_counts().head(10)
        most_common_paths = [
            {'path': path, 'count': count}
            for path, count in path_counts.items()
        ]

        return {
            'total_conversions': len(journeys),
            'avg_journey_length': journeys['journey_length'].mean(),
            'median_journey_length': journeys['journey_length'].median(),
            'avg_journey_duration_mins': journeys['journey_duration_mins'].mean(),
            'median_journey_duration_mins': journeys['journey_duration_mins'].median(),
            'most_common_paths': most_common_paths
        }

    def get_channel_influence(self) -> pd.DataFrame:
        """Analyze which touchpoints appear most in converting journeys."""
        journeys = self._get_user_journeys()

        if journeys.empty:
            return pd.DataFrame()

        # Count touchpoint appearances
        touchpoint_stats = {}

        for _, journey in journeys.iterrows():
            touchpoints = journey['touchpoints']
            seen = set()

            for i, tp in enumerate(touchpoints):
                if tp not in touchpoint_stats:
                    touchpoint_stats[tp] = {
                        'appearances': 0,
                        'as_first': 0,
                        'as_last': 0,
                        'as_middle': 0
                    }

                touchpoint_stats[tp]['appearances'] += 1

                if tp not in seen:  # Count position only once per journey
                    if i == 0:
                        touchpoint_stats[tp]['as_first'] += 1
                    elif i == len(touchpoints) - 1:
                        touchpoint_stats[tp]['as_last'] += 1
                    else:
                        touchpoint_stats[tp]['as_middle'] += 1
                    seen.add(tp)

        influence_df = pd.DataFrame([
            {
                'touchpoint': tp,
                'total_appearances': stats['appearances'],
                'as_first_touch': stats['as_first'],
                'as_last_touch': stats['as_last'],
                'as_middle_touch': stats['as_middle'],
                'appearance_rate': stats['appearances'] / len(journeys) * 100
            }
            for tp, stats in touchpoint_stats.items()
        ])

        return influence_df.sort_values('total_appearances', ascending=False)


class ChannelAttributor:
    """Attribution by marketing channel/platform."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with event data."""
        self.df = df.copy()

    def platform_attribution(self) -> pd.DataFrame:
        """Attribute conversions by platform (iOS/Android)."""
        conversions = self.df[self.df['event_type'] == 'checkout_completed']

        if conversions.empty or 'platform' not in conversions.columns:
            return pd.DataFrame()

        platform_conv = conversions.groupby('platform').agg({
            'amplitude_id': 'count'
        }).reset_index()
        platform_conv.columns = ['platform', 'conversions']
        platform_conv['conversion_pct'] = platform_conv['conversions'] / platform_conv['conversions'].sum() * 100

        # Get total users by platform for conversion rate
        total_users = self.df.groupby('platform')['amplitude_id'].nunique().reset_index()
        total_users.columns = ['platform', 'total_users']

        platform_conv = platform_conv.merge(total_users, on='platform', how='left')
        platform_conv['cvr'] = platform_conv['conversions'] / platform_conv['total_users'] * 100

        return platform_conv

    def device_attribution(self) -> pd.DataFrame:
        """Attribute conversions by device family."""
        conversions = self.df[self.df['event_type'] == 'checkout_completed']

        if conversions.empty or 'device_family' not in conversions.columns:
            return pd.DataFrame()

        device_conv = conversions.groupby('device_family').agg({
            'amplitude_id': 'count'
        }).reset_index()
        device_conv.columns = ['device_family', 'conversions']
        device_conv['conversion_pct'] = device_conv['conversions'] / device_conv['conversions'].sum() * 100

        return device_conv.sort_values('conversions', ascending=False).head(10)

    def customer_status_attribution(self) -> pd.DataFrame:
        """Attribute conversions by customer status (new vs returning)."""
        # Get homepage views with customer status
        homepage = self.df[self.df['event_type'] == 'homepage_viewed'].copy()

        if homepage.empty or 'event_properties' not in homepage.columns:
            return pd.DataFrame()

        # Parse customer status
        def get_customer_status(props):
            if pd.isna(props):
                return 'unknown'
            try:
                if isinstance(props, str):
                    props = json.loads(props)
                return props.get('customer_status', 'unknown')
            except:
                return 'unknown'

        homepage['customer_status'] = homepage['event_properties'].apply(get_customer_status)

        # Get converting users
        converting_users = set(
            self.df[self.df['event_type'] == 'checkout_completed']['amplitude_id'].unique()
        )

        homepage['converted'] = homepage['amplitude_id'].isin(converting_users)

        # Group by customer status
        status_stats = homepage.groupby('customer_status').agg({
            'amplitude_id': 'nunique',
            'converted': 'sum'
        }).reset_index()
        status_stats.columns = ['customer_status', 'users', 'conversions']
        status_stats['cvr'] = status_stats['conversions'] / status_stats['users'] * 100

        return status_stats
