"""Session & Funnel Analytics.

Deep analysis of user sessions, funnel drop-offs, and user journey paths.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class SessionAnalyzer:
    """Analyze user sessions and session-level metrics."""

    # Default session timeout (30 minutes of inactivity = new session)
    SESSION_TIMEOUT_MINUTES = 30

    def __init__(self, df: pd.DataFrame, session_timeout_minutes: int = 30):
        """
        Args:
            df: Event data DataFrame
            session_timeout_minutes: Minutes of inactivity to start new session
        """
        self.df = df.copy()
        self.session_timeout = session_timeout_minutes
        self._prepare_data()
        self._build_sessions()

    def _prepare_data(self):
        """Prepare data for session analysis."""
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df = self.df.sort_values(['amplitude_id', 'event_time'])
        logger.info(f"Prepared {len(self.df):,} events for session analysis")

    def _build_sessions(self):
        """Build session IDs based on time gaps."""
        logger.info("Building sessions...")

        # Calculate time since previous event for same user
        self.df['prev_time'] = self.df.groupby('amplitude_id')['event_time'].shift(1)
        self.df['time_gap'] = (self.df['event_time'] - self.df['prev_time']).dt.total_seconds() / 60

        # New session if gap > timeout or first event
        self.df['new_session'] = (
            self.df['time_gap'].isna() |
            (self.df['time_gap'] > self.session_timeout)
        ).astype(int)

        # Create session IDs
        self.df['session_id'] = self.df.groupby('amplitude_id')['new_session'].cumsum()
        self.df['session_key'] = self.df['amplitude_id'].astype(str) + '_' + self.df['session_id'].astype(str)

        # Build session summary
        self.sessions = self.df.groupby('session_key').agg({
            'amplitude_id': 'first',
            'event_time': ['min', 'max', 'count'],
            'event_type': lambda x: list(x),
            'platform': 'first'
        }).reset_index()
        self.sessions.columns = ['session_key', 'amplitude_id', 'session_start', 'session_end',
                                  'event_count', 'event_sequence', 'platform']

        # Session duration in minutes
        self.sessions['duration_minutes'] = (
            (self.sessions['session_end'] - self.sessions['session_start']).dt.total_seconds() / 60
        )

        # Entry and exit events
        self.sessions['entry_event'] = self.sessions['event_sequence'].apply(lambda x: x[0] if x else None)
        self.sessions['exit_event'] = self.sessions['event_sequence'].apply(lambda x: x[-1] if x else None)

        # Did session convert?
        self.sessions['converted'] = self.sessions['event_sequence'].apply(
            lambda x: 'checkout_completed' in x
        )

        # Bounce = single event session
        self.sessions['is_bounce'] = self.sessions['event_count'] == 1

        logger.info(f"Built {len(self.sessions):,} sessions from {self.sessions['amplitude_id'].nunique():,} users")

    def get_summary(self) -> Dict:
        """Get session summary statistics."""
        return {
            'total_sessions': len(self.sessions),
            'total_users': self.sessions['amplitude_id'].nunique(),
            'avg_sessions_per_user': len(self.sessions) / self.sessions['amplitude_id'].nunique(),
            'avg_session_duration': self.sessions['duration_minutes'].mean(),
            'median_session_duration': self.sessions['duration_minutes'].median(),
            'avg_events_per_session': self.sessions['event_count'].mean(),
            'bounce_rate': self.sessions['is_bounce'].mean() * 100,
            'conversion_rate': self.sessions['converted'].mean() * 100
        }

    def session_duration_distribution(self) -> pd.DataFrame:
        """Get session duration distribution."""
        # Bucket durations
        bins = [0, 1, 2, 5, 10, 15, 30, 60, float('inf')]
        labels = ['<1min', '1-2min', '2-5min', '5-10min', '10-15min', '15-30min', '30-60min', '60+min']

        self.sessions['duration_bucket'] = pd.cut(
            self.sessions['duration_minutes'],
            bins=bins,
            labels=labels
        )

        dist = self.sessions.groupby('duration_bucket').agg({
            'session_key': 'count',
            'converted': 'mean'
        }).reset_index()
        dist.columns = ['duration_bucket', 'session_count', 'conversion_rate']
        dist['conversion_rate'] = (dist['conversion_rate'] * 100).round(2)
        dist['pct_of_sessions'] = (dist['session_count'] / dist['session_count'].sum() * 100).round(1)

        return dist

    def bounce_rate_by_entry(self) -> pd.DataFrame:
        """Get bounce rate by entry event.

        Excludes post-conversion events (checkout_completed, order_delivered, etc.)
        since users leaving after these is expected behavior, not a bounce.
        """
        # Exclude post-conversion/completion events - leaving after these is expected
        post_conversion_events = [
            'checkout_completed',
            'order_delivered',
            'checkout_button_pressed',
            'payment_initiated',
            'order_info_page_viewed'
        ]

        # Filter to meaningful entry points only
        meaningful_sessions = self.sessions[
            ~self.sessions['entry_event'].isin(post_conversion_events)
        ]

        if len(meaningful_sessions) == 0:
            return pd.DataFrame(columns=['entry_event', 'sessions', 'bounce_rate', 'conversion_rate'])

        bounce_by_entry = meaningful_sessions.groupby('entry_event').agg({
            'session_key': 'count',
            'is_bounce': 'mean',
            'converted': 'mean'
        }).reset_index()
        bounce_by_entry.columns = ['entry_event', 'sessions', 'bounce_rate', 'conversion_rate']
        bounce_by_entry['bounce_rate'] = (bounce_by_entry['bounce_rate'] * 100).round(1)
        bounce_by_entry['conversion_rate'] = (bounce_by_entry['conversion_rate'] * 100).round(2)
        bounce_by_entry = bounce_by_entry.sort_values('sessions', ascending=False)

        return bounce_by_entry

    def bounce_rate_by_platform(self) -> pd.DataFrame:
        """Get bounce rate by platform."""
        bounce_by_platform = self.sessions.groupby('platform').agg({
            'session_key': 'count',
            'is_bounce': 'mean',
            'converted': 'mean',
            'duration_minutes': 'mean'
        }).reset_index()
        bounce_by_platform.columns = ['platform', 'sessions', 'bounce_rate', 'conversion_rate', 'avg_duration']
        bounce_by_platform['bounce_rate'] = (bounce_by_platform['bounce_rate'] * 100).round(1)
        bounce_by_platform['conversion_rate'] = (bounce_by_platform['conversion_rate'] * 100).round(2)
        bounce_by_platform['avg_duration'] = bounce_by_platform['avg_duration'].round(1)

        return bounce_by_platform

    def sessions_by_hour(self) -> pd.DataFrame:
        """Get session metrics by hour of day."""
        self.sessions['hour'] = self.sessions['session_start'].dt.hour

        by_hour = self.sessions.groupby('hour').agg({
            'session_key': 'count',
            'is_bounce': 'mean',
            'converted': 'mean',
            'duration_minutes': 'mean'
        }).reset_index()
        by_hour.columns = ['hour', 'sessions', 'bounce_rate', 'conversion_rate', 'avg_duration']
        by_hour['bounce_rate'] = (by_hour['bounce_rate'] * 100).round(1)
        by_hour['conversion_rate'] = (by_hour['conversion_rate'] * 100).round(2)

        return by_hour

    def get_session_depth_analysis(self) -> pd.DataFrame:
        """Analyze how session depth (event count) affects conversion."""
        bins = [1, 2, 3, 5, 10, 20, float('inf')]
        labels = ['1 event', '2 events', '3-4 events', '5-9 events', '10-19 events', '20+ events']

        self.sessions['depth_bucket'] = pd.cut(
            self.sessions['event_count'],
            bins=bins,
            labels=labels
        )

        depth_analysis = self.sessions.groupby('depth_bucket').agg({
            'session_key': 'count',
            'converted': 'mean',
            'duration_minutes': 'mean'
        }).reset_index()
        depth_analysis.columns = ['session_depth', 'sessions', 'conversion_rate', 'avg_duration']
        depth_analysis['conversion_rate'] = (depth_analysis['conversion_rate'] * 100).round(2)
        depth_analysis['pct_of_sessions'] = (depth_analysis['sessions'] / depth_analysis['sessions'].sum() * 100).round(1)

        return depth_analysis


class FunnelAnalyzer:
    """Deep funnel analysis with drop-off insights."""

    # Standard e-commerce funnel
    DEFAULT_FUNNEL = [
        'homepage_viewed',
        'product_page_viewed',
        'product_added',
        'cart_page_viewed',
        'checkout_button_pressed',
        'payment_initiated',
        'checkout_completed'
    ]

    def __init__(self, df: pd.DataFrame, funnel_steps: Optional[List[str]] = None):
        """
        Args:
            df: Event data DataFrame
            funnel_steps: Custom funnel steps (uses default if None)
        """
        self.df = df.copy()
        self.funnel_steps = funnel_steps or self.DEFAULT_FUNNEL
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for funnel analysis."""
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        logger.info(f"Analyzing funnel with {len(self.funnel_steps)} steps")

    def get_funnel(self) -> pd.DataFrame:
        """Get basic funnel metrics."""
        funnel_data = []

        for i, step in enumerate(self.funnel_steps):
            users = self.df[self.df['event_type'] == step]['amplitude_id'].nunique()
            events = len(self.df[self.df['event_type'] == step])

            funnel_data.append({
                'step_number': i + 1,
                'step': step,
                'users': users,
                'events': events
            })

        funnel_df = pd.DataFrame(funnel_data)

        # Calculate rates
        first_step_users = funnel_df.iloc[0]['users']
        funnel_df['pct_of_total'] = (funnel_df['users'] / first_step_users * 100).round(2)

        # Step-to-step conversion
        funnel_df['prev_users'] = funnel_df['users'].shift(1)
        funnel_df['step_conversion'] = (funnel_df['users'] / funnel_df['prev_users'] * 100).round(2)
        funnel_df['step_conversion'] = funnel_df['step_conversion'].fillna(100)

        # Drop-off
        funnel_df['drop_off'] = funnel_df['prev_users'] - funnel_df['users']
        funnel_df['drop_off'] = funnel_df['drop_off'].fillna(0).astype(int)
        funnel_df['drop_off_rate'] = (100 - funnel_df['step_conversion']).round(2)
        funnel_df.loc[funnel_df.index[0], 'drop_off_rate'] = 0

        return funnel_df[['step_number', 'step', 'users', 'events', 'pct_of_total',
                          'step_conversion', 'drop_off', 'drop_off_rate']]

    def get_funnel_by_segment(self, segment_col: str) -> pd.DataFrame:
        """Get funnel broken down by a segment (e.g., platform)."""
        segments = self.df[segment_col].dropna().unique()

        all_funnels = []
        for segment in segments:
            segment_df = self.df[self.df[segment_col] == segment]

            for i, step in enumerate(self.funnel_steps):
                users = segment_df[segment_df['event_type'] == step]['amplitude_id'].nunique()
                all_funnels.append({
                    'segment': segment,
                    'step_number': i + 1,
                    'step': step,
                    'users': users
                })

        result = pd.DataFrame(all_funnels)

        # Calculate pct of total within each segment
        first_step = result[result['step_number'] == 1][['segment', 'users']].rename(columns={'users': 'first_step_users'})
        result = result.merge(first_step, on='segment')
        result['pct_of_total'] = (result['users'] / result['first_step_users'] * 100).round(2)

        return result[['segment', 'step_number', 'step', 'users', 'pct_of_total']]

    def get_drop_off_analysis(self) -> pd.DataFrame:
        """Detailed drop-off analysis between each step."""
        drop_offs = []

        for i in range(len(self.funnel_steps) - 1):
            current_step = self.funnel_steps[i]
            next_step = self.funnel_steps[i + 1]

            current_users = set(self.df[self.df['event_type'] == current_step]['amplitude_id'])
            next_users = set(self.df[self.df['event_type'] == next_step]['amplitude_id'])

            continued = current_users & next_users
            dropped = current_users - next_users

            # Where did dropped users go instead?
            dropped_df = self.df[self.df['amplitude_id'].isin(dropped)]

            # Get their next events after the current step
            dropped_events = dropped_df.groupby('amplitude_id').apply(
                lambda x: x[x['event_type'] != current_step]['event_type'].tolist()
            )

            # Count alternative paths
            all_next_events = []
            for events in dropped_events:
                if events:
                    all_next_events.append(events[0])

            next_event_counts = Counter(all_next_events)
            top_alternatives = next_event_counts.most_common(3)

            drop_offs.append({
                'from_step': current_step,
                'to_step': next_step,
                'users_at_step': len(current_users),
                'continued': len(continued),
                'dropped': len(dropped),
                'drop_rate': round(len(dropped) / len(current_users) * 100, 1) if current_users else 0,
                'top_alternative_1': top_alternatives[0][0] if len(top_alternatives) > 0 else None,
                'alt_1_count': top_alternatives[0][1] if len(top_alternatives) > 0 else 0,
                'top_alternative_2': top_alternatives[1][0] if len(top_alternatives) > 1 else None,
                'alt_2_count': top_alternatives[1][1] if len(top_alternatives) > 1 else 0
            })

        return pd.DataFrame(drop_offs)

    def get_time_between_steps(self) -> pd.DataFrame:
        """Calculate average time between funnel steps."""
        time_data = []

        for i in range(len(self.funnel_steps) - 1):
            current_step = self.funnel_steps[i]
            next_step = self.funnel_steps[i + 1]

            # Get users who did both steps
            current_times = self.df[self.df['event_type'] == current_step][['amplitude_id', 'event_time']]
            current_times = current_times.groupby('amplitude_id')['event_time'].first().reset_index()
            current_times.columns = ['amplitude_id', 'current_time']

            next_times = self.df[self.df['event_type'] == next_step][['amplitude_id', 'event_time']]
            next_times = next_times.groupby('amplitude_id')['event_time'].first().reset_index()
            next_times.columns = ['amplitude_id', 'next_time']

            merged = current_times.merge(next_times, on='amplitude_id')
            merged['time_diff'] = (merged['next_time'] - merged['current_time']).dt.total_seconds() / 60

            # Filter to reasonable times (positive and less than 24 hours)
            merged = merged[(merged['time_diff'] > 0) & (merged['time_diff'] < 1440)]

            if len(merged) > 0:
                time_data.append({
                    'from_step': current_step,
                    'to_step': next_step,
                    'users': len(merged),
                    'avg_minutes': merged['time_diff'].mean(),
                    'median_minutes': merged['time_diff'].median(),
                    'p75_minutes': merged['time_diff'].quantile(0.75)
                })

        result = pd.DataFrame(time_data)
        for col in ['avg_minutes', 'median_minutes', 'p75_minutes']:
            if col in result.columns:
                result[col] = result[col].round(2)

        return result

    def get_biggest_leaks(self) -> pd.DataFrame:
        """Identify the biggest conversion leaks in the funnel."""
        funnel = self.get_funnel()

        # Calculate absolute impact of each drop-off
        funnel['impact_score'] = funnel['drop_off'] * funnel['drop_off_rate'] / 100

        leaks = funnel[funnel['drop_off'] > 0][['step', 'drop_off', 'drop_off_rate', 'impact_score']]
        leaks = leaks.sort_values('impact_score', ascending=False)
        leaks.columns = ['after_step', 'users_lost', 'drop_rate_pct', 'impact_score']

        # Add recommendations
        recommendations = {
            'homepage_viewed': 'Improve homepage engagement, clearer CTAs',
            'product_page_viewed': 'Better product discovery, personalization',
            'product_added': 'Reduce friction in add-to-cart, show benefits',
            'cart_page_viewed': 'Address cart abandonment (pricing, fees)',
            'checkout_button_pressed': 'Simplify checkout flow',
            'payment_initiated': 'Multiple payment options, trust signals'
        }

        leaks['recommendation'] = leaks['after_step'].map(recommendations)

        return leaks


class PathAnalyzer:
    """Analyze user journey paths through the app."""

    def __init__(self, df: pd.DataFrame, max_path_length: int = 10):
        """
        Args:
            df: Event data DataFrame
            max_path_length: Maximum events to consider in a path
        """
        self.df = df.copy()
        self.max_path_length = max_path_length
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for path analysis."""
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df = self.df.sort_values(['amplitude_id', 'event_time'])

        # Build paths per user (truncated to max length)
        self.user_paths = self.df.groupby('amplitude_id').apply(
            lambda x: list(x['event_type'].head(self.max_path_length))
        ).reset_index()
        self.user_paths.columns = ['amplitude_id', 'path']

        logger.info(f"Built paths for {len(self.user_paths):,} users")

    def get_common_paths(self, n: int = 20, min_length: int = 2) -> pd.DataFrame:
        """Get most common user paths."""
        # Filter by min length and convert to tuple for counting
        paths = self.user_paths[self.user_paths['path'].apply(len) >= min_length]['path']
        paths = paths.apply(lambda x: ' -> '.join(x[:5]))  # Limit display length

        path_counts = paths.value_counts().head(n).reset_index()
        path_counts.columns = ['path', 'users']
        path_counts['pct'] = (path_counts['users'] / len(self.user_paths) * 100).round(2)

        return path_counts

    def get_paths_to_conversion(self, n: int = 20) -> pd.DataFrame:
        """Get most common paths that lead to conversion."""
        # Users who converted
        converters = set(self.df[self.df['event_type'] == 'checkout_completed']['amplitude_id'])

        converter_paths = self.user_paths[self.user_paths['amplitude_id'].isin(converters)]

        # Get path up to conversion
        def path_to_checkout(path):
            if 'checkout_completed' in path:
                idx = path.index('checkout_completed')
                return ' -> '.join(path[:idx+1])
            return ' -> '.join(path)

        converter_paths = converter_paths.copy()
        converter_paths['path_str'] = converter_paths['path'].apply(path_to_checkout)

        path_counts = converter_paths['path_str'].value_counts().head(n).reset_index()
        path_counts.columns = ['conversion_path', 'users']
        path_counts['pct_of_converters'] = (path_counts['users'] / len(converters) * 100).round(2)

        return path_counts

    def get_paths_to_drop_off(self, drop_after: str = 'product_added', n: int = 15) -> pd.DataFrame:
        """Get paths of users who dropped off after a specific event."""
        # Users who did the event but didn't convert
        did_event = set(self.df[self.df['event_type'] == drop_after]['amplitude_id'])
        converted = set(self.df[self.df['event_type'] == 'checkout_completed']['amplitude_id'])
        dropped = did_event - converted

        dropped_paths = self.user_paths[self.user_paths['amplitude_id'].isin(dropped)]

        # Get what happened after the drop_after event
        def get_after_event(path):
            if drop_after in path:
                idx = path.index(drop_after)
                after = path[idx+1:idx+4]  # Next 3 events
                if after:
                    return ' -> '.join(after)
                return 'EXIT'
            return 'N/A'

        dropped_paths = dropped_paths.copy()
        dropped_paths['after_path'] = dropped_paths['path'].apply(get_after_event)

        path_counts = dropped_paths['after_path'].value_counts().head(n).reset_index()
        path_counts.columns = [f'after_{drop_after}', 'users']
        path_counts['pct'] = (path_counts['users'] / len(dropped) * 100).round(2)

        return path_counts

    def get_entry_points(self) -> pd.DataFrame:
        """Analyze entry points and their conversion rates."""
        # First event per user
        first_events = self.df.groupby('amplitude_id').first().reset_index()

        # Check if user converted
        converters = set(self.df[self.df['event_type'] == 'checkout_completed']['amplitude_id'])
        first_events['converted'] = first_events['amplitude_id'].isin(converters)

        entry_stats = first_events.groupby('event_type').agg({
            'amplitude_id': 'count',
            'converted': 'mean'
        }).reset_index()
        entry_stats.columns = ['entry_event', 'users', 'conversion_rate']
        entry_stats['conversion_rate'] = (entry_stats['conversion_rate'] * 100).round(2)
        entry_stats['pct_of_users'] = (entry_stats['users'] / entry_stats['users'].sum() * 100).round(1)
        entry_stats = entry_stats.sort_values('users', ascending=False)

        return entry_stats

    def get_exit_points(self) -> pd.DataFrame:
        """Analyze exit points (last event before leaving)."""
        # Last event per user
        last_events = self.df.groupby('amplitude_id').last().reset_index()

        exit_stats = last_events.groupby('event_type').agg({
            'amplitude_id': 'count'
        }).reset_index()
        exit_stats.columns = ['exit_event', 'users']
        exit_stats['pct_of_users'] = (exit_stats['users'] / exit_stats['users'].sum() * 100).round(1)
        exit_stats = exit_stats.sort_values('users', ascending=False)

        return exit_stats

    def get_event_flow(self) -> pd.DataFrame:
        """Get event transition probabilities (what event comes after each event)."""
        # Create next event column
        self.df['next_event'] = self.df.groupby('amplitude_id')['event_type'].shift(-1)

        # Count transitions
        transitions = self.df[self.df['next_event'].notna()].groupby(
            ['event_type', 'next_event']
        ).size().reset_index(name='count')

        # Calculate probability
        totals = transitions.groupby('event_type')['count'].sum().reset_index(name='total')
        transitions = transitions.merge(totals, on='event_type')
        transitions['probability'] = (transitions['count'] / transitions['total'] * 100).round(1)

        # Get top 3 next events for each event
        top_transitions = transitions.sort_values(
            ['event_type', 'probability'], ascending=[True, False]
        ).groupby('event_type').head(3)

        return top_transitions[['event_type', 'next_event', 'count', 'probability']]
