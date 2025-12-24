"""
Customer Journey Mapping
========================
Visualize user paths, find friction points, and understand conversion flows.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import timedelta


class CustomerJourneyMapper:
    """Analyze and visualize customer journeys through the app."""

    # Define the ideal funnel order
    FUNNEL_STAGES = [
        'homepage_viewed',
        'category_page_viewed',
        'merchant_page_viewed',
        'product_page_viewed',
        'product_added',
        'cart_page_viewed',
        'checkout_button_pressed',
        'payment_initiated',
        'checkout_completed',
        'order_delivered'
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for journey analysis."""
        if 'event_time' in self.df.columns:
            self.df['event_time'] = pd.to_datetime(self.df['event_time'])

        # Sort by user and time
        self.df = self.df.sort_values(['amplitude_id', 'event_time'])

    def extract_sessions(self, session_timeout_minutes: int = 30) -> pd.DataFrame:
        """Extract user sessions based on timeout."""
        df = self.df.copy()

        # Calculate time difference between events
        df['time_diff'] = df.groupby('amplitude_id')['event_time'].diff()

        # New session if gap > timeout
        timeout = timedelta(minutes=session_timeout_minutes)
        df['new_session'] = (df['time_diff'] > timeout) | (df['time_diff'].isna())

        # Assign session IDs
        df['session_id'] = df.groupby('amplitude_id')['new_session'].cumsum()
        df['session_key'] = df['amplitude_id'].astype(str) + '_' + df['session_id'].astype(str)

        return df

    def get_journey_sequences(self, max_length: int = 10) -> pd.DataFrame:
        """Get event sequences for each session."""
        df = self.extract_sessions()

        # Build sequences per session
        sequences = df.groupby('session_key').agg({
            'amplitude_id': 'first',
            'event_type': lambda x: list(x)[:max_length],
            'event_time': ['min', 'max'],
            'platform': 'first'
        }).reset_index()

        sequences.columns = ['session_key', 'user_id', 'journey', 'start_time', 'end_time', 'platform']
        sequences['journey_length'] = sequences['journey'].apply(len)
        sequences['journey_str'] = sequences['journey'].apply(lambda x: ' → '.join(x))

        # Check if converted
        sequences['converted'] = sequences['journey'].apply(
            lambda x: 'checkout_completed' in x
        )

        return sequences

    def get_top_journeys(self, n: int = 20, min_occurrences: int = 10) -> pd.DataFrame:
        """Get most common journey patterns."""
        sequences = self.get_journey_sequences()

        # Count journey patterns
        journey_counts = sequences.groupby('journey_str').agg({
            'session_key': 'count',
            'converted': 'sum',
            'journey_length': 'first'
        }).reset_index()

        journey_counts.columns = ['journey', 'occurrences', 'conversions', 'length']
        journey_counts['conversion_rate'] = journey_counts['conversions'] / journey_counts['occurrences']

        # Filter and sort
        journey_counts = journey_counts[journey_counts['occurrences'] >= min_occurrences]
        journey_counts = journey_counts.sort_values('occurrences', ascending=False).head(n)

        return journey_counts

    def get_flow_data(self) -> Dict:
        """Get data for Sankey diagram visualization."""
        df = self.extract_sessions()

        # Get transitions between events
        df['next_event'] = df.groupby('session_key')['event_type'].shift(-1)

        # Count transitions
        transitions = df[df['next_event'].notna()].groupby(
            ['event_type', 'next_event']
        ).size().reset_index(name='count')

        # Filter to significant transitions
        min_count = transitions['count'].quantile(0.1)
        transitions = transitions[transitions['count'] >= min_count]

        # Build node list
        all_events = list(set(transitions['event_type'].tolist() + transitions['next_event'].tolist()))

        # Order nodes by funnel stage
        def get_stage_order(event):
            if event in self.FUNNEL_STAGES:
                return self.FUNNEL_STAGES.index(event)
            return 100

        all_events = sorted(all_events, key=get_stage_order)
        event_to_idx = {event: idx for idx, event in enumerate(all_events)}

        # Build links
        links = {
            'source': [event_to_idx[row['event_type']] for _, row in transitions.iterrows()],
            'target': [event_to_idx[row['next_event']] for _, row in transitions.iterrows()],
            'value': transitions['count'].tolist()
        }

        return {
            'nodes': all_events,
            'links': links,
            'transitions': transitions
        }

    def get_drop_off_analysis(self) -> pd.DataFrame:
        """Analyze where users drop off in the funnel."""
        sequences = self.get_journey_sequences()

        results = []

        for i, stage in enumerate(self.FUNNEL_STAGES):
            # Users who reached this stage
            reached = sequences[sequences['journey'].apply(lambda x: stage in x)]
            reached_count = len(reached)

            if i < len(self.FUNNEL_STAGES) - 1:
                next_stage = self.FUNNEL_STAGES[i + 1]
                # Users who continued to next stage
                continued = reached[reached['journey'].apply(lambda x: next_stage in x)]
                continued_count = len(continued)

                drop_off_rate = 1 - (continued_count / reached_count) if reached_count > 0 else 0
            else:
                continued_count = reached_count
                drop_off_rate = 0

            results.append({
                'stage': stage,
                'stage_order': i + 1,
                'users_reached': reached_count,
                'users_continued': continued_count,
                'drop_off_rate': drop_off_rate,
                'conversion_at_stage': reached[reached['converted']].shape[0] / reached_count if reached_count > 0 else 0
            })

        return pd.DataFrame(results)

    def get_path_comparison(self) -> Dict:
        """Compare paths of converters vs non-converters."""
        sequences = self.get_journey_sequences()

        converters = sequences[sequences['converted']]
        non_converters = sequences[~sequences['converted']]

        # Average journey length
        avg_length_converters = converters['journey_length'].mean()
        avg_length_non_converters = non_converters['journey_length'].mean()

        # Most common first events
        first_events_converters = Counter(converters['journey'].apply(lambda x: x[0] if x else None))
        first_events_non_converters = Counter(non_converters['journey'].apply(lambda x: x[0] if x else None))

        # Events that appear more in converter journeys
        all_events_converters = Counter([e for j in converters['journey'] for e in j])
        all_events_non_converters = Counter([e for j in non_converters['journey'] for e in j])

        # Normalize
        total_converters = sum(all_events_converters.values())
        total_non_converters = sum(all_events_non_converters.values())

        event_lift = {}
        for event in set(all_events_converters.keys()) | set(all_events_non_converters.keys()):
            rate_converters = all_events_converters.get(event, 0) / total_converters if total_converters > 0 else 0
            rate_non_converters = all_events_non_converters.get(event, 0) / total_non_converters if total_non_converters > 0 else 0

            if rate_non_converters > 0:
                event_lift[event] = rate_converters / rate_non_converters
            else:
                event_lift[event] = float('inf') if rate_converters > 0 else 1

        return {
            'avg_journey_length': {
                'converters': avg_length_converters,
                'non_converters': avg_length_non_converters
            },
            'first_events': {
                'converters': dict(first_events_converters.most_common(5)),
                'non_converters': dict(first_events_non_converters.most_common(5))
            },
            'event_lift': event_lift,
            'total_sessions': {
                'converters': len(converters),
                'non_converters': len(non_converters)
            }
        }

    def get_friction_points(self) -> pd.DataFrame:
        """Identify friction points where users struggle."""
        df = self.extract_sessions()

        # Calculate time spent before each event
        df['time_to_event'] = df.groupby('session_key')['event_time'].diff().dt.total_seconds()

        # Events that take unusually long (potential friction)
        event_times = df.groupby('event_type').agg({
            'time_to_event': ['mean', 'median', 'std', 'count']
        }).reset_index()
        event_times.columns = ['event_type', 'avg_time', 'median_time', 'std_time', 'occurrences']

        # Calculate friction score (high time + high variance = friction)
        event_times['friction_score'] = (
            event_times['avg_time'] / event_times['avg_time'].max() * 0.5 +
            event_times['std_time'] / event_times['std_time'].max() * 0.5
        )

        # Add drop-off context
        drop_off = self.get_drop_off_analysis()
        event_times = event_times.merge(
            drop_off[['stage', 'drop_off_rate']],
            left_on='event_type',
            right_on='stage',
            how='left'
        )

        # Combine into overall friction score
        event_times['overall_friction'] = event_times['friction_score'] * 0.5 + event_times['drop_off_rate'].fillna(0) * 0.5

        return event_times.sort_values('overall_friction', ascending=False)

    def get_loop_analysis(self) -> pd.DataFrame:
        """Find where users loop back (repeat events) - sign of confusion."""
        sequences = self.get_journey_sequences()

        loop_data = []

        for event in self.df['event_type'].unique():
            # Count sessions where this event appears more than once
            sequences['event_count'] = sequences['journey'].apply(lambda x: x.count(event))
            sessions_with_loops = sequences[sequences['event_count'] > 1]

            if len(sessions_with_loops) > 0:
                loop_data.append({
                    'event': event,
                    'sessions_with_loops': len(sessions_with_loops),
                    'total_sessions': len(sequences[sequences['journey'].apply(lambda x: event in x)]),
                    'loop_rate': len(sessions_with_loops) / len(sequences[sequences['journey'].apply(lambda x: event in x)]),
                    'avg_repeats': sessions_with_loops['event_count'].mean()
                })

        return pd.DataFrame(loop_data).sort_values('loop_rate', ascending=False)

    def get_time_between_stages(self) -> pd.DataFrame:
        """Calculate average time between funnel stages."""
        df = self.extract_sessions()

        results = []

        for i in range(len(self.FUNNEL_STAGES) - 1):
            from_stage = self.FUNNEL_STAGES[i]
            to_stage = self.FUNNEL_STAGES[i + 1]

            # Get sessions that have both events
            sessions = df.groupby('session_key').apply(
                lambda x: self._time_between_events(x, from_stage, to_stage)
            ).dropna()

            if len(sessions) > 0:
                results.append({
                    'from_stage': from_stage,
                    'to_stage': to_stage,
                    'avg_seconds': sessions.mean(),
                    'median_seconds': sessions.median(),
                    'sessions': len(sessions)
                })

        return pd.DataFrame(results)

    def _time_between_events(self, session_df, from_event, to_event) -> Optional[float]:
        """Calculate time between two events in a session."""
        from_rows = session_df[session_df['event_type'] == from_event]
        to_rows = session_df[session_df['event_type'] == to_event]

        if len(from_rows) == 0 or len(to_rows) == 0:
            return None

        from_time = from_rows['event_time'].iloc[0]
        to_time = to_rows[to_rows['event_time'] > from_time]['event_time']

        if len(to_time) == 0:
            return None

        return (to_time.iloc[0] - from_time).total_seconds()
