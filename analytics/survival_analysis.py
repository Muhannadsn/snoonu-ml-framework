"""
Survival Analysis
=================
Time-to-event modeling for churn, retention, and customer lifecycle.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats


class SurvivalAnalyzer:
    """
    Survival analysis for customer lifecycle events.
    Analyzes time-to-churn, time-to-second-order, and customer lifetime.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for survival analysis."""
        if 'event_time' in self.df.columns:
            self.df['event_time'] = pd.to_datetime(self.df['event_time'])

        self.observation_end = self.df['event_time'].max()
        self.observation_start = self.df['event_time'].min()

    def get_user_orders(self) -> pd.DataFrame:
        """Extract order history per user."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        if len(orders) == 0:
            return pd.DataFrame()

        user_orders = orders.groupby('amplitude_id').agg({
            'event_time': ['min', 'max', 'count']
        }).reset_index()
        user_orders.columns = ['user_id', 'first_order', 'last_order', 'total_orders']

        # Calculate days since last order
        user_orders['days_since_last'] = (self.observation_end - user_orders['last_order']).dt.days

        # Customer lifetime in days
        user_orders['lifetime_days'] = (user_orders['last_order'] - user_orders['first_order']).dt.days

        return user_orders

    def get_time_to_second_order(self) -> pd.DataFrame:
        """Analyze time between first and second order."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()
        orders = orders.sort_values(['amplitude_id', 'event_time'])

        # Get first and second order times
        orders['order_num'] = orders.groupby('amplitude_id').cumcount() + 1

        first_orders = orders[orders['order_num'] == 1][['amplitude_id', 'event_time']].rename(
            columns={'event_time': 'first_order_time'}
        )
        second_orders = orders[orders['order_num'] == 2][['amplitude_id', 'event_time']].rename(
            columns={'event_time': 'second_order_time'}
        )

        # Merge
        result = first_orders.merge(second_orders, on='amplitude_id', how='left')

        # Calculate time to second order
        result['time_to_second_order_days'] = (
            result['second_order_time'] - result['first_order_time']
        ).dt.days

        # For users without second order, mark as censored
        result['has_second_order'] = result['second_order_time'].notna()

        # For censored users, time is from first order to observation end
        result.loc[~result['has_second_order'], 'time_to_second_order_days'] = (
            self.observation_end - result.loc[~result['has_second_order'], 'first_order_time']
        ).dt.days

        return result

    def kaplan_meier_curve(self, event_type: str = 'churn', churn_days: int = 30) -> pd.DataFrame:
        """
        Calculate Kaplan-Meier survival curve.

        event_type: 'churn' for time-to-churn, 'second_order' for time-to-second-order
        """
        if event_type == 'churn':
            data = self._prepare_churn_data(churn_days)
            time_col = 'survival_time'
            event_col = 'churned'
        else:
            data = self.get_time_to_second_order()
            time_col = 'time_to_second_order_days'
            event_col = 'has_second_order'

        # Remove invalid times
        data = data[data[time_col] > 0].copy()

        # Get unique time points
        times = sorted(data[time_col].unique())

        results = []
        n_at_risk = len(data)
        survival_prob = 1.0

        for t in times:
            # Events at this time
            events_at_t = len(data[(data[time_col] == t) & (data[event_col])])
            censored_at_t = len(data[(data[time_col] == t) & (~data[event_col])])

            # Survival probability update
            if n_at_risk > 0:
                hazard = events_at_t / n_at_risk
                survival_prob = survival_prob * (1 - hazard)

            results.append({
                'time': t,
                'n_at_risk': n_at_risk,
                'events': events_at_t,
                'censored': censored_at_t,
                'survival_probability': survival_prob,
                'cumulative_hazard': -np.log(survival_prob) if survival_prob > 0 else np.inf
            })

            # Update n at risk
            n_at_risk = n_at_risk - events_at_t - censored_at_t

        return pd.DataFrame(results)

    def _prepare_churn_data(self, churn_days: int = 30) -> pd.DataFrame:
        """Prepare data for churn survival analysis."""
        user_orders = self.get_user_orders()

        if len(user_orders) == 0:
            return pd.DataFrame()

        # Define churn: no order in last N days
        user_orders['churned'] = user_orders['days_since_last'] >= churn_days

        # Survival time: days active (from first to last order) or until observation end
        user_orders['survival_time'] = np.where(
            user_orders['churned'],
            user_orders['lifetime_days'] + churn_days,  # Churned: lifetime + churn period
            user_orders['days_since_last']  # Active: days since last (censored)
        )

        return user_orders

    def get_hazard_by_period(self, period: str = 'week') -> pd.DataFrame:
        """Calculate hazard rate by time period after first order."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()
        orders = orders.sort_values(['amplitude_id', 'event_time'])

        # Get first order time per user
        first_orders = orders.groupby('amplitude_id')['event_time'].min().reset_index()
        first_orders.columns = ['amplitude_id', 'first_order_time']

        # Get all orders with time since first
        orders = orders.merge(first_orders, on='amplitude_id')
        orders['days_since_first'] = (orders['event_time'] - orders['first_order_time']).dt.days

        if period == 'week':
            orders['period'] = orders['days_since_first'] // 7
            period_label = 'Week'
        elif period == 'month':
            orders['period'] = orders['days_since_first'] // 30
            period_label = 'Month'
        else:
            orders['period'] = orders['days_since_first']
            period_label = 'Day'

        # Count active users per period
        user_activity = orders.groupby(['amplitude_id', 'period']).size().reset_index(name='orders')

        # For each period, count users still active
        periods = sorted(user_activity['period'].unique())

        results = []
        for p in periods:
            # Users who were active before or at this period
            users_at_start = user_activity[user_activity['period'] <= p]['amplitude_id'].nunique()

            # Users who are still active (have activity in this or later period)
            users_still_active = user_activity[user_activity['period'] >= p]['amplitude_id'].nunique()

            # Churn rate this period
            churned = users_at_start - users_still_active if p > 0 else 0
            hazard_rate = churned / users_at_start if users_at_start > 0 else 0

            results.append({
                'period': p,
                'period_label': f'{period_label} {p}',
                'users_at_start': users_at_start,
                'users_remaining': users_still_active,
                'churned': churned,
                'hazard_rate': hazard_rate,
                'retention_rate': 1 - hazard_rate
            })

        return pd.DataFrame(results)

    def get_cohort_survival(self, cohort_period: str = 'week') -> pd.DataFrame:
        """Calculate survival curves by cohort (when user made first order)."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()
        orders = orders.sort_values(['amplitude_id', 'event_time'])

        # Get first order per user (cohort assignment)
        first_orders = orders.groupby('amplitude_id')['event_time'].min().reset_index()
        first_orders.columns = ['amplitude_id', 'first_order_time']

        if cohort_period == 'week':
            first_orders['cohort'] = first_orders['first_order_time'].dt.isocalendar().week
            first_orders['cohort_label'] = 'Week ' + first_orders['cohort'].astype(str)
        else:
            first_orders['cohort'] = first_orders['first_order_time'].dt.month
            first_orders['cohort_label'] = first_orders['first_order_time'].dt.strftime('%b')

        # Get all orders with cohort info
        orders = orders.merge(first_orders, on='amplitude_id')
        orders['days_since_first'] = (orders['event_time'] - orders['first_order_time']).dt.days
        orders['week_since_first'] = orders['days_since_first'] // 7

        # Calculate retention by cohort and week
        cohorts = first_orders['cohort'].unique()

        results = []
        for cohort in sorted(cohorts)[:8]:  # Limit to 8 cohorts
            cohort_users = first_orders[first_orders['cohort'] == cohort]['amplitude_id'].unique()
            cohort_orders = orders[orders['amplitude_id'].isin(cohort_users)]
            cohort_label = first_orders[first_orders['cohort'] == cohort]['cohort_label'].iloc[0]

            initial_users = len(cohort_users)

            for week in range(13):  # 13 weeks
                active_users = cohort_orders[cohort_orders['week_since_first'] >= week]['amplitude_id'].nunique()
                survival_rate = active_users / initial_users if initial_users > 0 else 0

                results.append({
                    'cohort': cohort,
                    'cohort_label': cohort_label,
                    'week': week,
                    'initial_users': initial_users,
                    'active_users': active_users,
                    'survival_rate': survival_rate
                })

        return pd.DataFrame(results)

    def get_median_survival_time(self, event_type: str = 'churn', churn_days: int = 30) -> Dict:
        """Get median survival time and confidence interval."""
        km_curve = self.kaplan_meier_curve(event_type, churn_days)

        if len(km_curve) == 0:
            return {'median': None, 'lower_ci': None, 'upper_ci': None}

        # Find where survival probability crosses 0.5
        below_50 = km_curve[km_curve['survival_probability'] <= 0.5]

        if len(below_50) == 0:
            median = None  # Median not reached
        else:
            median = below_50['time'].iloc[0]

        # Approximate CI using Greenwood's formula (simplified)
        total_n = km_curve['n_at_risk'].iloc[0]
        se = np.sqrt(0.5 * 0.5 / total_n) if total_n > 0 else 0

        return {
            'median': median,
            'mean_survival': (km_curve['time'] * km_curve['survival_probability']).sum() / km_curve['survival_probability'].sum() if km_curve['survival_probability'].sum() > 0 else None,
            'survival_at_7_days': km_curve[km_curve['time'] >= 7]['survival_probability'].iloc[0] if len(km_curve[km_curve['time'] >= 7]) > 0 else None,
            'survival_at_30_days': km_curve[km_curve['time'] >= 30]['survival_probability'].iloc[0] if len(km_curve[km_curve['time'] >= 30]) > 0 else None,
            'survival_at_90_days': km_curve[km_curve['time'] >= 90]['survival_probability'].iloc[0] if len(km_curve[km_curve['time'] >= 90]) > 0 else None,
        }

    def get_risk_factors(self, churn_days: int = 30) -> pd.DataFrame:
        """Identify factors associated with higher churn risk."""
        churn_data = self._prepare_churn_data(churn_days)

        if len(churn_data) == 0:
            return pd.DataFrame()

        # Get user features
        user_features = self._get_user_features()

        # Merge with churn data
        analysis = churn_data.merge(user_features, left_on='user_id', right_on='amplitude_id', how='left')

        # Calculate churn rate by segment
        results = []

        # By platform
        if 'platform' in analysis.columns:
            platform_churn = analysis.groupby('platform').agg({
                'churned': ['sum', 'count']
            }).reset_index()
            platform_churn.columns = ['segment_value', 'churned', 'total']
            platform_churn['segment'] = 'platform'
            platform_churn['churn_rate'] = platform_churn['churned'] / platform_churn['total']
            results.append(platform_churn)

        # By order frequency
        if 'total_orders' in analysis.columns:
            analysis['order_bucket'] = pd.cut(
                analysis['total_orders'],
                bins=[0, 1, 2, 5, 10, float('inf')],
                labels=['1 order', '2 orders', '3-5 orders', '6-10 orders', '10+ orders']
            )
            freq_churn = analysis.groupby('order_bucket').agg({
                'churned': ['sum', 'count']
            }).reset_index()
            freq_churn.columns = ['segment_value', 'churned', 'total']
            freq_churn['segment'] = 'order_frequency'
            freq_churn['churn_rate'] = freq_churn['churned'] / freq_churn['total']
            results.append(freq_churn)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _get_user_features(self) -> pd.DataFrame:
        """Extract user features for risk analysis."""
        features = self.df.groupby('amplitude_id').agg({
            'platform': 'first',
            'event_type': 'count'
        }).reset_index()
        features.columns = ['amplitude_id', 'platform', 'total_events']

        return features

    def get_retention_curve(self, by: str = 'day') -> pd.DataFrame:
        """Simple retention curve showing % of users returning over time."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        # Get first order per user
        first_orders = orders.groupby('amplitude_id')['event_time'].min().reset_index()
        first_orders.columns = ['amplitude_id', 'first_order_time']

        # Merge to get days since first order
        orders = orders.merge(first_orders, on='amplitude_id')
        orders['days_since_first'] = (orders['event_time'] - orders['first_order_time']).dt.days

        initial_users = first_orders['amplitude_id'].nunique()

        results = []
        max_days = min(90, int(orders['days_since_first'].max()))

        for day in range(0, max_days + 1):
            if by == 'week' and day % 7 != 0:
                continue

            # Users who made any order on or after this day
            retained = orders[orders['days_since_first'] >= day]['amplitude_id'].nunique()
            retention = retained / initial_users if initial_users > 0 else 0

            results.append({
                'days_since_first_order': day,
                'users_retained': retained,
                'retention_rate': retention
            })

        return pd.DataFrame(results)
