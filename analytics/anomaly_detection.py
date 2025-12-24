"""Anomaly Detection.

Detection of unusual patterns in orders, user behavior, and merchant performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import json
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies in user behavior, orders, and merchant performance."""

    def __init__(self, df: pd.DataFrame, sensitivity: float = 2.0):
        """
        Args:
            df: Event data DataFrame
            sensitivity: Z-score threshold for anomaly detection (default 2.0 = ~95%)
        """
        self.df = df.copy()
        self.sensitivity = sensitivity
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for anomaly detection."""
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df['date'] = self.df['event_time'].dt.date
        self.df['hour'] = self.df['event_time'].dt.hour

        # Parse event properties
        def parse_props(props):
            if isinstance(props, str):
                try:
                    return json.loads(props)
                except:
                    return {}
            return props if isinstance(props, dict) else {}

        if 'event_properties' in self.df.columns:
            self.df['props'] = self.df['event_properties'].apply(parse_props)
        else:
            self.df['props'] = [{}] * len(self.df)

        logger.info(f"Prepared {len(self.df):,} events for anomaly detection")

    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        if len(series) < 3:
            return pd.Series([False] * len(series))

        z_scores = np.abs(stats.zscore(series.fillna(series.mean())))
        return z_scores > self.sensitivity

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        if len(series) < 3:
            return pd.Series([False] * len(series))

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (series < lower_bound) | (series > upper_bound)

    def detect_volume_anomalies(self) -> pd.DataFrame:
        """Detect unusual order/event volume patterns."""
        logger.info("Detecting volume anomalies...")

        # Hourly volume
        hourly_volume = self.df.groupby(
            self.df['event_time'].dt.floor('H')
        ).agg({
            'amplitude_id': 'count'
        }).reset_index()
        hourly_volume.columns = ['datetime', 'events']

        # Detect anomalies
        hourly_volume['is_anomaly'] = self._detect_outliers_zscore(hourly_volume['events'])
        hourly_volume['z_score'] = np.abs(stats.zscore(hourly_volume['events'].fillna(0)))

        # Expected range
        mean_events = hourly_volume['events'].mean()
        std_events = hourly_volume['events'].std()
        hourly_volume['expected_min'] = max(0, mean_events - self.sensitivity * std_events)
        hourly_volume['expected_max'] = mean_events + self.sensitivity * std_events

        # Anomaly type
        hourly_volume['anomaly_type'] = hourly_volume.apply(
            lambda x: 'spike' if x['is_anomaly'] and x['events'] > x['expected_max']
            else ('drop' if x['is_anomaly'] and x['events'] < x['expected_min'] else 'normal'),
            axis=1
        )

        anomalies = hourly_volume[hourly_volume['is_anomaly']].copy()
        anomalies['hour'] = anomalies['datetime'].dt.hour
        anomalies['date'] = anomalies['datetime'].dt.date

        return anomalies[['datetime', 'date', 'hour', 'events', 'z_score', 'anomaly_type', 'expected_min', 'expected_max']]

    def detect_user_behavior_anomalies(self) -> pd.DataFrame:
        """Detect unusual user behavior patterns (potential fraud signals)."""
        logger.info("Detecting user behavior anomalies...")

        # Calculate user-level metrics
        user_stats = self.df.groupby('amplitude_id').agg({
            'event_time': ['count', 'min', 'max'],
            'event_type': lambda x: x.nunique()
        }).reset_index()
        user_stats.columns = ['amplitude_id', 'total_events', 'first_event', 'last_event', 'unique_event_types']

        # Calculate derived metrics
        user_stats['session_duration_hours'] = (
            (user_stats['last_event'] - user_stats['first_event']).dt.total_seconds() / 3600
        )
        user_stats['events_per_hour'] = user_stats['total_events'] / np.maximum(user_stats['session_duration_hours'], 0.1)

        # Checkouts per user
        checkout_counts = self.df[self.df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()
        user_stats['checkouts'] = user_stats['amplitude_id'].map(checkout_counts).fillna(0)

        # Detect anomalies in each metric
        anomaly_flags = pd.DataFrame()
        anomaly_flags['amplitude_id'] = user_stats['amplitude_id']

        # Flag 1: Extremely high event rate
        anomaly_flags['high_event_rate'] = self._detect_outliers_iqr(user_stats['events_per_hour'])

        # Flag 2: Unusually high number of checkouts
        anomaly_flags['high_checkouts'] = self._detect_outliers_iqr(user_stats['checkouts'])

        # Flag 3: Very short session with many actions
        fast_heavy = (user_stats['session_duration_hours'] < 0.1) & (user_stats['total_events'] > 50)
        anomaly_flags['fast_heavy_usage'] = fast_heavy

        # Flag 4: Only checkout events (skip browsing - potential automation)
        checkout_only = self.df.groupby('amplitude_id')['event_type'].apply(
            lambda x: (x == 'checkout_completed').sum() / len(x) > 0.5
        )
        anomaly_flags['checkout_heavy'] = anomaly_flags['amplitude_id'].map(checkout_only).fillna(False)

        # Combine flags
        anomaly_flags['total_flags'] = (
            anomaly_flags['high_event_rate'].astype(int) +
            anomaly_flags['high_checkouts'].astype(int) +
            anomaly_flags['fast_heavy_usage'].astype(int) +
            anomaly_flags['checkout_heavy'].astype(int)
        )

        anomaly_flags['is_suspicious'] = anomaly_flags['total_flags'] >= 2

        # Get suspicious users with context
        suspicious = anomaly_flags[anomaly_flags['is_suspicious']].merge(
            user_stats, on='amplitude_id'
        )

        suspicious['flags'] = suspicious.apply(
            lambda x: ', '.join([
                f for f, v in [
                    ('High Event Rate', x['high_event_rate']),
                    ('High Checkouts', x['high_checkouts']),
                    ('Fast Heavy Usage', x['fast_heavy_usage']),
                    ('Checkout Heavy', x['checkout_heavy'])
                ] if v
            ]),
            axis=1
        )

        return suspicious[['amplitude_id', 'total_events', 'checkouts', 'events_per_hour',
                          'session_duration_hours', 'total_flags', 'flags']]

    def detect_merchant_anomalies(self) -> pd.DataFrame:
        """Detect anomalies in merchant performance."""
        logger.info("Detecting merchant anomalies...")

        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        # Extract merchant info
        if 'event_data_merchant_name' in checkouts.columns:
            checkouts['merchant_name'] = checkouts['event_data_merchant_name']
        else:
            checkouts['merchant_name'] = checkouts['props'].apply(lambda x: x.get('merchant_name', 'Unknown'))

        checkouts = checkouts[checkouts['merchant_name'] != 'Unknown']

        if len(checkouts) == 0:
            return pd.DataFrame({'message': ['No merchant data available']})

        # Daily orders per merchant
        daily_merchant = checkouts.groupby(['date', 'merchant_name']).size().reset_index(name='orders')

        # Calculate merchant baseline
        merchant_baseline = daily_merchant.groupby('merchant_name').agg({
            'orders': ['mean', 'std', 'count']
        }).reset_index()
        merchant_baseline.columns = ['merchant_name', 'avg_daily_orders', 'std_orders', 'days_active']

        # Only analyze merchants with enough history
        merchant_baseline = merchant_baseline[merchant_baseline['days_active'] >= 3]

        # Detect daily anomalies
        daily_merchant = daily_merchant.merge(merchant_baseline, on='merchant_name')

        daily_merchant['z_score'] = (
            (daily_merchant['orders'] - daily_merchant['avg_daily_orders']) /
            np.maximum(daily_merchant['std_orders'], 1)
        )

        daily_merchant['is_anomaly'] = np.abs(daily_merchant['z_score']) > self.sensitivity
        daily_merchant['anomaly_type'] = daily_merchant.apply(
            lambda x: 'surge' if x['z_score'] > self.sensitivity
            else ('drop' if x['z_score'] < -self.sensitivity else 'normal'),
            axis=1
        )

        anomalies = daily_merchant[daily_merchant['is_anomaly']].copy()

        return anomalies[['date', 'merchant_name', 'orders', 'avg_daily_orders', 'z_score', 'anomaly_type']]

    def detect_conversion_anomalies(self) -> pd.DataFrame:
        """Detect anomalies in conversion rates."""
        logger.info("Detecting conversion anomalies...")

        # Hourly conversion rates
        hourly = self.df.groupby(self.df['event_time'].dt.floor('H')).agg({
            'amplitude_id': 'nunique'
        }).reset_index()
        hourly.columns = ['datetime', 'unique_users']

        # Checkout users per hour
        checkout_hourly = self.df[self.df['event_type'] == 'checkout_completed'].groupby(
            self.df[self.df['event_type'] == 'checkout_completed']['event_time'].dt.floor('H')
        )['amplitude_id'].nunique().reset_index()
        checkout_hourly.columns = ['datetime', 'checkout_users']

        hourly = hourly.merge(checkout_hourly, on='datetime', how='left')
        hourly['checkout_users'] = hourly['checkout_users'].fillna(0)
        hourly['conversion_rate'] = (hourly['checkout_users'] / hourly['unique_users'] * 100).round(2)

        # Detect anomalies in conversion rate
        hourly['is_anomaly'] = self._detect_outliers_zscore(hourly['conversion_rate'])
        hourly['z_score'] = np.abs(stats.zscore(hourly['conversion_rate'].fillna(0)))

        mean_cvr = hourly['conversion_rate'].mean()
        hourly['expected_cvr'] = mean_cvr
        hourly['anomaly_type'] = hourly.apply(
            lambda x: 'high' if x['is_anomaly'] and x['conversion_rate'] > mean_cvr
            else ('low' if x['is_anomaly'] and x['conversion_rate'] < mean_cvr else 'normal'),
            axis=1
        )

        anomalies = hourly[hourly['is_anomaly']].copy()

        return anomalies[['datetime', 'unique_users', 'checkout_users', 'conversion_rate',
                         'expected_cvr', 'z_score', 'anomaly_type']]

    def detect_payment_anomalies(self) -> pd.DataFrame:
        """Detect anomalies in payment patterns."""
        logger.info("Detecting payment anomalies...")

        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        if len(checkouts) == 0:
            return pd.DataFrame({'message': ['No checkout data available']})

        # Extract payment method
        checkouts['payment_method'] = checkouts['props'].apply(lambda x: x.get('payment_method', 'unknown'))

        # Payment method distribution by hour
        payment_hourly = checkouts.groupby(['date', 'payment_method']).size().reset_index(name='count')

        # Overall payment method shares
        payment_shares = checkouts['payment_method'].value_counts(normalize=True)

        # Detect if any payment method share is unusual
        daily_shares = checkouts.groupby(['date', 'payment_method']).size().unstack(fill_value=0)
        daily_shares = daily_shares.div(daily_shares.sum(axis=1), axis=0)

        anomalies = []
        for payment_method in daily_shares.columns:
            method_series = daily_shares[payment_method]
            is_anomaly = self._detect_outliers_zscore(method_series)

            for date, is_anom in zip(daily_shares.index, is_anomaly):
                if is_anom:
                    anomalies.append({
                        'date': date,
                        'payment_method': payment_method,
                        'share': method_series[date] * 100,
                        'expected_share': payment_shares.get(payment_method, 0) * 100,
                        'anomaly_type': 'high' if method_series[date] > payment_shares.get(payment_method, 0) else 'low'
                    })

        return pd.DataFrame(anomalies)

    def get_anomaly_summary(self) -> Dict:
        """Get summary of all detected anomalies."""
        volume = self.detect_volume_anomalies()
        user = self.detect_user_behavior_anomalies()
        merchant = self.detect_merchant_anomalies()
        conversion = self.detect_conversion_anomalies()

        return {
            'volume_anomalies': len(volume),
            'volume_spikes': len(volume[volume['anomaly_type'] == 'spike']) if 'anomaly_type' in volume.columns else 0,
            'volume_drops': len(volume[volume['anomaly_type'] == 'drop']) if 'anomaly_type' in volume.columns else 0,
            'suspicious_users': len(user) if 'amplitude_id' in user.columns else 0,
            'merchant_anomalies': len(merchant) if 'merchant_name' in merchant.columns else 0,
            'conversion_anomalies': len(conversion),
            'sensitivity': self.sensitivity
        }

    def get_alerts(self, severity_threshold: str = 'medium') -> List[Dict]:
        """Generate actionable alerts for detected anomalies.

        Args:
            severity_threshold: 'low', 'medium', or 'high'
        """
        alerts = []
        severity_levels = {'low': 1, 'medium': 2, 'high': 3}
        threshold = severity_levels.get(severity_threshold, 2)

        # Volume anomalies
        volume = self.detect_volume_anomalies()
        if len(volume) > 0 and 'anomaly_type' in volume.columns:
            spikes = volume[volume['anomaly_type'] == 'spike']
            drops = volume[volume['anomaly_type'] == 'drop']

            if len(spikes) > 0:
                max_spike = spikes.loc[spikes['events'].idxmax()]
                alerts.append({
                    'type': 'volume_spike',
                    'severity': 'medium',
                    'severity_level': 2,
                    'title': 'Traffic Spike Detected',
                    'message': f"Unusual traffic spike at {max_spike['datetime']}: {max_spike['events']:.0f} events (expected max: {max_spike['expected_max']:.0f})",
                    'action': 'Monitor system performance and capacity'
                })

            if len(drops) > 0:
                max_drop = drops.loc[drops['events'].idxmin()]
                alerts.append({
                    'type': 'volume_drop',
                    'severity': 'high',
                    'severity_level': 3,
                    'title': 'Traffic Drop Detected',
                    'message': f"Significant traffic drop at {max_drop['datetime']}: {max_drop['events']:.0f} events (expected min: {max_drop['expected_min']:.0f})",
                    'action': 'Investigate potential outage or tracking issues'
                })

        # Suspicious users
        user = self.detect_user_behavior_anomalies()
        if len(user) > 0 and 'amplitude_id' in user.columns:
            if len(user) > 10:
                alerts.append({
                    'type': 'suspicious_activity',
                    'severity': 'high',
                    'severity_level': 3,
                    'title': f'{len(user)} Suspicious Users Detected',
                    'message': f"Users showing potential bot/fraud behavior",
                    'action': 'Review user accounts and consider blocking'
                })
            elif len(user) > 0:
                alerts.append({
                    'type': 'suspicious_activity',
                    'severity': 'medium',
                    'severity_level': 2,
                    'title': f'{len(user)} Suspicious Users Detected',
                    'message': f"Users showing unusual behavior patterns",
                    'action': 'Monitor and investigate if pattern continues'
                })

        # Merchant anomalies
        merchant = self.detect_merchant_anomalies()
        if len(merchant) > 0 and 'merchant_name' in merchant.columns:
            drops = merchant[merchant['anomaly_type'] == 'drop']
            if len(drops) > 0:
                affected_merchants = drops['merchant_name'].unique()
                alerts.append({
                    'type': 'merchant_drop',
                    'severity': 'medium',
                    'severity_level': 2,
                    'title': f'{len(affected_merchants)} Merchants with Order Drops',
                    'message': f"Unusual order decline for: {', '.join(affected_merchants[:3])}",
                    'action': 'Check merchant status and availability'
                })

        # Conversion anomalies
        conversion = self.detect_conversion_anomalies()
        if len(conversion) > 0:
            low_cvr = conversion[conversion['anomaly_type'] == 'low']
            if len(low_cvr) > 0:
                alerts.append({
                    'type': 'low_conversion',
                    'severity': 'high',
                    'severity_level': 3,
                    'title': 'Low Conversion Rate Detected',
                    'message': f"{len(low_cvr)} periods with abnormally low conversion",
                    'action': 'Check checkout flow and payment systems'
                })

        # Filter by severity threshold
        filtered_alerts = [a for a in alerts if a['severity_level'] >= threshold]

        return sorted(filtered_alerts, key=lambda x: x['severity_level'], reverse=True)

    def export_anomalies_report(self) -> Dict[str, pd.DataFrame]:
        """Export all anomaly data for detailed analysis."""
        return {
            'volume_anomalies': self.detect_volume_anomalies(),
            'user_anomalies': self.detect_user_behavior_anomalies(),
            'merchant_anomalies': self.detect_merchant_anomalies(),
            'conversion_anomalies': self.detect_conversion_anomalies(),
            'payment_anomalies': self.detect_payment_anomalies()
        }
