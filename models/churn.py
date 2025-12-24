"""Churn Prediction Model.

Predicts which users are likely to stop ordering/using the app.

Business Use Cases:
- Target at-risk users with retention campaigns
- Prioritize customer service for high-value churning users
- Measure health of user base over time
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional
import logging

from .base import BasePredictor

logger = logging.getLogger(__name__)


class ChurnPredictor(BasePredictor):
    """Predict user churn based on behavioral patterns."""

    def __init__(self, config: Dict = None, model_type: str = 'random_forest'):
        super().__init__(config)
        self.model_type = model_type

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build churn prediction features from event data.

        Features capture:
        - Engagement patterns (sessions, events, time spent)
        - Purchase behavior (orders, cart, checkout)
        - Recency signals (days since last activity)
        - Platform/device patterns
        """
        logger.info("Building churn prediction features...")

        # Parse event properties if needed
        if 'event_properties' in df.columns and df['event_properties'].dtype == 'object':
            df = self._parse_event_properties(df)

        # Get reference date for recency calculations
        max_date = df['event_time'].max()

        # Group by user
        user_features = df.groupby('amplitude_id').agg({
            'event_time': ['min', 'max', 'count'],
            'event_type': 'nunique'
        }).reset_index()

        user_features.columns = ['amplitude_id', 'first_event', 'last_event',
                                  'total_events', 'unique_event_types']

        # Recency features
        user_features['days_since_last_event'] = (
            max_date - user_features['last_event']
        ).dt.total_seconds() / 86400

        user_features['account_age_days'] = (
            max_date - user_features['first_event']
        ).dt.total_seconds() / 86400

        user_features['active_days'] = df.groupby('amplitude_id')['event_time'].apply(
            lambda x: x.dt.date.nunique()
        ).values

        # Engagement intensity
        user_features['events_per_day'] = (
            user_features['total_events'] / user_features['active_days'].clip(lower=1)
        )

        # Session approximation (events within 30 min = same session)
        df_sorted = df.sort_values(['amplitude_id', 'event_time'])
        df_sorted['time_diff'] = df_sorted.groupby('amplitude_id')['event_time'].diff()
        df_sorted['new_session'] = df_sorted['time_diff'] > pd.Timedelta(minutes=30)
        sessions = df_sorted.groupby('amplitude_id')['new_session'].sum() + 1
        user_features['total_sessions'] = user_features['amplitude_id'].map(sessions).fillna(1)

        user_features['events_per_session'] = (
            user_features['total_events'] / user_features['total_sessions'].clip(lower=1)
        )

        # Funnel behavior
        funnel_events = {
            'homepage_views': 'homepage_viewed',
            'product_views': 'product_page_viewed',
            'cart_adds': 'product_added',
            'checkouts': 'checkout_completed',
            'orders_delivered': 'order_delivered'
        }

        for feature_name, event_type in funnel_events.items():
            counts = df[df['event_type'] == event_type].groupby('amplitude_id').size()
            user_features[feature_name] = user_features['amplitude_id'].map(counts).fillna(0)

        # Conversion indicators
        user_features['has_ordered'] = (user_features['checkouts'] > 0).astype(int)
        user_features['has_multiple_orders'] = (user_features['checkouts'] > 1).astype(int)

        # Funnel conversion rates
        user_features['view_to_cart_rate'] = (
            user_features['cart_adds'] / user_features['product_views'].clip(lower=1)
        ).clip(upper=1)

        user_features['cart_to_checkout_rate'] = (
            user_features['checkouts'] / user_features['cart_adds'].clip(lower=1)
        ).clip(upper=1)

        # Platform features
        platform_counts = df.groupby(['amplitude_id', 'platform']).size().unstack(fill_value=0)
        for platform in platform_counts.columns:
            user_features[f'platform_{platform.lower()}'] = user_features['amplitude_id'].map(
                platform_counts[platform]
            ).fillna(0)

        # Hour of day patterns
        df['hour'] = df['event_time'].dt.hour
        hour_dist = df.groupby('amplitude_id')['hour'].agg(['mean', 'std'])
        user_features['avg_hour'] = user_features['amplitude_id'].map(hour_dist['mean']).fillna(12)
        user_features['hour_std'] = user_features['amplitude_id'].map(hour_dist['std']).fillna(0)

        # Day of week patterns
        df['dayofweek'] = df['event_time'].dt.dayofweek
        dow_dist = df.groupby('amplitude_id')['dayofweek'].agg(['mean', 'std'])
        user_features['avg_dayofweek'] = user_features['amplitude_id'].map(dow_dist['mean']).fillna(3)
        user_features['dayofweek_std'] = user_features['amplitude_id'].map(dow_dist['std']).fillna(0)

        # Weekend vs weekday
        df['is_weekend'] = df['dayofweek'].isin([5, 6])
        weekend_ratio = df.groupby('amplitude_id')['is_weekend'].mean()
        user_features['weekend_ratio'] = user_features['amplitude_id'].map(weekend_ratio).fillna(0.3)

        # Drop date columns
        user_features = user_features.drop(columns=['first_event', 'last_event'], errors='ignore')

        logger.info(f"Generated {len(user_features.columns) - 1} features for {len(user_features)} users")

        return user_features

    def _parse_event_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse JSON event properties."""
        import json

        def safe_parse(x):
            if pd.isna(x) or x == '':
                return {}
            try:
                return json.loads(x) if isinstance(x, str) else x
            except:
                return {}

        df['event_props_parsed'] = df['event_properties'].apply(safe_parse)
        return df

    def prepare_labels(self, df_history: pd.DataFrame, df_future: pd.DataFrame) -> pd.Series:
        """Create churn labels.

        Churn = 1 if user was active in history but NOT in future period.
        Churn = 0 if user was active in both periods.
        """
        logger.info("Creating churn labels...")

        history_users = set(df_history['amplitude_id'].unique())
        future_users = set(df_future['amplitude_id'].unique())

        # Users who churned = in history but not in future
        churned_users = history_users - future_users
        retained_users = history_users & future_users

        logger.info(f"History users: {len(history_users)}")
        logger.info(f"Future users: {len(future_users)}")
        logger.info(f"Churned: {len(churned_users)} ({len(churned_users)/len(history_users):.1%})")
        logger.info(f"Retained: {len(retained_users)} ({len(retained_users)/len(history_users):.1%})")

        # Create labels series
        labels = pd.Series(index=list(history_users), dtype=int)
        labels.loc[list(churned_users)] = 1
        labels.loc[list(retained_users)] = 0

        return labels

    def run_prediction(self, df_history: pd.DataFrame, df_future: pd.DataFrame,
                       test_size: float = 0.2) -> Dict:
        """Full pipeline: features -> labels -> train -> evaluate.

        Args:
            df_history: Historical event data (for feature building)
            df_future: Future event data (for labels - did they return?)
            test_size: Fraction for test split

        Returns:
            Dictionary with model metrics and predictions
        """
        # Build features from history
        features = self.prepare_features(df_history)

        # Build labels from future
        labels = self.prepare_labels(df_history, df_future)

        # Align features and labels
        features = features.set_index('amplitude_id')
        common_users = features.index.intersection(labels.index)
        features = features.loc[common_users]
        labels = labels.loc[common_users]

        logger.info(f"Training on {len(common_users)} users")

        # Train model
        metrics = self.train(features, labels, test_size=test_size)

        # Get feature importance
        importance = self.get_feature_importance(top_n=15)
        logger.info("\nTop 15 Important Features:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Add predictions for all users
        features['churn_probability'] = self.predict_proba(features)
        features['churn_prediction'] = self.predict(features)

        # Get predictions with amplitude_id
        predictions = features[['churn_probability', 'churn_prediction']].reset_index()
        predictions = predictions.rename(columns={'index': 'amplitude_id'})

        return {
            'metrics': metrics,
            'feature_importance': importance,
            'predictions': predictions
        }

    def identify_at_risk_users(self, predictions: pd.DataFrame,
                                threshold: float = 0.7) -> pd.DataFrame:
        """Identify high-risk users for intervention.

        Args:
            predictions: DataFrame with amplitude_id and churn_probability
            threshold: Probability threshold for high risk (default 0.7)

        Returns:
            DataFrame of at-risk users sorted by risk
        """
        at_risk = predictions[predictions['churn_probability'] >= threshold].copy()
        at_risk = at_risk.sort_values('churn_probability', ascending=False)

        logger.info(f"Identified {len(at_risk)} at-risk users (>{threshold:.0%} churn probability)")

        return at_risk
