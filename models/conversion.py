"""Conversion Prediction Model.

Predicts which users/sessions are likely to convert (place an order).

Business Use Cases:
- Real-time personalization for high-intent users
- Identify friction points in the funnel
- Optimize marketing spend on likely converters
- A/B test impact predictions
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional
import logging

from .base import BasePredictor

logger = logging.getLogger(__name__)


class ConversionPredictor(BasePredictor):
    """Predict session/user conversion probability."""

    def __init__(self, config: Dict = None, model_type: str = 'gradient_boosting'):
        super().__init__(config)
        self.model_type = model_type

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=50,
                random_state=42
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )

    def prepare_features(self, df: pd.DataFrame, level: str = 'user') -> pd.DataFrame:
        """Build conversion prediction features.

        Args:
            df: Event data
            level: 'user' for user-level or 'session' for session-level

        Features capture:
        - Funnel progress (how far did they get?)
        - Engagement depth (time, events, pages viewed)
        - Historical behavior (past conversions, frequency)
        - Context (platform, time of day, day of week)
        """
        logger.info(f"Building {level}-level conversion features...")

        if level == 'session':
            return self._build_session_features(df)
        else:
            return self._build_user_features(df)

    def _build_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build user-level features for conversion prediction."""
        max_date = df['event_time'].max()

        # Define funnel stages
        funnel_stages = [
            'homepage_viewed',
            'category_page_viewed',
            'merchant_page_viewed',
            'product_page_viewed',
            'product_added',
            'cart_page_viewed',
            'checkout_button_pressed',
            'payment_initiated',
            'checkout_completed'
        ]

        # Basic user aggregations
        user_features = df.groupby('amplitude_id').agg({
            'event_time': ['min', 'max', 'count'],
            'event_type': 'nunique'
        }).reset_index()
        user_features.columns = ['amplitude_id', 'first_event', 'last_event',
                                  'total_events', 'unique_event_types']

        # Funnel stage counts
        for stage in funnel_stages:
            counts = df[df['event_type'] == stage].groupby('amplitude_id').size()
            user_features[f'n_{stage}'] = user_features['amplitude_id'].map(counts).fillna(0)

        # Funnel progress (furthest stage reached)
        def get_furthest_stage(user_id, df, stages):
            user_events = set(df[df['amplitude_id'] == user_id]['event_type'].unique())
            for i, stage in enumerate(reversed(stages)):
                if stage in user_events:
                    return len(stages) - i
            return 0

        # This is slow, optimize with vectorized approach
        stage_set = set(funnel_stages)
        user_stage_presence = df[df['event_type'].isin(stage_set)].groupby(
            ['amplitude_id', 'event_type']
        ).size().unstack(fill_value=0)

        for i, stage in enumerate(funnel_stages):
            if stage in user_stage_presence.columns:
                user_stage_presence[f'reached_{i}'] = (user_stage_presence[stage] > 0).astype(int)

        # Calculate max funnel stage reached
        stage_cols = [c for c in user_stage_presence.columns if c.startswith('reached_')]
        if stage_cols:
            user_stage_presence['max_stage'] = user_stage_presence[stage_cols].sum(axis=1)
            user_features['funnel_depth'] = user_features['amplitude_id'].map(
                user_stage_presence['max_stage']
            ).fillna(0)
        else:
            user_features['funnel_depth'] = 0

        # Time features
        user_features['session_duration_mins'] = (
            (user_features['last_event'] - user_features['first_event']).dt.total_seconds() / 60
        ).clip(lower=0, upper=180)  # Cap at 3 hours

        user_features['events_per_minute'] = (
            user_features['total_events'] / user_features['session_duration_mins'].clip(lower=1)
        )

        # Search behavior
        search_counts = df[df['event_type'] == 'search_made'].groupby('amplitude_id').size()
        user_features['n_searches'] = user_features['amplitude_id'].map(search_counts).fillna(0)
        user_features['has_searched'] = (user_features['n_searches'] > 0).astype(int)

        # Cart behavior
        cart_events = df[df['event_type'].isin(['product_added', 'cart_page_viewed', 'cart_created'])]
        cart_counts = cart_events.groupby('amplitude_id').size()
        user_features['cart_interactions'] = user_features['amplitude_id'].map(cart_counts).fillna(0)

        # Payment friction
        payment_change = df[df['event_type'] == 'change_payment_method_viewed'].groupby('amplitude_id').size()
        user_features['payment_changes'] = user_features['amplitude_id'].map(payment_change).fillna(0)
        user_features['had_payment_friction'] = (user_features['payment_changes'] > 0).astype(int)

        # Platform
        platform_mode = df.groupby('amplitude_id')['platform'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown')
        user_features['platform'] = user_features['amplitude_id'].map(platform_mode)
        user_features = pd.get_dummies(user_features, columns=['platform'], prefix='platform')

        # Time of day patterns
        df['hour'] = df['event_time'].dt.hour
        hour_stats = df.groupby('amplitude_id')['hour'].agg(['mean', 'min', 'max'])
        user_features['avg_hour'] = user_features['amplitude_id'].map(hour_stats['mean']).fillna(12)
        user_features['first_hour'] = user_features['amplitude_id'].map(hour_stats['min']).fillna(12)

        # Is peak ordering time (lunch 11-14, dinner 18-22)
        user_features['is_lunch_time'] = user_features['avg_hour'].between(11, 14).astype(int)
        user_features['is_dinner_time'] = user_features['avg_hour'].between(18, 22).astype(int)

        # Day of week
        df['dayofweek'] = df['event_time'].dt.dayofweek
        dow_mode = df.groupby('amplitude_id')['dayofweek'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3)
        user_features['dayofweek'] = user_features['amplitude_id'].map(dow_mode).fillna(3)
        user_features['is_weekend'] = user_features['dayofweek'].isin([5, 6]).astype(int)

        # Merchant engagement
        merchant_views = df[df['event_type'] == 'merchant_page_viewed'].groupby('amplitude_id')['event_type'].count()
        user_features['merchants_viewed'] = user_features['amplitude_id'].map(merchant_views).fillna(0)

        unique_merchants = df[df['event_type'] == 'merchant_page_viewed']
        if 'event_properties' in unique_merchants.columns:
            # Try to extract merchant_id from properties
            pass  # Skip for now, would need JSON parsing

        # Product engagement
        product_views = df[df['event_type'] == 'product_page_viewed'].groupby('amplitude_id').size()
        user_features['products_viewed'] = user_features['amplitude_id'].map(product_views).fillna(0)

        # Ratios
        user_features['view_to_add_ratio'] = (
            user_features['n_product_added'] / user_features['products_viewed'].clip(lower=1)
        ).clip(upper=1)

        user_features['add_to_checkout_ratio'] = (
            user_features['n_checkout_completed'] / user_features['n_product_added'].clip(lower=1)
        ).clip(upper=1)

        # Drop datetime columns
        user_features = user_features.drop(columns=['first_event', 'last_event'], errors='ignore')

        logger.info(f"Generated {len(user_features.columns) - 1} features for {len(user_features)} users")

        return user_features

    def _build_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build session-level features."""
        # Sessionize the data
        df = df.sort_values(['amplitude_id', 'event_time'])
        df['time_diff'] = df.groupby('amplitude_id')['event_time'].diff()
        df['new_session'] = (df['time_diff'] > pd.Timedelta(minutes=30)) | df['time_diff'].isna()
        df['session_id'] = df.groupby('amplitude_id')['new_session'].cumsum()
        df['session_key'] = df['amplitude_id'].astype(str) + '_' + df['session_id'].astype(str)

        # Build features per session
        session_features = df.groupby('session_key').agg({
            'amplitude_id': 'first',
            'event_time': ['min', 'max', 'count'],
            'event_type': 'nunique'
        }).reset_index()

        session_features.columns = ['session_key', 'amplitude_id', 'session_start',
                                     'session_end', 'events_in_session', 'unique_events']

        session_features['session_duration_mins'] = (
            (session_features['session_end'] - session_features['session_start']).dt.total_seconds() / 60
        ).clip(lower=0, upper=120)

        # Funnel progress in session
        funnel_events = ['homepage_viewed', 'product_page_viewed', 'product_added',
                        'checkout_button_pressed', 'checkout_completed']

        for event in funnel_events:
            event_in_session = df[df['event_type'] == event].groupby('session_key').size()
            session_features[f'has_{event}'] = session_features['session_key'].map(
                event_in_session
            ).fillna(0).clip(upper=1).astype(int)

        # Calculate session funnel depth
        session_features['funnel_depth'] = (
            session_features['has_homepage_viewed'] +
            session_features['has_product_page_viewed'] +
            session_features['has_product_added'] +
            session_features['has_checkout_button_pressed']
        )

        session_features = session_features.drop(columns=['session_start', 'session_end'])

        return session_features

    def prepare_labels(self, features: pd.DataFrame, df: pd.DataFrame = None) -> pd.Series:
        """Create conversion labels.

        Conversion = 1 if user/session had checkout_completed event.
        """
        if 'n_checkout_completed' in features.columns:
            labels = (features['n_checkout_completed'] > 0).astype(int)
        elif 'has_checkout_completed' in features.columns:
            labels = features['has_checkout_completed'].astype(int)
        else:
            # Need to compute from raw data
            converters = set(df[df['event_type'] == 'checkout_completed']['amplitude_id'].unique())
            if 'amplitude_id' in features.columns:
                labels = features['amplitude_id'].isin(converters).astype(int)
            else:
                labels = features.index.isin(converters).astype(int)

        logger.info(f"Conversion rate: {labels.mean():.2%}")

        return labels

    def run_prediction(self, df: pd.DataFrame, level: str = 'user',
                       test_size: float = 0.2) -> Dict:
        """Full pipeline: features -> labels -> train -> evaluate."""

        # Build features
        features = self.prepare_features(df, level=level)

        # Build labels
        labels = self.prepare_labels(features, df)

        # Prepare for training
        if 'amplitude_id' in features.columns:
            features = features.set_index('amplitude_id')
        if 'session_key' in features.columns:
            features = features.set_index('session_key')

        # Remove target leakage (checkout-related columns)
        leaky_patterns = ['checkout_completed', 'checkout', 'add_to_checkout', 'payment']
        leaky_cols = [c for c in features.columns if any(p in c.lower() for p in leaky_patterns)]
        features = features.drop(columns=leaky_cols, errors='ignore')
        logger.info(f"Removed {len(leaky_cols)} leaky features: {leaky_cols}")

        logger.info(f"Training conversion model on {len(features)} samples")

        # Train model
        metrics = self.train(features, labels, test_size=test_size)

        # Get feature importance
        importance = self.get_feature_importance(top_n=15)
        logger.info("\nTop 15 Important Features:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Add predictions
        features['conversion_probability'] = self.predict_proba(features)
        features['conversion_prediction'] = self.predict(features)

        return {
            'metrics': metrics,
            'feature_importance': importance,
            'predictions': features[['conversion_probability', 'conversion_prediction']].reset_index()
        }

    def score_realtime(self, session_events: List[Dict]) -> float:
        """Score a session in real-time based on current events.

        Args:
            session_events: List of event dictionaries from current session

        Returns:
            Conversion probability
        """
        # Convert to DataFrame
        df = pd.DataFrame(session_events)

        # Build features
        features = self._build_session_features(df)

        # Predict
        return self.predict_proba(features)[0]
