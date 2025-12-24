"""Lifetime Value (LTV) Prediction Model.

Predicts the expected lifetime value of customers.

Business Use Cases:
- Customer acquisition cost optimization (spend up to LTV)
- VIP customer identification and treatment
- Retention ROI calculations
- Cohort analysis and forecasting
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LTVPredictor:
    """Predict customer lifetime value based on early behavior."""

    def __init__(self, config: Dict = None, model_type: str = 'gradient_boosting'):
        self.config = config or {}
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build LTV prediction features from event data.

        Features capture early indicators of high-value customers:
        - First order behavior (AOV, time to first order)
        - Engagement depth (sessions, events, variety)
        - Order patterns (frequency, recency)
        - Product preferences (categories, merchants)
        """
        logger.info("Building LTV prediction features...")

        # Parse event properties for order values
        df = self._extract_order_values(df)

        max_date = df['event_time'].max()

        # Basic user aggregations
        user_features = df.groupby('amplitude_id').agg({
            'event_time': ['min', 'max', 'count'],
            'event_type': 'nunique'
        }).reset_index()
        user_features.columns = ['amplitude_id', 'first_event', 'last_event',
                                  'total_events', 'unique_event_types']

        # Order metrics
        orders = df[df['event_type'] == 'checkout_completed'].copy()

        if len(orders) > 0:
            order_stats = orders.groupby('amplitude_id').agg({
                'event_time': ['min', 'count'],
                'order_total': ['sum', 'mean', 'min', 'max', 'std']
            }).reset_index()

            order_stats.columns = ['amplitude_id', 'first_order_time', 'order_count',
                                    'total_revenue', 'avg_order_value', 'min_order_value',
                                    'max_order_value', 'order_value_std']

            user_features = user_features.merge(order_stats, on='amplitude_id', how='left')

            # Fill NaN for non-ordering users
            order_cols = ['order_count', 'total_revenue', 'avg_order_value',
                         'min_order_value', 'max_order_value', 'order_value_std']
            user_features[order_cols] = user_features[order_cols].fillna(0)

            # Time to first order
            user_features['time_to_first_order_hours'] = (
                (user_features['first_order_time'] - user_features['first_event']).dt.total_seconds() / 3600
            ).fillna(-1)  # -1 for users who never ordered

            # Days since first order
            user_features['days_since_first_order'] = (
                (max_date - user_features['first_order_time']).dt.total_seconds() / 86400
            ).fillna(-1)

            user_features = user_features.drop(columns=['first_order_time'], errors='ignore')
        else:
            user_features['order_count'] = 0
            user_features['total_revenue'] = 0
            user_features['avg_order_value'] = 0
            user_features['min_order_value'] = 0
            user_features['max_order_value'] = 0
            user_features['order_value_std'] = 0
            user_features['time_to_first_order_hours'] = -1
            user_features['days_since_first_order'] = -1

        # Account age and recency
        user_features['account_age_days'] = (
            (max_date - user_features['first_event']).dt.total_seconds() / 86400
        )

        user_features['days_since_last_activity'] = (
            (max_date - user_features['last_event']).dt.total_seconds() / 86400
        )

        # Active days
        active_days = df.groupby('amplitude_id')['event_time'].apply(
            lambda x: x.dt.date.nunique()
        )
        user_features['active_days'] = user_features['amplitude_id'].map(active_days).fillna(1)

        # Order frequency (orders per active day)
        user_features['order_frequency'] = (
            user_features['order_count'] / user_features['active_days'].clip(lower=1)
        )

        # Revenue velocity (revenue per day since first order)
        user_features['revenue_velocity'] = np.where(
            user_features['days_since_first_order'] > 0,
            user_features['total_revenue'] / user_features['days_since_first_order'],
            user_features['total_revenue']
        )

        # Engagement metrics
        user_features['events_per_day'] = (
            user_features['total_events'] / user_features['account_age_days'].clip(lower=1)
        )

        # Session count (approximation)
        df_sorted = df.sort_values(['amplitude_id', 'event_time'])
        df_sorted['time_diff'] = df_sorted.groupby('amplitude_id')['event_time'].diff()
        df_sorted['new_session'] = df_sorted['time_diff'] > pd.Timedelta(minutes=30)
        sessions = df_sorted.groupby('amplitude_id')['new_session'].sum() + 1
        user_features['total_sessions'] = user_features['amplitude_id'].map(sessions).fillna(1)

        # Sessions per day
        user_features['sessions_per_day'] = (
            user_features['total_sessions'] / user_features['account_age_days'].clip(lower=1)
        )

        # Funnel behavior
        funnel_events = {
            'homepage_views': 'homepage_viewed',
            'searches': 'search_made',
            'merchant_views': 'merchant_page_viewed',
            'product_views': 'product_page_viewed',
            'cart_adds': 'product_added'
        }

        for feature_name, event_type in funnel_events.items():
            counts = df[df['event_type'] == event_type].groupby('amplitude_id').size()
            user_features[feature_name] = user_features['amplitude_id'].map(counts).fillna(0)

        # Browsing to buying ratios
        user_features['views_per_order'] = np.where(
            user_features['order_count'] > 0,
            user_features['product_views'] / user_features['order_count'],
            user_features['product_views']
        )

        user_features['searches_per_order'] = np.where(
            user_features['order_count'] > 0,
            user_features['searches'] / user_features['order_count'],
            user_features['searches']
        )

        # Merchant diversity (if we have merchant data)
        merchant_events = df[df['event_type'].isin(['merchant_page_viewed', 'checkout_completed'])]
        if 'merchant_id' in df.columns:
            merchant_diversity = df[df['event_type'] == 'checkout_completed'].groupby(
                'amplitude_id'
            )['merchant_id'].nunique()
            user_features['unique_merchants_ordered'] = user_features['amplitude_id'].map(
                merchant_diversity
            ).fillna(0)
        else:
            user_features['unique_merchants_ordered'] = 0

        # Platform
        platform_mode = df.groupby('amplitude_id')['platform'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        )
        user_features['platform'] = user_features['amplitude_id'].map(platform_mode)
        user_features = pd.get_dummies(user_features, columns=['platform'], prefix='platform')

        # Time patterns
        df['hour'] = df['event_time'].dt.hour
        hour_mean = df.groupby('amplitude_id')['hour'].mean()
        user_features['avg_hour'] = user_features['amplitude_id'].map(hour_mean).fillna(12)

        # Weekend ordering tendency
        df['is_weekend'] = df['event_time'].dt.dayofweek.isin([5, 6])
        weekend_ratio = df.groupby('amplitude_id')['is_weekend'].mean()
        user_features['weekend_ratio'] = user_features['amplitude_id'].map(weekend_ratio).fillna(0.3)

        # Derived features
        user_features['is_repeat_customer'] = (user_features['order_count'] > 1).astype(int)
        user_features['is_high_value'] = (user_features['avg_order_value'] > user_features['avg_order_value'].median()).astype(int)

        # Drop datetime columns
        user_features = user_features.drop(columns=['first_event', 'last_event'], errors='ignore')

        logger.info(f"Generated {len(user_features.columns) - 1} features for {len(user_features)} users")

        return user_features

    def _extract_order_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract order_total from event_properties."""
        import json

        if 'order_total' in df.columns:
            return df

        df = df.copy()

        def extract_order_total(row):
            if row['event_type'] != 'checkout_completed':
                return np.nan
            props = row.get('event_properties', '{}')
            if pd.isna(props) or props == '':
                return np.nan
            try:
                if isinstance(props, str):
                    props = json.loads(props)
                return float(props.get('order_total', 0))
            except:
                return np.nan

        df['order_total'] = df.apply(extract_order_total, axis=1)

        return df

    def prepare_labels(self, features: pd.DataFrame) -> pd.Series:
        """Create LTV labels (total_revenue is our target)."""
        labels = features['total_revenue'].copy()
        logger.info(f"LTV distribution: mean={labels.mean():.2f}, median={labels.median():.2f}, max={labels.max():.2f}")
        return labels

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train the LTV model."""
        # Remove target from features
        X = X.drop(columns=['total_revenue', 'amplitude_id'], errors='ignore')

        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()
        X_numeric = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)

        self.feature_columns = numeric_cols

        # Split data - stratify by having orders or not
        has_orders = (y > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=has_orders
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, None)  # LTV can't be negative

        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mean_actual': y_test.mean(),
            'mean_predicted': y_pred.mean(),
            'test_size': len(X_test),
            'train_size': len(X_train)
        }

        logger.info(f"Model Performance:")
        logger.info(f"  MAE: ${self.metrics['mae']:.2f}")
        logger.info(f"  RMSE: ${self.metrics['rmse']:.2f}")
        logger.info(f"  R²: {self.metrics['r2']:.3f}")

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict LTV for new data."""
        X_numeric = X[self.feature_columns].copy()
        X_numeric = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_numeric)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, None)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance rankings."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    def run_prediction(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Full pipeline: features -> labels -> train -> evaluate."""

        # Build features
        features = self.prepare_features(df)

        # Only train on users who have ordered (can't predict LTV for non-customers)
        ordering_users = features[features['order_count'] > 0].copy()
        logger.info(f"Training on {len(ordering_users)} ordering users")

        # Build labels
        labels = self.prepare_labels(ordering_users)

        # Train model
        metrics = self.train(ordering_users, labels, test_size=test_size)

        # Get feature importance
        importance = self.get_feature_importance(top_n=15)
        logger.info("\nTop 15 Important Features:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Predict for all ordering users
        features_for_pred = ordering_users.drop(columns=['amplitude_id', 'total_revenue'], errors='ignore')
        ordering_users['predicted_ltv'] = self.predict(features_for_pred)

        return {
            'metrics': metrics,
            'feature_importance': importance,
            'predictions': ordering_users[['amplitude_id', 'total_revenue', 'predicted_ltv', 'order_count']].copy()
        }

    def segment_by_ltv(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Segment users by predicted LTV."""
        df = predictions.copy()

        # Create LTV tiers
        df['ltv_percentile'] = df['predicted_ltv'].rank(pct=True)

        conditions = [
            df['ltv_percentile'] >= 0.9,
            df['ltv_percentile'] >= 0.7,
            df['ltv_percentile'] >= 0.4,
            df['ltv_percentile'] >= 0.0
        ]
        labels = ['Diamond', 'Gold', 'Silver', 'Bronze']

        df['ltv_tier'] = np.select(conditions, labels, default='Bronze')

        # Summary stats
        tier_summary = df.groupby('ltv_tier').agg({
            'amplitude_id': 'count',
            'predicted_ltv': ['mean', 'sum'],
            'total_revenue': ['mean', 'sum'],
            'order_count': 'mean'
        }).round(2)

        logger.info("\nLTV Tier Summary:")
        logger.info(tier_summary.to_string())

        return df
