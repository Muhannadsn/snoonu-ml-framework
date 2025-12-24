"""Base predictor class for all models."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class BasePredictor:
    """Base class for all prediction models."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override in subclass to build model-specific features."""
        raise NotImplementedError

    def prepare_labels(self, df: pd.DataFrame, future_df: pd.DataFrame = None) -> pd.Series:
        """Override in subclass to build model-specific labels."""
        raise NotImplementedError

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train the model with train/test split."""
        # Remove any non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()

        # Handle missing values
        X_numeric = X_numeric.fillna(0)

        # Replace infinities
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)

        self.feature_columns = numeric_cols

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Positive class rate: {y.mean():.2%}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_size': len(X_test),
            'train_size': len(X_train)
        }

        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy: {self.metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {self.metrics['precision']:.3f}")
        logger.info(f"  Recall: {self.metrics['recall']:.3f}")
        logger.info(f"  F1 Score: {self.metrics['f1']:.3f}")
        logger.info(f"  AUC-ROC: {self.metrics['auc_roc']:.3f}")

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on new data."""
        X_numeric = X[self.feature_columns].copy()
        X_numeric = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_numeric)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities on new data."""
        X_numeric = X[self.feature_columns].copy()
        X_numeric = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_numeric)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance rankings."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'metrics': self.metrics,
                'config': self.config
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.metrics = data['metrics']
        self.config = data['config']
        logger.info(f"Model loaded from {path}")
