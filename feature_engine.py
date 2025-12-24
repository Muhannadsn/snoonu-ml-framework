"""
Feature Engineering Module
==========================
Generates user-level and session-level features from raw event data.
All features are configurable and automatically generated.
"""

import pandas as pd
import numpy as np
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Generate ML features from Snoonu event data.

    Usage:
        engine = FeatureEngine(config_path='config.yaml')
        user_features = engine.build_user_features(df)
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.events = self.config.get('events', {})
        self.feature_config = self.config.get('features', {})

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _parse_props(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse event_properties JSON if not already parsed."""
        if 'props' in df.columns:
            return df

        def safe_parse(x):
            if pd.isna(x):
                return {}
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                return {}

        df = df.copy()
        df['props'] = df['event_properties'].apply(safe_parse)
        return df

    def build_user_features(self, df: pd.DataFrame,
                            reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Build user-level features for all users in the dataset.

        Args:
            df: Raw event DataFrame
            reference_date: Date to calculate recency from (default: max date in data)

        Returns:
            DataFrame with one row per user and feature columns
        """
        logger.info("Building user features...")

        df = self._parse_props(df)

        if reference_date is None:
            reference_date = df['event_time'].max()

        # Get all unique users
        users = df['amplitude_id'].unique()
        logger.info(f"Processing {len(users):,} users")

        # Build feature DataFrames
        features = []

        # Basic counts
        features.append(self._user_event_counts(df))

        # Recency features
        features.append(self._user_recency_features(df, reference_date))

        # Order features
        features.append(self._user_order_features(df))

        # Behavioral features
        features.append(self._user_behavioral_features(df))

        # Platform features
        features.append(self._user_platform_features(df))

        # Funnel features
        features.append(self._user_funnel_features(df))

        # Merge all features
        user_df = features[0]
        for feat_df in features[1:]:
            user_df = user_df.merge(feat_df, on='amplitude_id', how='left')

        # Fill NaN values
        user_df = user_df.fillna(0)

        logger.info(f"Generated {len(user_df.columns) - 1} features for {len(user_df):,} users")

        return user_df

    def _user_event_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count events per user."""
        # Total events
        total_events = df.groupby('amplitude_id').size().reset_index(name='total_events')

        # Events by type
        event_counts = df.pivot_table(
            index='amplitude_id',
            columns='event_type',
            aggfunc='size',
            fill_value=0
        ).reset_index()

        # Rename columns
        event_counts.columns = ['amplitude_id'] + [f'count_{col}' for col in event_counts.columns[1:]]

        return total_events.merge(event_counts, on='amplitude_id', how='left')

    def _user_recency_features(self, df: pd.DataFrame,
                               reference_date: datetime) -> pd.DataFrame:
        """Calculate recency features."""
        user_times = df.groupby('amplitude_id').agg({
            'event_time': ['min', 'max']
        }).reset_index()
        user_times.columns = ['amplitude_id', 'first_event', 'last_event']

        # Days since first/last event
        user_times['days_since_first_event'] = (
            reference_date - user_times['first_event']
        ).dt.total_seconds() / 86400

        user_times['days_since_last_event'] = (
            reference_date - user_times['last_event']
        ).dt.total_seconds() / 86400

        # Last order time
        orders = df[df['event_type'] == 'checkout_completed']
        if len(orders) > 0:
            last_order = orders.groupby('amplitude_id')['event_time'].max().reset_index()
            last_order.columns = ['amplitude_id', 'last_order_time']
            last_order['days_since_last_order'] = (
                reference_date - last_order['last_order_time']
            ).dt.total_seconds() / 86400
            user_times = user_times.merge(
                last_order[['amplitude_id', 'days_since_last_order']],
                on='amplitude_id', how='left'
            )
        else:
            user_times['days_since_last_order'] = np.nan

        return user_times[['amplitude_id', 'days_since_first_event',
                          'days_since_last_event', 'days_since_last_order']]

    def _user_order_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order-related features."""
        orders = df[df['event_type'] == 'checkout_completed'].copy()

        if len(orders) == 0:
            return pd.DataFrame({'amplitude_id': df['amplitude_id'].unique()})

        # Extract order properties
        orders['order_total'] = orders['props'].apply(
            lambda x: float(x.get('order_total', 0)) if x.get('order_total') else 0
        )
        orders['products_quantity'] = orders['props'].apply(
            lambda x: int(x.get('products_quantity', 0)) if x.get('products_quantity') else 0
        )
        orders['has_promo'] = orders['props'].apply(
            lambda x: 1 if x.get('promo_code') and x.get('promo_code') != 'N/A' else 0
        )
        orders['is_group_order'] = orders['props'].apply(
            lambda x: 1 if x.get('group_order') is True else 0
        )
        orders['is_multi_cart'] = orders['props'].apply(
            lambda x: 1 if x.get('multi_cart') or x.get('multicart') else 0
        )
        orders['merchant_id'] = orders['props'].apply(
            lambda x: x.get('merchant_id')
        )

        # Aggregate by user
        order_features = orders.groupby('amplitude_id').agg({
            'order_total': ['count', 'sum', 'mean', 'max'],
            'products_quantity': ['sum', 'mean'],
            'has_promo': 'sum',
            'is_group_order': 'sum',
            'is_multi_cart': 'sum',
            'merchant_id': 'nunique'
        }).reset_index()

        # Flatten column names
        order_features.columns = [
            'amplitude_id',
            'total_orders', 'total_revenue', 'avg_order_value', 'max_order_value',
            'total_products', 'avg_products_per_order',
            'promo_orders', 'group_orders', 'multi_cart_orders',
            'unique_merchants'
        ]

        # Calculate percentages
        order_features['pct_promo_orders'] = (
            order_features['promo_orders'] / order_features['total_orders']
        ).fillna(0)
        order_features['pct_group_orders'] = (
            order_features['group_orders'] / order_features['total_orders']
        ).fillna(0)

        return order_features

    def _user_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral features."""
        # Searches
        searches = df[df['event_type'] == 'search_made'].groupby('amplitude_id').size()
        searches = searches.reset_index(name='total_searches')

        # Product views
        product_views = df[df['event_type'] == 'product_page_viewed'].groupby('amplitude_id').size()
        product_views = product_views.reset_index(name='total_product_views')

        # Products added
        products_added = df[df['event_type'] == 'product_added'].groupby('amplitude_id').size()
        products_added = products_added.reset_index(name='total_products_added')

        # Cart views
        cart_views = df[df['event_type'] == 'cart_page_viewed'].groupby('amplitude_id').size()
        cart_views = cart_views.reset_index(name='total_cart_views')

        # Unique days active
        df_temp = df.copy()
        df_temp['date'] = df_temp['event_time'].dt.date
        days_active = df_temp.groupby('amplitude_id')['date'].nunique()
        days_active = days_active.reset_index(name='days_active')

        # Session count (approximate by homepage views)
        sessions = df[df['event_type'] == 'homepage_viewed'].groupby('amplitude_id').size()
        sessions = sessions.reset_index(name='total_sessions')

        # Merge all
        all_users = pd.DataFrame({'amplitude_id': df['amplitude_id'].unique()})
        for feat_df in [searches, product_views, products_added, cart_views, days_active, sessions]:
            all_users = all_users.merge(feat_df, on='amplitude_id', how='left')

        return all_users.fillna(0)

    def _user_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate platform features."""
        # Events per platform
        platform_counts = df.groupby(['amplitude_id', 'platform']).size().unstack(fill_value=0)
        platform_counts = platform_counts.reset_index()

        # Determine primary platform
        platform_cols = [col for col in platform_counts.columns if col != 'amplitude_id']
        if platform_cols:
            platform_counts['primary_platform'] = platform_counts[platform_cols].idxmax(axis=1)

            # Calculate iOS percentage
            if 'ios' in platform_cols:
                platform_counts['pct_ios'] = (
                    platform_counts['ios'] /
                    platform_counts[platform_cols].sum(axis=1)
                )
            else:
                platform_counts['pct_ios'] = 0

            return platform_counts[['amplitude_id', 'primary_platform', 'pct_ios']]

        return pd.DataFrame({'amplitude_id': df['amplitude_id'].unique()})

    def _user_funnel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate funnel-related features."""
        funnel_stages = self.events.get('funnel', [
            'homepage_viewed', 'category_page_viewed', 'merchant_page_viewed',
            'product_page_viewed', 'product_added', 'cart_page_viewed',
            'checkout_button_pressed', 'payment_initiated', 'checkout_completed'
        ])

        # Get max funnel stage reached by each user
        stage_map = {stage: i for i, stage in enumerate(funnel_stages)}

        df_temp = df[df['event_type'].isin(funnel_stages)].copy()
        df_temp['funnel_stage'] = df_temp['event_type'].map(stage_map)

        funnel_features = df_temp.groupby('amplitude_id').agg({
            'funnel_stage': ['max', 'mean']
        }).reset_index()
        funnel_features.columns = ['amplitude_id', 'max_funnel_stage', 'avg_funnel_stage']

        # Conversion rate (checkouts / homepage views)
        homepage = df[df['event_type'] == 'homepage_viewed'].groupby('amplitude_id').size()
        checkouts = df[df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()

        conversion = pd.DataFrame({
            'amplitude_id': homepage.index,
            'homepage_views': homepage.values
        })
        checkout_df = pd.DataFrame({
            'amplitude_id': checkouts.index,
            'checkouts': checkouts.values
        })
        conversion = conversion.merge(checkout_df, on='amplitude_id', how='left').fillna(0)
        conversion['user_conversion_rate'] = conversion['checkouts'] / conversion['homepage_views']

        funnel_features = funnel_features.merge(
            conversion[['amplitude_id', 'user_conversion_rate']],
            on='amplitude_id', how='left'
        )

        return funnel_features.fillna(0)

    def build_rfm_features(self, df: pd.DataFrame,
                           reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Build RFM (Recency, Frequency, Monetary) features.

        Args:
            df: Raw event DataFrame
            reference_date: Date to calculate recency from

        Returns:
            DataFrame with RFM scores per user
        """
        logger.info("Building RFM features...")

        df = self._parse_props(df)

        if reference_date is None:
            reference_date = df['event_time'].max()

        orders = df[df['event_type'] == 'checkout_completed'].copy()

        if len(orders) == 0:
            logger.warning("No orders found in data")
            return pd.DataFrame()

        # Extract order total
        orders['order_total'] = orders['props'].apply(
            lambda x: float(x.get('order_total', 0)) if x.get('order_total') else 0
        )

        # Calculate RFM
        rfm = orders.groupby('amplitude_id').agg({
            'event_time': 'max',  # Recency
            'order_total': ['count', 'sum']  # Frequency, Monetary
        }).reset_index()

        rfm.columns = ['amplitude_id', 'last_order_date', 'frequency', 'monetary']

        # Calculate recency in days
        rfm['recency'] = (reference_date - rfm['last_order_date']).dt.total_seconds() / 86400

        # RFM scores (1-5, 5 is best)
        n_quantiles = self.config.get('models', {}).get('segmentation', {}).get('rfm_quantiles', 5)

        # Recency: lower is better
        rfm['R_score'] = pd.qcut(rfm['recency'], n_quantiles, labels=range(n_quantiles, 0, -1), duplicates='drop')

        # Frequency: higher is better
        rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), n_quantiles, labels=range(1, n_quantiles + 1), duplicates='drop')

        # Monetary: higher is better
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), n_quantiles, labels=range(1, n_quantiles + 1), duplicates='drop')

        # Combined RFM score
        rfm['R_score'] = rfm['R_score'].astype(int)
        rfm['F_score'] = rfm['F_score'].astype(int)
        rfm['M_score'] = rfm['M_score'].astype(int)
        rfm['RFM_score'] = rfm['R_score'] * 100 + rfm['F_score'] * 10 + rfm['M_score']

        # RFM segment
        def get_rfm_segment(row):
            if row['R_score'] >= 4 and row['F_score'] >= 4:
                return 'Champions'
            elif row['R_score'] >= 3 and row['F_score'] >= 3:
                return 'Loyal'
            elif row['R_score'] >= 4:
                return 'Recent'
            elif row['F_score'] >= 4:
                return 'Frequent'
            elif row['R_score'] <= 2 and row['F_score'] <= 2:
                return 'At Risk'
            else:
                return 'Regular'

        rfm['RFM_segment'] = rfm.apply(get_rfm_segment, axis=1)

        logger.info(f"Generated RFM features for {len(rfm):,} users")

        return rfm[['amplitude_id', 'recency', 'frequency', 'monetary',
                    'R_score', 'F_score', 'M_score', 'RFM_score', 'RFM_segment']]

    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Get summary statistics for features."""
        summary = {
            'n_users': len(features_df),
            'n_features': len(features_df.columns) - 1,
            'feature_stats': features_df.describe().to_dict()
        }
        return summary

    def print_feature_summary(self, features_df: pd.DataFrame) -> None:
        """Print formatted feature summary."""
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"Users: {len(features_df):,}")
        print(f"Features: {len(features_df.columns) - 1}")
        print("\nFeature statistics:")
        print(features_df.describe().round(2).to_string())
        print("=" * 60 + "\n")


# Convenience function
def build_features(df: pd.DataFrame, config_path: str = 'config.yaml') -> pd.DataFrame:
    """
    Quick function to build user features.

    Usage:
        from feature_engine import build_features
        features = build_features(df)
    """
    engine = FeatureEngine(config_path)
    return engine.build_user_features(df)


if __name__ == '__main__':
    # Test the feature engine
    import sys
    from data_loader import DataLoader

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '/Users/muhannadsaad/Desktop/investigation/dec_15/dec_15_25.parquet'

    # Load data
    loader = DataLoader()
    df = loader.load(data_path)

    # Build features
    engine = FeatureEngine()
    features = engine.build_user_features(df)
    engine.print_feature_summary(features)

    # Build RFM
    rfm = engine.build_rfm_features(df)
    print("\nRFM Segments:")
    print(rfm['RFM_segment'].value_counts())
