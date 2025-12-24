"""Recommendation-based CRM Segments.

Export user lists with personalized recommendations for marketing campaigns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class RecommendationSegments:
    """Build and export CRM segments with personalized recommendations."""

    def __init__(self, recommender, df: pd.DataFrame):
        """
        Args:
            recommender: Fitted recommendation model
            df: Full event data for user analysis
        """
        self.recommender = recommender
        self.df = df
        self._build_user_profiles()

    def _build_user_profiles(self):
        """Build user profiles for segmentation."""
        logger.info("Building user profiles for segmentation...")

        # Extract checkout data
        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()
        checkouts = self._extract_order_info(checkouts)

        # User order stats - handle case where order_total may not exist
        if 'order_total' in checkouts.columns and checkouts['order_total'].notna().any():
            self.user_stats = checkouts.groupby('amplitude_id').agg({
                'event_time': ['min', 'max', 'count'],
                'order_total': ['sum', 'mean']
            }).reset_index()
            self.user_stats.columns = [
                'amplitude_id', 'first_order', 'last_order', 'order_count',
                'total_revenue', 'avg_order_value'
            ]
        else:
            # No order_total available - just use counts
            self.user_stats = checkouts.groupby('amplitude_id').agg({
                'event_time': ['min', 'max', 'count']
            }).reset_index()
            self.user_stats.columns = [
                'amplitude_id', 'first_order', 'last_order', 'order_count'
            ]
            self.user_stats['total_revenue'] = 0
            self.user_stats['avg_order_value'] = 0

        # Calculate recency
        max_date = self.df['event_time'].max()
        if hasattr(max_date, 'tz') and max_date.tz is not None:
            max_date = max_date.tz_localize(None)

        self.user_stats['last_order'] = pd.to_datetime(self.user_stats['last_order'])
        if self.user_stats['last_order'].dt.tz is not None:
            self.user_stats['last_order'] = self.user_stats['last_order'].dt.tz_localize(None)

        self.user_stats['days_since_order'] = (
            max_date - self.user_stats['last_order']
        ).dt.days

        # User merchants
        self.user_merchants = (
            checkouts.groupby('amplitude_id')['merchant_id']
            .apply(set)
            .to_dict()
        )

        logger.info(f"Built profiles for {len(self.user_stats)} users")

    def _extract_order_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract order details from event_properties or direct columns."""
        df = df.copy()

        # Check if merchant info is in direct columns (Dec 9 format)
        if 'event_data_merchant_id' in df.columns:
            df['merchant_id'] = df['event_data_merchant_name']  # Use name as ID for compatibility
            df['merchant_name'] = df['event_data_merchant_name'] if 'event_data_merchant_name' in df.columns else 'Unknown'
            df['order_total'] = 0  # Not available in this format
            return df

        # Otherwise parse from event_properties (Dec 15 format)
        def parse_props(row):
            props = row.get('event_properties', {})
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except:
                    props = {}
            elif not isinstance(props, dict):
                props = {}

            return pd.Series({
                'merchant_id': props.get('merchant_name', props.get('merchant_id')),  # Prefer name for compatibility
                'merchant_name': props.get('merchant_name', 'Unknown'),
                'order_total': float(props.get('order_total', 0) or 0)
            })

        if 'event_properties' in df.columns:
            order_info = df.apply(parse_props, axis=1)
            df['merchant_id'] = order_info['merchant_id']
            df['merchant_name'] = order_info['merchant_name']
            df['order_total'] = order_info['order_total']

        return df

    def high_affinity_prospects(self, merchant_id: str,
                                 min_score: float = 0.5,
                                 max_users: int = 1000) -> pd.DataFrame:
        """Users likely to order from a merchant but haven't yet.

        Perfect for: New merchant promotions, targeted discounts.

        Args:
            merchant_id: Target merchant ID
            min_score: Minimum affinity score
            max_users: Maximum users to return

        Returns:
            DataFrame with user_id, score, and top recommendations
        """
        logger.info(f"Finding high affinity prospects for merchant {merchant_id}...")

        results = []

        # Get users who haven't ordered from this merchant
        for user_id in self.recommender.user_to_idx.keys():
            user_merchants = self.user_merchants.get(user_id, set())

            if merchant_id in user_merchants:
                continue  # Already ordered

            # Get recommendations for this user
            recs = self.recommender.recommend(user_id, n=20, exclude_ordered=True)

            # Check if target merchant is in recommendations
            for rec_merchant, score in recs:
                if rec_merchant == merchant_id and score >= min_score:
                    user_stats = self.user_stats[
                        self.user_stats['amplitude_id'] == user_id
                    ].iloc[0] if user_id in self.user_stats['amplitude_id'].values else None

                    results.append({
                        'amplitude_id': user_id,
                        'target_merchant_id': merchant_id,
                        'target_merchant_name': self.recommender.merchant_names.get(merchant_id, 'Unknown'),
                        'affinity_score': score,
                        'order_count': user_stats['order_count'] if user_stats is not None else 0,
                        'days_since_order': user_stats['days_since_order'] if user_stats is not None else None
                    })
                    break

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('affinity_score', ascending=False).head(max_users)

        logger.info(f"Found {len(df)} high affinity prospects")
        return df

    def lapsed_with_recommendations(self, min_days_inactive: int = 14,
                                    max_days_inactive: int = 60,
                                    n_recommendations: int = 3) -> pd.DataFrame:
        """Lapsed users with personalized re-engagement recommendations.

        Perfect for: Win-back campaigns with personalized merchant suggestions.

        Args:
            min_days_inactive: Minimum days since last order
            max_days_inactive: Maximum days (exclude very old users)
            n_recommendations: Number of recommendations per user

        Returns:
            DataFrame with user_id, days_inactive, and personalized recs
        """
        logger.info(f"Finding lapsed users ({min_days_inactive}-{max_days_inactive} days)...")

        # Filter lapsed users
        lapsed = self.user_stats[
            (self.user_stats['days_since_order'] >= min_days_inactive) &
            (self.user_stats['days_since_order'] <= max_days_inactive)
        ].copy()

        results = []

        for _, user in lapsed.iterrows():
            user_id = user['amplitude_id']

            if user_id not in self.recommender.user_to_idx:
                continue

            # Get recommendations
            recs = self.recommender.recommend(user_id, n=n_recommendations, exclude_ordered=True)

            rec_dict = {
                'amplitude_id': user_id,
                'days_since_order': user['days_since_order'],
                'order_count': user['order_count'],
                'total_revenue': user['total_revenue'],
                'avg_order_value': user['avg_order_value']
            }

            for i, (merchant_id, score) in enumerate(recs, 1):
                rec_dict[f'rec_{i}_id'] = merchant_id
                rec_dict[f'rec_{i}_name'] = self.recommender.merchant_names.get(merchant_id, 'Unknown')
                rec_dict[f'rec_{i}_score'] = score

            results.append(rec_dict)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('days_since_order', ascending=True)

        logger.info(f"Found {len(df)} lapsed users with recommendations")
        return df

    def category_explorers(self, target_merchants: List[str],
                           category_name: str,
                           min_score: float = 0.3,
                           max_users: int = 5000) -> pd.DataFrame:
        """Users who'd likely enjoy a category but haven't tried it.

        Perfect for: Category expansion campaigns, cuisine discovery.

        Args:
            target_merchants: List of merchant IDs in the target category
            category_name: Name for the category (for labeling)
            min_score: Minimum average affinity score
            max_users: Maximum users to return

        Returns:
            DataFrame with users likely to enjoy the category
        """
        logger.info(f"Finding users who'd enjoy {category_name}...")

        target_set = set(target_merchants)
        results = []

        for user_id in self.recommender.user_to_idx.keys():
            user_merchants = self.user_merchants.get(user_id, set())

            # Skip if user already orders from this category
            if user_merchants & target_set:
                continue

            # Get recommendations
            recs = self.recommender.recommend(user_id, n=50, exclude_ordered=True)

            # Calculate affinity to category
            category_scores = [
                score for merchant_id, score in recs
                if merchant_id in target_set
            ]

            if not category_scores:
                continue

            avg_score = np.mean(category_scores)
            max_score = max(category_scores)

            if avg_score >= min_score:
                user_stats = self.user_stats[
                    self.user_stats['amplitude_id'] == user_id
                ]

                results.append({
                    'amplitude_id': user_id,
                    'category': category_name,
                    'avg_category_score': avg_score,
                    'max_category_score': max_score,
                    'n_relevant_merchants': len(category_scores),
                    'order_count': user_stats['order_count'].iloc[0] if len(user_stats) > 0 else 0
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('avg_category_score', ascending=False).head(max_users)

        logger.info(f"Found {len(df)} potential {category_name} explorers")
        return df

    def merchant_superfans(self, merchant_id: str,
                           top_n: int = 100) -> pd.DataFrame:
        """Top users for a specific merchant by order count.

        Perfect for: Loyalty programs, VIP campaigns.

        Args:
            merchant_id: Target merchant ID
            top_n: Number of top users to return

        Returns:
            DataFrame with top users for the merchant
        """
        logger.info(f"Finding superfans for merchant {merchant_id}...")

        # Get order counts per user for this merchant
        checkouts = self.df[self.df['event_type'] == 'checkout_completed'].copy()
        checkouts = self._extract_order_info(checkouts)
        checkouts = checkouts[checkouts['merchant_id'] == merchant_id]

        if len(checkouts) == 0:
            logger.warning(f"No orders found for merchant {merchant_id}")
            return pd.DataFrame()

        merchant_stats = checkouts.groupby('amplitude_id').agg({
            'event_time': ['min', 'max', 'count'],
            'order_total': ['sum', 'mean']
        }).reset_index()

        merchant_stats.columns = [
            'amplitude_id', 'first_order', 'last_order', 'order_count',
            'total_spent', 'avg_order_value'
        ]

        merchant_stats['merchant_id'] = merchant_id
        merchant_stats['merchant_name'] = self.recommender.merchant_names.get(merchant_id, 'Unknown')

        # Sort by order count and total spent
        merchant_stats = merchant_stats.sort_values(
            ['order_count', 'total_spent'],
            ascending=[False, False]
        ).head(top_n)

        logger.info(f"Found {len(merchant_stats)} superfans")
        return merchant_stats

    def cross_sell_targets(self, source_merchant: str,
                           target_merchant: str,
                           min_score: float = 0.3,
                           max_users: int = 1000) -> pd.DataFrame:
        """Users who ordered from A and would likely enjoy B.

        Perfect for: Bundle promotions, partner campaigns.

        Args:
            source_merchant: Merchant user already ordered from
            target_merchant: Merchant to recommend
            min_score: Minimum affinity score
            max_users: Maximum users to return

        Returns:
            DataFrame with cross-sell targets
        """
        logger.info(f"Finding cross-sell targets: {source_merchant} -> {target_merchant}...")

        results = []

        for user_id in self.recommender.user_to_idx.keys():
            user_merchants = self.user_merchants.get(user_id, set())

            # Must have ordered from source, not from target
            if source_merchant not in user_merchants:
                continue
            if target_merchant in user_merchants:
                continue

            # Get recommendations
            recs = self.recommender.recommend(user_id, n=50, exclude_ordered=True)

            # Check affinity to target
            for rec_merchant, score in recs:
                if rec_merchant == target_merchant and score >= min_score:
                    user_stats = self.user_stats[
                        self.user_stats['amplitude_id'] == user_id
                    ]

                    results.append({
                        'amplitude_id': user_id,
                        'source_merchant_id': source_merchant,
                        'source_merchant_name': self.recommender.merchant_names.get(source_merchant, 'Unknown'),
                        'target_merchant_id': target_merchant,
                        'target_merchant_name': self.recommender.merchant_names.get(target_merchant, 'Unknown'),
                        'affinity_score': score,
                        'order_count': user_stats['order_count'].iloc[0] if len(user_stats) > 0 else 0
                    })
                    break

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('affinity_score', ascending=False).head(max_users)

        logger.info(f"Found {len(df)} cross-sell targets")
        return df

    def new_user_onboarding(self, max_orders: int = 2,
                            n_recommendations: int = 5) -> pd.DataFrame:
        """New users with personalized onboarding recommendations.

        Perfect for: Onboarding email sequences, new user engagement.

        Args:
            max_orders: Maximum order count to consider "new"
            n_recommendations: Number of recommendations per user

        Returns:
            DataFrame with new users and their recommendations
        """
        logger.info(f"Finding new users (≤{max_orders} orders)...")

        # Filter new users
        new_users = self.user_stats[
            self.user_stats['order_count'] <= max_orders
        ].copy()

        results = []

        for _, user in new_users.iterrows():
            user_id = user['amplitude_id']

            if user_id not in self.recommender.user_to_idx:
                continue

            # Get recommendations
            recs = self.recommender.recommend(user_id, n=n_recommendations, exclude_ordered=True)

            rec_dict = {
                'amplitude_id': user_id,
                'order_count': user['order_count'],
                'days_since_first_order': (
                    self.df['event_time'].max() - pd.to_datetime(user['first_order'])
                ).days if pd.notna(user['first_order']) else None,
                'avg_order_value': user['avg_order_value']
            }

            for i, (merchant_id, score) in enumerate(recs, 1):
                rec_dict[f'rec_{i}_id'] = merchant_id
                rec_dict[f'rec_{i}_name'] = self.recommender.merchant_names.get(merchant_id, 'Unknown')
                rec_dict[f'rec_{i}_score'] = score

            results.append(rec_dict)

        df = pd.DataFrame(results)

        logger.info(f"Found {len(df)} new users for onboarding")
        return df

    def export_segment(self, segment_df: pd.DataFrame,
                       segment_name: str,
                       output_dir: str = 'outputs/recommendations/exports') -> str:
        """Export a segment to CSV for CRM upload.

        Args:
            segment_df: Segment DataFrame to export
            segment_name: Name for the segment
            output_dir: Output directory

        Returns:
            Path to exported file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{segment_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        segment_df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(segment_df)} users to {filepath}")

        return filepath

    def get_segment_summary(self) -> pd.DataFrame:
        """Get summary of available segment types."""
        segments = [
            {
                'segment': 'high_affinity_prospects',
                'description': 'Users likely to order from specific merchant',
                'use_case': 'New merchant promotions',
                'required_params': 'merchant_id'
            },
            {
                'segment': 'lapsed_with_recommendations',
                'description': 'Inactive users with personalized recs',
                'use_case': 'Win-back campaigns',
                'required_params': 'min_days_inactive'
            },
            {
                'segment': 'category_explorers',
                'description': 'Users who\'d enjoy a new category',
                'use_case': 'Category expansion',
                'required_params': 'target_merchants, category_name'
            },
            {
                'segment': 'merchant_superfans',
                'description': 'Top users for a merchant',
                'use_case': 'Loyalty programs',
                'required_params': 'merchant_id'
            },
            {
                'segment': 'cross_sell_targets',
                'description': 'Users who ordered A, would like B',
                'use_case': 'Bundle promotions',
                'required_params': 'source_merchant, target_merchant'
            },
            {
                'segment': 'new_user_onboarding',
                'description': 'New users with discovery recs',
                'use_case': 'Onboarding sequences',
                'required_params': 'max_orders'
            }
        ]

        return pd.DataFrame(segments)
