"""Recommendation Engine - Core Algorithms.

Implements collaborative filtering approaches for merchant recommendations.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


class BaseRecommender:
    """Base class for recommendation algorithms."""

    def __init__(self):
        self.is_fitted = False
        self.user_ids = None
        self.merchant_ids = None
        self.merchant_names = {}

    def fit(self, df: pd.DataFrame) -> 'BaseRecommender':
        """Train the recommender on interaction data."""
        raise NotImplementedError

    def recommend(self, user_id: int, n: int = 10,
                  exclude_ordered: bool = True) -> List[Tuple[str, float]]:
        """Generate top-N recommendations for a user."""
        raise NotImplementedError

    def recommend_batch(self, user_ids: List[int], n: int = 10,
                        exclude_ordered: bool = True) -> pd.DataFrame:
        """Generate recommendations for multiple users."""
        results = []
        for user_id in user_ids:
            recs = self.recommend(user_id, n, exclude_ordered)
            for rank, (merchant_id, score) in enumerate(recs, 1):
                results.append({
                    'amplitude_id': user_id,
                    'merchant_id': merchant_id,
                    'merchant_name': self.merchant_names.get(merchant_id, 'Unknown'),
                    'score': score,
                    'rank': rank
                })
        return pd.DataFrame(results)

    def _build_interaction_matrix(self, df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """Build user-merchant interaction matrix from checkout data."""
        logger.info("Building user-merchant interaction matrix...")

        # Extract checkout events
        checkouts = df[df['event_type'] == 'checkout_completed'].copy()

        if len(checkouts) == 0:
            raise ValueError("No checkout_completed events found in data")

        # Parse merchant_id from event_properties
        checkouts = self._extract_merchant_info(checkouts)

        # Filter valid merchants
        checkouts = checkouts[checkouts['merchant_id'].notna()]

        logger.info(f"Found {len(checkouts)} orders across {checkouts['merchant_id'].nunique()} merchants")

        # Create user and merchant mappings
        unique_users = checkouts['amplitude_id'].unique()
        unique_merchants = checkouts['merchant_id'].unique()

        user_to_idx = {u: i for i, u in enumerate(unique_users)}
        merchant_to_idx = {m: i for i, m in enumerate(unique_merchants)}
        idx_to_user = {i: u for u, i in user_to_idx.items()}
        idx_to_merchant = {i: m for m, i in merchant_to_idx.items()}

        # Store merchant names
        merchant_name_map = checkouts.groupby('merchant_id')['merchant_name'].first().to_dict()

        # Build interaction counts
        interactions = checkouts.groupby(['amplitude_id', 'merchant_id']).size().reset_index(name='count')

        # Create sparse matrix
        rows = [user_to_idx[u] for u in interactions['amplitude_id']]
        cols = [merchant_to_idx[m] for m in interactions['merchant_id']]
        values = interactions['count'].values

        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(unique_users), len(unique_merchants))
        )

        logger.info(f"Built matrix: {matrix.shape[0]} users x {matrix.shape[1]} merchants")
        logger.info(f"Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2%}")

        mappings = {
            'user_to_idx': user_to_idx,
            'merchant_to_idx': merchant_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_merchant': idx_to_merchant,
            'merchant_names': merchant_name_map
        }

        return matrix, mappings, checkouts

    def _extract_merchant_info(self, df: pd.DataFrame, use_name_as_id: bool = True) -> pd.DataFrame:
        """Extract merchant_id and merchant_name from event_properties or direct columns.

        Args:
            df: DataFrame with checkout events
            use_name_as_id: If True, use merchant_name as merchant_id for cross-dataset compatibility
        """
        df = df.copy()

        # Check if merchant info is in direct columns (Dec 9 format)
        if 'event_data_merchant_id' in df.columns:
            df['merchant_id'] = df['event_data_merchant_id']
            df['merchant_name'] = df['event_data_merchant_name'] if 'event_data_merchant_name' in df.columns else 'Unknown'
        else:
            # Parse from event_properties (Dec 15 format)
            def parse_merchant(row):
                props = row.get('event_properties', {})
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except:
                        props = {}
                elif not isinstance(props, dict):
                    props = {}
                return pd.Series({
                    'merchant_id': props.get('merchant_id'),
                    'merchant_name': props.get('merchant_name', 'Unknown')
                })

            if 'event_properties' in df.columns:
                merchant_info = df.apply(parse_merchant, axis=1)
                df['merchant_id'] = merchant_info['merchant_id']
                df['merchant_name'] = merchant_info['merchant_name']

        # Use merchant_name as ID for cross-dataset compatibility
        if use_name_as_id and 'merchant_name' in df.columns:
            df['merchant_id'] = df['merchant_name']

        return df


class PopularityRecommender(BaseRecommender):
    """Baseline recommender using global popularity."""

    def __init__(self):
        super().__init__()
        self.popularity_scores = None
        self.user_history = {}

    def fit(self, df: pd.DataFrame) -> 'PopularityRecommender':
        """Fit popularity model."""
        logger.info("Fitting Popularity Recommender...")

        matrix, mappings, checkouts = self._build_interaction_matrix(df)

        self.user_to_idx = mappings['user_to_idx']
        self.idx_to_merchant = mappings['idx_to_merchant']
        self.merchant_to_idx = mappings['merchant_to_idx']
        self.merchant_names = mappings['merchant_names']
        self.matrix = matrix

        # Calculate popularity (order count per merchant)
        merchant_counts = np.array(matrix.sum(axis=0)).flatten()
        total_orders = merchant_counts.sum()
        self.popularity_scores = merchant_counts / total_orders

        # Store user history for exclusion
        for user_id, user_idx in self.user_to_idx.items():
            user_row = matrix[user_idx].toarray().flatten()
            self.user_history[user_id] = set(
                self.idx_to_merchant[i] for i in np.where(user_row > 0)[0]
            )

        self.is_fitted = True
        logger.info(f"Fitted on {len(self.user_to_idx)} users, {len(self.merchant_to_idx)} merchants")

        return self

    def recommend(self, user_id: int, n: int = 10,
                  exclude_ordered: bool = True) -> List[Tuple[str, float]]:
        """Recommend most popular merchants."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get all merchants sorted by popularity
        merchant_scores = [
            (self.idx_to_merchant[i], self.popularity_scores[i])
            for i in range(len(self.popularity_scores))
        ]
        merchant_scores.sort(key=lambda x: x[1], reverse=True)

        # Exclude already ordered
        if exclude_ordered and user_id in self.user_history:
            ordered = self.user_history[user_id]
            merchant_scores = [(m, s) for m, s in merchant_scores if m not in ordered]

        return merchant_scores[:n]


class ItemItemRecommender(BaseRecommender):
    """Item-Item Collaborative Filtering using cosine similarity."""

    def __init__(self, min_support: int = 5, n_neighbors: int = 20):
        """
        Args:
            min_support: Minimum orders for a merchant to be considered
            n_neighbors: Number of similar merchants to use for scoring
        """
        super().__init__()
        self.min_support = min_support
        self.n_neighbors = n_neighbors
        self.similarity_matrix = None
        self.user_history = {}

    def fit(self, df: pd.DataFrame) -> 'ItemItemRecommender':
        """Fit Item-Item collaborative filtering model."""
        logger.info("Fitting Item-Item Recommender...")

        matrix, mappings, checkouts = self._build_interaction_matrix(df)

        self.user_to_idx = mappings['user_to_idx']
        self.idx_to_user = mappings['idx_to_user']
        self.merchant_to_idx = mappings['merchant_to_idx']
        self.idx_to_merchant = mappings['idx_to_merchant']
        self.merchant_names = mappings['merchant_names']
        self.matrix = matrix

        # Filter merchants with minimum support
        merchant_counts = np.array(matrix.sum(axis=0)).flatten()
        valid_merchants = np.where(merchant_counts >= self.min_support)[0]
        logger.info(f"Merchants with >= {self.min_support} orders: {len(valid_merchants)}/{len(merchant_counts)}")

        # Compute item-item similarity (transpose matrix for item-based)
        # Each column is a merchant, each row is a user
        item_matrix = matrix.T.tocsr()  # merchants x users

        # Normalize rows (L2 norm) for cosine similarity
        from sklearn.preprocessing import normalize
        item_matrix_norm = normalize(item_matrix, norm='l2', axis=1)

        # Compute cosine similarity between items
        logger.info("Computing merchant similarity matrix...")
        self.similarity_matrix = cosine_similarity(item_matrix_norm)

        # Zero out self-similarity
        np.fill_diagonal(self.similarity_matrix, 0)

        # Store user history
        for user_id, user_idx in self.user_to_idx.items():
            user_row = matrix[user_idx].toarray().flatten()
            ordered_idx = np.where(user_row > 0)[0]
            self.user_history[user_id] = {
                'indices': set(ordered_idx),
                'merchants': set(self.idx_to_merchant[i] for i in ordered_idx),
                'weights': {i: user_row[i] for i in ordered_idx}
            }

        self.is_fitted = True
        logger.info(f"Fitted Item-Item model on {matrix.shape[0]} users, {matrix.shape[1]} merchants")

        return self

    def recommend(self, user_id: int, n: int = 10,
                  exclude_ordered: bool = True) -> List[Tuple[str, float]]:
        """Generate recommendations using item-item similarity."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_history:
            # Cold start - return empty or popularity fallback
            return []

        history = self.user_history[user_id]
        ordered_indices = list(history['indices'])
        weights = history['weights']

        # Score each merchant based on similarity to user's history
        n_merchants = self.similarity_matrix.shape[0]
        scores = np.zeros(n_merchants)

        for ordered_idx in ordered_indices:
            # Weight by number of times user ordered from this merchant
            order_weight = weights[ordered_idx]
            # Add similarity contribution
            scores += self.similarity_matrix[ordered_idx] * np.log1p(order_weight)

        # Normalize by number of items in history
        scores /= max(len(ordered_indices), 1)

        # Exclude already ordered merchants if requested
        if exclude_ordered:
            for idx in ordered_indices:
                scores[idx] = -1

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n]

        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:
                merchant_id = self.idx_to_merchant[idx]
                recommendations.append((merchant_id, float(scores[idx])))

        return recommendations

    def get_similar_merchants(self, merchant_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get merchants similar to a given merchant."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if merchant_id not in self.merchant_to_idx:
            return []

        merchant_idx = self.merchant_to_idx[merchant_id]
        similarities = self.similarity_matrix[merchant_idx]

        top_indices = np.argsort(similarities)[::-1][:n]

        similar = []
        for idx in top_indices:
            if similarities[idx] > 0:
                similar_id = self.idx_to_merchant[idx]
                similar.append((similar_id, float(similarities[idx])))

        return similar

    def explain_recommendation(self, user_id: int, merchant_id: str) -> Dict:
        """Explain why a merchant was recommended to a user."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_history:
            return {'error': 'User not found in training data'}

        if merchant_id not in self.merchant_to_idx:
            return {'error': 'Merchant not found in training data'}

        history = self.user_history[user_id]
        target_idx = self.merchant_to_idx[merchant_id]

        # Find which of user's past orders contributed to this recommendation
        contributions = []
        for ordered_idx in history['indices']:
            sim = self.similarity_matrix[target_idx, ordered_idx]
            if sim > 0:
                ordered_merchant = self.idx_to_merchant[ordered_idx]
                contributions.append({
                    'merchant_id': ordered_merchant,
                    'merchant_name': self.merchant_names.get(ordered_merchant, 'Unknown'),
                    'similarity': float(sim),
                    'order_count': history['weights'][ordered_idx]
                })

        contributions.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'recommended_merchant': merchant_id,
            'recommended_merchant_name': self.merchant_names.get(merchant_id, 'Unknown'),
            'based_on': contributions[:5],  # Top 5 contributing merchants
            'explanation': f"Recommended because you ordered from similar restaurants"
        }

    def get_stats(self) -> Dict:
        """Get model statistics."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        return {
            'n_users': len(self.user_to_idx),
            'n_merchants': len(self.merchant_to_idx),
            'matrix_density': self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]),
            'avg_orders_per_user': self.matrix.sum() / len(self.user_to_idx),
            'avg_users_per_merchant': self.matrix.sum() / len(self.merchant_to_idx),
            'min_support': self.min_support,
            'n_neighbors': self.n_neighbors
        }
