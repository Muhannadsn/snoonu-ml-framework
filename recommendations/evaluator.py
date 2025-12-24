"""Recommendation Evaluator - Accuracy Metrics.

Implements standard recommendation system evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluate recommendation quality with standard metrics."""

    def __init__(self):
        self.results = {}

    def precision_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K: What fraction of top-K recommendations were relevant?

        Args:
            recommended: Ordered list of recommended merchant IDs
            relevant: Set of actually ordered merchant IDs
            k: Number of recommendations to consider

        Returns:
            Precision score between 0 and 1
        """
        if k <= 0:
            return 0.0

        top_k = recommended[:k]
        hits = len(set(top_k) & relevant)
        return hits / k

    def recall_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K: What fraction of relevant items are in top-K?

        Args:
            recommended: Ordered list of recommended merchant IDs
            relevant: Set of actually ordered merchant IDs
            k: Number of recommendations to consider

        Returns:
            Recall score between 0 and 1
        """
        if not relevant:
            return 0.0

        top_k = recommended[:k]
        hits = len(set(top_k) & relevant)
        return hits / len(relevant)

    def hit_rate_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Hit Rate@K: Was there at least one relevant item in top-K?

        Args:
            recommended: Ordered list of recommended merchant IDs
            relevant: Set of actually ordered merchant IDs
            k: Number of recommendations to consider

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & relevant) > 0 else 0.0

    def ndcg_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K.

        Measures ranking quality - gives higher scores when relevant items
        appear earlier in the recommendation list.

        Args:
            recommended: Ordered list of recommended merchant IDs
            relevant: Set of actually ordered merchant IDs
            k: Number of recommendations to consider

        Returns:
            NDCG score between 0 and 1
        """
        if not relevant:
            return 0.0

        top_k = recommended[:k]

        # DCG: sum of relevance / log2(position + 1)
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because position is 1-indexed

        # Ideal DCG: perfect ranking (all relevant items first)
        n_relevant = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def average_precision(self, recommended: List[str], relevant: Set[str]) -> float:
        """Average Precision: Mean of precision at each relevant item position.

        Args:
            recommended: Ordered list of recommended merchant IDs
            relevant: Set of actually ordered merchant IDs

        Returns:
            AP score between 0 and 1
        """
        if not relevant:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for i, item in enumerate(recommended):
            if item in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        if hits == 0:
            return 0.0

        return sum_precisions / len(relevant)

    def coverage(self, all_recommendations: List[List[str]],
                 all_merchants: Set[str]) -> float:
        """Catalog Coverage: What fraction of merchants were ever recommended?

        Args:
            all_recommendations: List of recommendation lists for all users
            all_merchants: Set of all available merchant IDs

        Returns:
            Coverage score between 0 and 1
        """
        if not all_merchants:
            return 0.0

        recommended_merchants = set()
        for recs in all_recommendations:
            recommended_merchants.update(recs)

        return len(recommended_merchants) / len(all_merchants)

    def diversity(self, recommendations: List[str],
                  similarity_matrix: np.ndarray,
                  merchant_to_idx: Dict) -> float:
        """Intra-List Diversity: How different are the recommended items?

        Args:
            recommendations: List of recommended merchant IDs
            similarity_matrix: Merchant-merchant similarity matrix
            merchant_to_idx: Mapping from merchant ID to matrix index

        Returns:
            Diversity score (1 - average similarity)
        """
        if len(recommendations) < 2:
            return 1.0

        indices = [merchant_to_idx.get(m) for m in recommendations if m in merchant_to_idx]
        indices = [i for i in indices if i is not None]

        if len(indices) < 2:
            return 1.0

        # Calculate average pairwise similarity
        total_sim = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_sim += similarity_matrix[indices[i], indices[j]]
                count += 1

        avg_similarity = total_sim / count if count > 0 else 0.0

        # Diversity = 1 - similarity
        return 1.0 - avg_similarity

    def evaluate_user(self, recommended: List[str], relevant: Set[str],
                      k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate recommendations for a single user at multiple K values.

        Args:
            recommended: Ordered list of recommended merchant IDs
            relevant: Set of actually ordered merchant IDs
            k_values: List of K values to evaluate

        Returns:
            Dictionary with metrics at each K
        """
        results = {}

        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(recommended, relevant, k)
            results[f'recall@{k}'] = self.recall_at_k(recommended, relevant, k)
            results[f'hit_rate@{k}'] = self.hit_rate_at_k(recommended, relevant, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(recommended, relevant, k)

        results['map'] = self.average_precision(recommended, relevant)

        return results

    def evaluate_model(self, recommender, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       k_values: List[int] = [5, 10, 20],
                       sample_users: Optional[int] = None) -> Dict:
        """Full model evaluation with temporal train/test split.

        Args:
            recommender: Fitted recommender model
            train_df: Training data (historical orders)
            test_df: Test data (future orders)
            k_values: List of K values to evaluate
            sample_users: If set, evaluate on random sample of users

        Returns:
            Dictionary with aggregated metrics
        """
        logger.info("Evaluating recommendation model...")

        # Get test set ground truth (what users actually ordered in test period)
        test_checkouts = test_df[test_df['event_type'] == 'checkout_completed'].copy()
        test_checkouts = self._extract_merchant_info(test_checkouts)
        test_checkouts = test_checkouts[test_checkouts['merchant_id'].notna()]

        # Build user -> ordered merchants mapping for test set
        test_ground_truth = (
            test_checkouts.groupby('amplitude_id')['merchant_id']
            .apply(set)
            .to_dict()
        )

        logger.info(f"Test set: {len(test_ground_truth)} users with orders")

        # Get users who appear in both train and test
        train_users = set(recommender.user_to_idx.keys())
        test_users = set(test_ground_truth.keys())
        eval_users = list(train_users & test_users)

        logger.info(f"Users in both train and test: {len(eval_users)}")

        if sample_users and sample_users < len(eval_users):
            np.random.seed(42)
            eval_users = list(np.random.choice(eval_users, sample_users, replace=False))
            logger.info(f"Sampled {len(eval_users)} users for evaluation")

        # Evaluate each user
        all_metrics = {f'{metric}@{k}': [] for k in k_values
                       for metric in ['precision', 'recall', 'hit_rate', 'ndcg']}
        all_metrics['map'] = []
        all_recommendations = []

        for user_id in eval_users:
            # Get recommendations (excluding merchants from training)
            recs = recommender.recommend(user_id, n=max(k_values), exclude_ordered=True)
            rec_merchants = [m for m, _ in recs]
            all_recommendations.append(rec_merchants)

            # Get ground truth
            relevant = test_ground_truth[user_id]

            # Calculate metrics
            user_metrics = self.evaluate_user(rec_merchants, relevant, k_values)

            for key, value in user_metrics.items():
                all_metrics[key].append(value)

        # Aggregate metrics (mean across users)
        aggregated = {}
        for key, values in all_metrics.items():
            aggregated[f'mean_{key}'] = np.mean(values) if values else 0.0
            aggregated[f'std_{key}'] = np.std(values) if values else 0.0

        # Calculate coverage
        all_merchants = set(recommender.merchant_to_idx.keys())
        aggregated['coverage'] = self.coverage(all_recommendations, all_merchants)

        # Calculate average diversity (if model has similarity matrix)
        if hasattr(recommender, 'similarity_matrix') and recommender.similarity_matrix is not None:
            diversities = [
                self.diversity(recs, recommender.similarity_matrix, recommender.merchant_to_idx)
                for recs in all_recommendations
            ]
            aggregated['mean_diversity'] = np.mean(diversities)

        aggregated['n_eval_users'] = len(eval_users)

        self.results = aggregated

        # Log summary
        logger.info("\n=== Evaluation Results ===")
        for k in k_values:
            logger.info(f"  Precision@{k}: {aggregated[f'mean_precision@{k}']:.4f}")
            logger.info(f"  Recall@{k}: {aggregated[f'mean_recall@{k}']:.4f}")
            logger.info(f"  Hit Rate@{k}: {aggregated[f'mean_hit_rate@{k}']:.4f}")
            logger.info(f"  NDCG@{k}: {aggregated[f'mean_ndcg@{k}']:.4f}")
        logger.info(f"  MAP: {aggregated['mean_map']:.4f}")
        logger.info(f"  Coverage: {aggregated['coverage']:.4f}")
        if 'mean_diversity' in aggregated:
            logger.info(f"  Diversity: {aggregated['mean_diversity']:.4f}")

        return aggregated

    def _extract_merchant_info(self, df: pd.DataFrame, use_name_as_id: bool = True) -> pd.DataFrame:
        """Extract merchant_id from event_properties or direct columns.

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
            # Otherwise parse from event_properties (Dec 15 format)
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

    def compare_models(self, models: Dict[str, object],
                       train_df: pd.DataFrame, test_df: pd.DataFrame,
                       k_values: List[int] = [5, 10]) -> pd.DataFrame:
        """Compare multiple recommendation models.

        Args:
            models: Dictionary of {model_name: fitted_model}
            train_df: Training data
            test_df: Test data
            k_values: K values to evaluate

        Returns:
            DataFrame comparing model performance
        """
        results = []

        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(model, train_df, test_df, k_values)

            row = {'model': model_name}
            for k in k_values:
                row[f'P@{k}'] = metrics[f'mean_precision@{k}']
                row[f'R@{k}'] = metrics[f'mean_recall@{k}']
                row[f'HR@{k}'] = metrics[f'mean_hit_rate@{k}']
                row[f'NDCG@{k}'] = metrics[f'mean_ndcg@{k}']
            row['MAP'] = metrics['mean_map']
            row['Coverage'] = metrics['coverage']
            if 'mean_diversity' in metrics:
                row['Diversity'] = metrics['mean_diversity']

            results.append(row)

        comparison = pd.DataFrame(results)
        return comparison

    def get_results_summary(self) -> str:
        """Get formatted summary of last evaluation."""
        if not self.results:
            return "No evaluation results available. Run evaluate_model() first."

        lines = ["Recommendation Model Evaluation Summary", "=" * 40]

        # Extract K values from keys
        k_values = sorted(set(
            int(k.split('@')[1]) for k in self.results.keys()
            if '@' in k and k.startswith('mean_precision')
        ))

        for k in k_values:
            lines.append(f"\n@{k} Metrics:")
            lines.append(f"  Precision: {self.results.get(f'mean_precision@{k}', 0):.4f}")
            lines.append(f"  Recall:    {self.results.get(f'mean_recall@{k}', 0):.4f}")
            lines.append(f"  Hit Rate:  {self.results.get(f'mean_hit_rate@{k}', 0):.4f}")
            lines.append(f"  NDCG:      {self.results.get(f'mean_ndcg@{k}', 0):.4f}")

        lines.append(f"\nOverall:")
        lines.append(f"  MAP:       {self.results.get('mean_map', 0):.4f}")
        lines.append(f"  Coverage:  {self.results.get('coverage', 0):.4f}")
        if 'mean_diversity' in self.results:
            lines.append(f"  Diversity: {self.results['mean_diversity']:.4f}")
        lines.append(f"  Evaluated Users: {self.results.get('n_eval_users', 0)}")

        return "\n".join(lines)
