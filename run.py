#!/usr/bin/env python3
"""
Snoonu ML Framework - CLI Entry Point
======================================
Run ML models and analysis on Snoonu event data.

Usage:
    python run.py --data data/dec_15.parquet --task churn
    python run.py --data data/dec_15.parquet --task segment
    python run.py --data data/dec_15.parquet --task eda
    python run.py --data data/dec_15.parquet --task all

    # With future data for temporal predictions:
    python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

from data_loader import DataLoader
from feature_engine import FeatureEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SnoonuML:
    """Main class for running ML tasks."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.loader = DataLoader(config_path)
        self.feature_engine = FeatureEngine(config_path)
        self.output_dir = Path(self.config.get('output', {}).get('dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration file."""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def run(self, data_path: str, task: str, future_path: str = None) -> None:
        """
        Run a specific task.

        Args:
            data_path: Path to data file
            task: Task to run (eda, churn, segment, conversion, all)
            future_path: Path to future data (for temporal prediction models)
        """
        logger.info(f"Starting task: {task}")
        logger.info(f"Data: {data_path}")

        # Load data
        df = self.loader.load(data_path)
        self.loader.print_summary(df)

        # Load future data if provided
        df_future = None
        if future_path:
            logger.info(f"Future data: {future_path}")
            df_future = self.loader.load(future_path)

        # Run task
        if task == 'eda':
            self._run_eda(df)
        elif task == 'segment':
            self._run_segmentation(df)
        elif task == 'churn':
            self._run_churn(df)
        elif task == 'conversion':
            self._run_conversion(df)
        elif task == 'features':
            self._run_features(df)
        elif task == 'predict_churn':
            self._run_churn_prediction(df, df_future)
        elif task == 'predict_conversion':
            self._run_conversion_prediction(df)
        elif task == 'predict_ltv':
            self._run_ltv_prediction(df)
        elif task == 'predict_all':
            if df_future is not None:
                self._run_churn_prediction(df, df_future)
            self._run_conversion_prediction(df)
            self._run_ltv_prediction(df)
        elif task == 'cohort':
            self._run_cohort_analysis(df)
        elif task == 'recommend':
            self._run_recommendations(df, df_future)
        elif task == 'trending':
            self._run_trending(df)
        elif task == 'all':
            self._run_eda(df)
            self._run_features(df)
            self._run_segmentation(df)
            self._run_cohort_analysis(df)
        else:
            logger.error(f"Unknown task: {task}")
            logger.info("Available tasks: eda, features, segment, churn, conversion, cohort, recommend, trending, predict_churn, predict_conversion, predict_ltv, all")
            return

        logger.info(f"Task '{task}' completed. Outputs saved to {self.output_dir}/")

    def _run_eda(self, df: pd.DataFrame) -> None:
        """Run exploratory data analysis."""
        print("\n" + "=" * 70)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 70)

        # Basic stats
        total_events = len(df)
        unique_users = df['amplitude_id'].nunique()

        print(f"\n[OVERVIEW]")
        print(f"  Total events: {total_events:,}")
        print(f"  Unique users: {unique_users:,}")
        print(f"  Events per user: {total_events / unique_users:.2f}")

        # Event breakdown
        print(f"\n[EVENT TYPES]")
        event_counts = df['event_type'].value_counts()
        for event, count in event_counts.items():
            pct = count / total_events * 100
            users = df[df['event_type'] == event]['amplitude_id'].nunique()
            print(f"  {event}: {count:,} ({pct:.1f}%) | {users:,} users")

        # Funnel metrics
        homepage_users = df[df['event_type'] == 'homepage_viewed']['amplitude_id'].nunique()
        checkout_users = df[df['event_type'] == 'checkout_completed']['amplitude_id'].nunique()
        total_orders = len(df[df['event_type'] == 'checkout_completed'])

        print(f"\n[FUNNEL METRICS]")
        print(f"  Homepage users: {homepage_users:,}")
        print(f"  Checkout users: {checkout_users:,}")
        print(f"  Total orders: {total_orders:,}")

        cvr = checkout_users / homepage_users * 100 if homepage_users > 0 else 0
        opu = total_orders / checkout_users if checkout_users > 0 else 0

        print(f"\n[KEY METRICS]")
        print(f"  CVR (user-based): {cvr:.2f}%")
        print(f"  Orders per user: {opu:.2f}")

        # Orders per user distribution
        orders_by_user = df[df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()

        print(f"\n[ORDER DISTRIBUTION]")
        print(f"  Users with 1 order: {(orders_by_user == 1).sum():,}")
        print(f"  Users with 2 orders: {(orders_by_user == 2).sum():,}")
        print(f"  Users with 3+ orders: {(orders_by_user >= 3).sum():,}")
        print(f"  Max orders (single user): {orders_by_user.max()}")

        # Platform
        print(f"\n[PLATFORM]")
        platform_counts = df['platform'].value_counts()
        for platform, count in platform_counts.items():
            pct = count / total_events * 100
            print(f"  {platform}: {count:,} ({pct:.1f}%)")

        print("=" * 70 + "\n")

        # Save EDA report
        eda_report = {
            'total_events': total_events,
            'unique_users': unique_users,
            'events_per_user': total_events / unique_users,
            'cvr': cvr,
            'opu': opu,
            'total_orders': total_orders,
            'event_counts': event_counts.to_dict(),
            'platform_counts': platform_counts.to_dict()
        }

        report_path = self.output_dir / 'eda_report.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(eda_report, f, default_flow_style=False)
        logger.info(f"EDA report saved to {report_path}")

    def _run_features(self, df: pd.DataFrame) -> None:
        """Generate and save user features."""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)

        # Build user features
        features = self.feature_engine.build_user_features(df)
        self.feature_engine.print_feature_summary(features)

        # Save features
        features_path = self.output_dir / 'user_features.csv'
        features.to_csv(features_path, index=False)
        logger.info(f"User features saved to {features_path}")

        # Build RFM features
        rfm = self.feature_engine.build_rfm_features(df)
        if len(rfm) > 0:
            print("\n[RFM SEGMENTS]")
            print(rfm['RFM_segment'].value_counts().to_string())

            rfm_path = self.output_dir / 'rfm_features.csv'
            rfm.to_csv(rfm_path, index=False)
            logger.info(f"RFM features saved to {rfm_path}")

        print("=" * 70 + "\n")

    def _run_segmentation(self, df: pd.DataFrame) -> None:
        """Run customer segmentation."""
        print("\n" + "=" * 70)
        print("CUSTOMER SEGMENTATION")
        print("=" * 70)

        # Get RFM features
        rfm = self.feature_engine.build_rfm_features(df)

        if len(rfm) == 0:
            logger.warning("No order data found for segmentation")
            return

        # Segment summary
        print("\n[SEGMENT DISTRIBUTION]")
        segment_counts = rfm['RFM_segment'].value_counts()
        for segment, count in segment_counts.items():
            pct = count / len(rfm) * 100
            print(f"  {segment}: {count:,} users ({pct:.1f}%)")

        # Segment profiles
        print("\n[SEGMENT PROFILES]")
        segment_profiles = rfm.groupby('RFM_segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'amplitude_id': 'count'
        }).round(2)
        segment_profiles.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary', 'Users']
        print(segment_profiles.to_string())

        # Save segments
        segments_path = self.output_dir / 'customer_segments.csv'
        rfm.to_csv(segments_path, index=False)
        logger.info(f"Customer segments saved to {segments_path}")

        print("=" * 70 + "\n")

    def _run_churn(self, df: pd.DataFrame) -> None:
        """Run churn prediction (placeholder for full model)."""
        print("\n" + "=" * 70)
        print("CHURN ANALYSIS")
        print("=" * 70)

        # Build features
        features = self.feature_engine.build_user_features(df)

        # Simple churn risk based on recency
        churn_window = self.config.get('models', {}).get('churn', {}).get('churn_window_days', 30)

        if 'days_since_last_order' in features.columns:
            features['churn_risk'] = 'Low'
            features.loc[features['days_since_last_order'] > churn_window / 2, 'churn_risk'] = 'Medium'
            features.loc[features['days_since_last_order'] > churn_window, 'churn_risk'] = 'High'
            features.loc[features['days_since_last_order'].isna(), 'churn_risk'] = 'No Orders'

            print("\n[CHURN RISK DISTRIBUTION]")
            risk_counts = features['churn_risk'].value_counts()
            for risk, count in risk_counts.items():
                pct = count / len(features) * 100
                print(f"  {risk}: {count:,} users ({pct:.1f}%)")

            # Save churn predictions
            churn_path = self.output_dir / 'churn_predictions.csv'
            features[['amplitude_id', 'days_since_last_order', 'total_orders', 'churn_risk']].to_csv(
                churn_path, index=False
            )
            logger.info(f"Churn predictions saved to {churn_path}")
        else:
            logger.warning("Could not calculate churn risk - missing recency data")

        print("=" * 70 + "\n")

    def _run_conversion(self, df: pd.DataFrame) -> None:
        """Run conversion analysis (placeholder for full model)."""
        print("\n" + "=" * 70)
        print("CONVERSION ANALYSIS")
        print("=" * 70)

        # Funnel analysis
        funnel_stages = [
            'homepage_viewed', 'product_page_viewed', 'product_added',
            'cart_page_viewed', 'checkout_button_pressed', 'checkout_completed'
        ]

        print("\n[FUNNEL CONVERSION]")
        prev_users = None
        for stage in funnel_stages:
            users = df[df['event_type'] == stage]['amplitude_id'].nunique()
            if prev_users:
                cvr = users / prev_users * 100
                drop = prev_users - users
                print(f"  {stage}: {users:,} users ({cvr:.1f}% from prev, -{drop:,} dropped)")
            else:
                print(f"  {stage}: {users:,} users")
            prev_users = users

        print("=" * 70 + "\n")

    def _run_churn_prediction(self, df_history: pd.DataFrame, df_future: pd.DataFrame = None) -> None:
        """Run ML-based churn prediction."""
        print("\n" + "=" * 70)
        print("CHURN PREDICTION MODEL")
        print("=" * 70)

        from models.churn import ChurnPredictor

        predictor = ChurnPredictor(config=self.config, model_type='random_forest')

        if df_future is None:
            # Without future data, use features-only approach
            logger.warning("No future data provided. Using feature-based prediction only.")
            features = predictor.prepare_features(df_history)
            features_path = self.output_dir / 'churn_features.csv'
            features.to_csv(features_path, index=False)
            logger.info(f"Churn features saved to {features_path}")
            print("\nTo get true churn labels, provide --future data file")
            return

        # Run full prediction with temporal split
        results = predictor.run_prediction(df_history, df_future)

        # Print results
        metrics = results['metrics']
        print(f"\n[MODEL PERFORMANCE]")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")

        print(f"\n[CONFUSION MATRIX]")
        cm = metrics['confusion_matrix']
        print(f"  True Neg:  {cm[0][0]:,} | False Pos: {cm[0][1]:,}")
        print(f"  False Neg: {cm[1][0]:,} | True Pos:  {cm[1][1]:,}")

        print(f"\n[TOP FEATURES]")
        for _, row in results['feature_importance'].head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Identify at-risk users
        at_risk = predictor.identify_at_risk_users(results['predictions'], threshold=0.7)
        print(f"\n[AT-RISK USERS]")
        print(f"  High churn risk (>70%): {len(at_risk):,} users")

        # Save outputs
        results['predictions'].to_csv(self.output_dir / 'churn_predictions_ml.csv', index=False)
        results['feature_importance'].to_csv(self.output_dir / 'churn_feature_importance.csv', index=False)
        at_risk.to_csv(self.output_dir / 'churn_at_risk_users.csv', index=False)
        predictor.save(self.output_dir / 'churn_model.pkl')

        logger.info("Churn prediction outputs saved to outputs/")
        print("=" * 70 + "\n")

    def _run_conversion_prediction(self, df: pd.DataFrame) -> None:
        """Run ML-based conversion prediction."""
        print("\n" + "=" * 70)
        print("CONVERSION PREDICTION MODEL")
        print("=" * 70)

        from models.conversion import ConversionPredictor

        predictor = ConversionPredictor(config=self.config, model_type='gradient_boosting')

        # Run prediction
        results = predictor.run_prediction(df, level='user')

        # Print results
        metrics = results['metrics']
        print(f"\n[MODEL PERFORMANCE]")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")

        print(f"\n[TOP FEATURES]")
        for _, row in results['feature_importance'].head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # High intent users
        predictions = results['predictions']
        high_intent = predictions[predictions['conversion_probability'] > 0.7]
        print(f"\n[HIGH INTENT USERS]")
        print(f"  Likely to convert (>70%): {len(high_intent):,} users")

        # Save outputs
        predictions.to_csv(self.output_dir / 'conversion_predictions.csv', index=False)
        results['feature_importance'].to_csv(self.output_dir / 'conversion_feature_importance.csv', index=False)
        predictor.save(self.output_dir / 'conversion_model.pkl')

        logger.info("Conversion prediction outputs saved to outputs/")
        print("=" * 70 + "\n")

    def _run_ltv_prediction(self, df: pd.DataFrame) -> None:
        """Run ML-based LTV prediction."""
        print("\n" + "=" * 70)
        print("LTV PREDICTION MODEL")
        print("=" * 70)

        from models.ltv import LTVPredictor

        predictor = LTVPredictor(config=self.config, model_type='gradient_boosting')

        # Run prediction
        results = predictor.run_prediction(df)

        # Print results
        metrics = results['metrics']
        print(f"\n[MODEL PERFORMANCE]")
        print(f"  MAE:  ${metrics['mae']:.2f}")
        print(f"  RMSE: ${metrics['rmse']:.2f}")
        print(f"  R²:   {metrics['r2']:.3f}")

        print(f"\n[TOP FEATURES]")
        for _, row in results['feature_importance'].head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Segment by LTV
        segmented = predictor.segment_by_ltv(results['predictions'])

        print(f"\n[LTV SEGMENTS]")
        segment_summary = segmented.groupby('ltv_tier').agg({
            'amplitude_id': 'count',
            'predicted_ltv': 'mean',
            'total_revenue': 'mean'
        }).round(2)
        segment_summary.columns = ['Users', 'Avg Predicted LTV', 'Avg Actual Revenue']
        print(segment_summary.to_string())

        # Save outputs
        segmented.to_csv(self.output_dir / 'ltv_predictions.csv', index=False)
        results['feature_importance'].to_csv(self.output_dir / 'ltv_feature_importance.csv', index=False)

        logger.info("LTV prediction outputs saved to outputs/")
        print("=" * 70 + "\n")

    def _run_cohort_analysis(self, df: pd.DataFrame) -> None:
        """Run cohort analysis."""
        print("\n" + "=" * 70)
        print("COHORT ANALYSIS")
        print("=" * 70)

        from cohort_engine import CohortEngine, SegmentBuilder, PredefinedSegments

        engine = CohortEngine(config=self.config)

        # Build cohorts by first order week
        cohorts = engine.build_cohorts(df, cohort_type='first_order_week')

        if len(cohorts) == 0:
            logger.warning("No order data found for cohort analysis")
            return

        print(f"\n[COHORTS]")
        print(f"  Total cohorts: {cohorts['cohort'].nunique()}")
        print(f"  Total users: {len(cohorts)}")

        # Cohort summary
        summary = engine.get_cohort_summary(df, cohorts)
        print(f"\n[COHORT SUMMARY]")
        print(summary.to_string(index=False))

        # Retention heatmap
        retention = engine.calculate_retention(df, cohorts, period='week', max_periods=8)
        if len(retention) > 0:
            print(f"\n[RETENTION HEATMAP (Weekly)]")
            print(retention.round(1).to_string())

        # Time to 2nd order
        time_to_2nd = engine.calculate_time_to_nth_order(df, n=2)
        if len(time_to_2nd) > 0:
            print(f"\n[TIME TO 2ND ORDER]")
            print(f"  Users with 2+ orders: {len(time_to_2nd):,}")
            print(f"  Median days: {time_to_2nd['days_to_nth_order'].median():.1f}")
            print(f"  Mean days: {time_to_2nd['days_to_nth_order'].mean():.1f}")
            print(f"  75th percentile: {time_to_2nd['days_to_nth_order'].quantile(0.75):.1f}")

        # LTV curves
        ltv_curves = engine.calculate_cumulative_ltv(df, cohorts, max_periods=8)

        # Predefined segments demo
        print(f"\n[PREDEFINED SEGMENTS]")
        segments = [
            ("Cart Abandoners", PredefinedSegments.cart_abandoners(df)),
            ("One-Time Buyers", PredefinedSegments.one_time_buyers(df)),
            ("Repeat Buyers (2+)", PredefinedSegments.repeat_buyers(df)),
            ("Power Users (5+)", PredefinedSegments.power_users(df)),
            ("Browsers (never bought)", PredefinedSegments.browsers_not_buyers(df)),
        ]

        for name, segment in segments:
            print(f"  {name}: {segment.count():,} users")

        # Save outputs
        cohorts.to_csv(self.output_dir / 'cohort_assignments.csv', index=False)
        summary.to_csv(self.output_dir / 'cohort_summary.csv', index=False)
        retention.to_csv(self.output_dir / 'cohort_retention.csv')
        time_to_2nd.to_csv(self.output_dir / 'time_to_2nd_order.csv', index=False)
        ltv_curves.to_csv(self.output_dir / 'cohort_ltv_curves.csv', index=False)

        logger.info("Cohort analysis outputs saved to outputs/")
        print("=" * 70 + "\n")

    def _run_recommendations(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> None:
        """Run recommendation engine with evaluation."""
        print("\n" + "=" * 70)
        print("RECOMMENDATION ENGINE")
        print("=" * 70)

        from recommendations import ItemItemRecommender, PopularityRecommender, RecommendationEvaluator, RecommendationSegments

        # Train recommender
        print("\n[TRAINING ITEM-ITEM RECOMMENDER]")
        recommender = ItemItemRecommender(min_support=3, n_neighbors=20)
        recommender.fit(df_train)

        # Print model stats
        stats = recommender.get_stats()
        print(f"  Users: {stats['n_users']:,}")
        print(f"  Merchants: {stats['n_merchants']:,}")
        print(f"  Matrix density: {stats['matrix_density']:.4%}")
        print(f"  Avg orders/user: {stats['avg_orders_per_user']:.2f}")

        # Evaluate if test data provided
        if df_test is not None:
            print("\n[MODEL EVALUATION]")
            evaluator = RecommendationEvaluator()

            # Also train popularity baseline for comparison
            popularity = PopularityRecommender()
            popularity.fit(df_train)

            # Compare models
            comparison = evaluator.compare_models(
                {'ItemItem': recommender, 'Popularity': popularity},
                df_train, df_test,
                k_values=[5, 10]
            )

            print("\n  Model Comparison:")
            print(comparison.to_string(index=False))

            # Detailed metrics for ItemItem
            metrics = evaluator.evaluate_model(recommender, df_train, df_test, k_values=[5, 10, 20])
            print(f"\n[DETAILED METRICS - ItemItem]")
            for k in [5, 10, 20]:
                print(f"\n  @{k}:")
                print(f"    Precision: {metrics[f'mean_precision@{k}']:.4f}")
                print(f"    Recall:    {metrics[f'mean_recall@{k}']:.4f}")
                print(f"    Hit Rate:  {metrics[f'mean_hit_rate@{k}']:.4f}")
                print(f"    NDCG:      {metrics[f'mean_ndcg@{k}']:.4f}")
            print(f"\n  Overall:")
            print(f"    MAP:       {metrics['mean_map']:.4f}")
            print(f"    Coverage:  {metrics['coverage']:.4f}")
            if 'mean_diversity' in metrics:
                print(f"    Diversity: {metrics['mean_diversity']:.4f}")

            # Save metrics
            import json
            metrics_path = self.output_dir / 'recommendations' / 'model_metrics.json'
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            print("\n[NOTE] Provide --future data for model evaluation")

        # Generate sample recommendations
        print("\n[SAMPLE RECOMMENDATIONS]")
        sample_users = list(recommender.user_to_idx.keys())[:5]
        for user_id in sample_users:
            recs = recommender.recommend(user_id, n=3)
            print(f"\n  User {user_id}:")
            for merchant_id, score in recs:
                name = recommender.merchant_names.get(merchant_id, 'Unknown')
                print(f"    - {name} (score: {score:.3f})")

        # Build CRM segments
        print("\n[CRM SEGMENTS]")
        segments = RecommendationSegments(recommender, df_train)

        # Lapsed users with recommendations
        lapsed = segments.lapsed_with_recommendations(min_days_inactive=7, max_days_inactive=30)
        print(f"  Lapsed users (7-30 days): {len(lapsed):,}")
        if len(lapsed) > 0:
            lapsed_path = segments.export_segment(lapsed, 'lapsed_with_recs',
                                                   str(self.output_dir / 'recommendations' / 'exports'))
            print(f"    Exported to: {lapsed_path}")

        # New user onboarding
        new_users = segments.new_user_onboarding(max_orders=2)
        print(f"  New users (≤2 orders): {len(new_users):,}")
        if len(new_users) > 0:
            new_path = segments.export_segment(new_users, 'new_user_onboarding',
                                                str(self.output_dir / 'recommendations' / 'exports'))
            print(f"    Exported to: {new_path}")

        # Save model and all user recommendations
        import pickle
        model_path = self.output_dir / 'recommendations' / 'item_item_model.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(recommender, f)
        print(f"\n[MODEL SAVED] {model_path}")

        # Generate recommendations for all users
        all_users = list(recommender.user_to_idx.keys())
        all_recs = recommender.recommend_batch(all_users, n=10)
        recs_path = self.output_dir / 'recommendations' / 'user_recommendations.parquet'
        all_recs.to_parquet(recs_path, index=False)
        print(f"[RECOMMENDATIONS SAVED] {recs_path}")

        logger.info("Recommendation outputs saved to outputs/recommendations/")
        print("=" * 70 + "\n")

    def _run_trending(self, df: pd.DataFrame) -> None:
        """Run trending analysis - works great with limited data."""
        print("\n" + "=" * 70)
        print("TRENDING & POPULARITY ANALYSIS")
        print("=" * 70)

        from recommendations import TrendingEngine

        engine = TrendingEngine(df)

        # Summary
        summary = engine.get_summary()
        print(f"\n[DATA SUMMARY]")
        print(f"  Orders: {summary['total_orders']:,}")
        print(f"  Merchants: {summary['unique_merchants']:,}")
        print(f"  Customers: {summary['unique_customers']:,}")
        print(f"  Date range: {summary['date_range']}")

        # Popular Now
        print(f"\n[POPULAR NOW - Top 15]")
        popular = engine.popular_now(n=15)
        for _, row in popular.iterrows():
            print(f"  {row['rank']:2}. {row['merchant'][:40]:<40} | {row['order_count']:,} orders | {row['unique_customers']:,} customers")

        # Time-based popularity
        print(f"\n[POPULAR BY TIME SLOT]")
        for slot in ['breakfast', 'lunch', 'dinner', 'late_night']:
            time_pop = engine.popular_by_time(slot, n=5)
            if len(time_pop) > 0:
                print(f"\n  {slot.upper()}:")
                for _, row in time_pop.iterrows():
                    print(f"    {row['rank']}. {row['merchant'][:35]:<35} ({row['order_count']} orders)")

        # Platform popularity
        print(f"\n[POPULAR BY PLATFORM]")
        for platform in ['ios', 'android']:
            plat_pop = engine.popular_by_platform(platform, n=5)
            if len(plat_pop) > 0:
                print(f"\n  {platform.upper()}:")
                for _, row in plat_pop.iterrows():
                    print(f"    {row['rank']}. {row['merchant'][:35]:<35} ({row['order_count']} orders)")

        # Trending velocity
        print(f"\n[TRENDING NOW - Highest Velocity]")
        trending = engine.trending_velocity(n=10)
        for _, row in trending.iterrows():
            print(f"  {row['rank']:2}. {row['merchant'][:40]:<40} | {row['orders_per_hour']:.1f} orders/hr")

        # New customer favorites
        print(f"\n[NEW CUSTOMER FAVORITES]")
        new_faves = engine.new_customer_favorites(n=10)
        for _, row in new_faves.iterrows():
            print(f"  {row['rank']:2}. {row['merchant'][:40]:<40} | {row['new_customer_orders']:,} new customers ({row['new_customer_pct']:.0f}%)")

        # Repeat customer favorites
        print(f"\n[HIGHEST REPEAT RATES]")
        repeat_faves = engine.repeat_customer_favorites(n=10)
        for _, row in repeat_faves.iterrows():
            print(f"  {row['rank']:2}. {row['merchant'][:40]:<40} | {row['repeat_rate']:.0f}% repeat rate ({row['repeat_customers']}/{row['total_customers']} customers)")

        # Weekend vs Weekday
        print(f"\n[WEEKEND VS WEEKDAY]")
        wknd_wkdy = engine.weekend_vs_weekday(n=5)
        for period, data in wknd_wkdy.items():
            if len(data) > 0:
                print(f"\n  {period.upper()}:")
                for _, row in data.iterrows():
                    print(f"    {row['rank']}. {row['merchant'][:35]:<35} ({row['order_count']} orders)")

        # Save outputs
        trending_dir = self.output_dir / 'trending'
        trending_dir.mkdir(parents=True, exist_ok=True)

        popular.to_csv(trending_dir / 'popular_now.csv', index=False)
        trending.to_csv(trending_dir / 'trending_velocity.csv', index=False)
        new_faves.to_csv(trending_dir / 'new_customer_favorites.csv', index=False)
        repeat_faves.to_csv(trending_dir / 'repeat_customer_favorites.csv', index=False)

        # Save time-based
        all_time_slots = []
        for slot in ['breakfast', 'lunch', 'afternoon', 'dinner', 'late_night']:
            time_pop = engine.popular_by_time(slot, n=10)
            if len(time_pop) > 0:
                all_time_slots.append(time_pop)
        if all_time_slots:
            pd.concat(all_time_slots).to_csv(trending_dir / 'popular_by_time.csv', index=False)

        logger.info(f"Trending outputs saved to {trending_dir}/")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Snoonu ML Framework - Run ML models on event data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run.py --data data/dec_15.parquet --task eda
  python run.py --data data/dec_15.parquet --task segment

  # Prediction models
  python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn
  python run.py --data data/dec_15.parquet --task predict_conversion
  python run.py --data data/dec_15.parquet --task predict_ltv

Available tasks:
  eda                - Exploratory data analysis
  features           - Generate user features
  segment            - Customer segmentation (RFM)
  churn              - Simple churn risk analysis
  conversion         - Conversion funnel analysis
  cohort             - Cohort analysis with retention & LTV curves
  recommend          - Recommendation engine with evaluation
  trending           - Trending & popularity analysis (works with limited data!)
  predict_churn      - ML churn prediction (needs --future)
  predict_conversion - ML conversion prediction
  predict_ltv        - ML lifetime value prediction
  predict_all        - Run all prediction models
  all                - Run all basic analyses
        """
    )

    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to data file (parquet or csv)'
    )

    parser.add_argument(
        '--future', '-f',
        help='Path to future data file (for temporal prediction models)'
    )

    parser.add_argument(
        '--task', '-t',
        default='eda',
        choices=['eda', 'features', 'segment', 'churn', 'conversion', 'cohort', 'recommend', 'trending',
                 'predict_churn', 'predict_conversion', 'predict_ltv', 'predict_all', 'all'],
        help='Task to run (default: eda)'
    )

    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--output', '-o',
        default='outputs',
        help='Output directory (default: outputs)'
    )

    args = parser.parse_args()

    # Check data file exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    # Check future file if provided
    if args.future and not Path(args.future).exists():
        logger.error(f"Future data file not found: {args.future}")
        sys.exit(1)

    # Run
    ml = SnoonuML(config_path=args.config)
    ml.run(data_path=args.data, task=args.task, future_path=args.future)


if __name__ == '__main__':
    main()
