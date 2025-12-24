# API Reference

Complete reference for all backend classes and methods in the Snoonu ML Framework.

---

## Table of Contents

1. [Data Loading](#data-loading)
2. [Feature Engineering](#feature-engineering)
3. [ML Models](#ml-models)
4. [Analytics Modules](#analytics-modules)
5. [Recommendations](#recommendations)
6. [Cohort & Segments](#cohort--segments)

---

## Data Loading

### `data_loader.DataLoader`

Load and validate Amplitude event data.

```python
from data_loader import DataLoader

loader = DataLoader(config_path='config.yaml')
df = loader.load('data/events.parquet')
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `load(data_path, validate=True)` | `data_path`: str, `validate`: bool | DataFrame | Load parquet/CSV with optional validation |
| `get_summary(df)` | `df`: DataFrame | dict | Summary stats (events, users, date range) |
| `print_summary(df)` | `df`: DataFrame | None | Print formatted summary |
| `filter_events(df, event_types)` | `df`: DataFrame, `event_types`: list | DataFrame | Filter to specific events |
| `get_funnel_events(df)` | `df`: DataFrame | DataFrame | Get funnel progression events |
| `get_order_events(df)` | `df`: DataFrame | DataFrame | Get checkout + delivery events |
| `parse_event_properties(df)` | `df`: DataFrame | DataFrame | Parse JSON properties to dict |
| `extract_property(df, prop, default)` | `df`: DataFrame, `prop`: str, `default`: any | Series | Extract single property value |

#### Example

```python
loader = DataLoader()
df = loader.load('data/dec_15.parquet')

# Get summary
summary = loader.get_summary(df)
# {'total_events': 3600000, 'unique_users': 130000, ...}

# Filter to orders only
orders = loader.get_order_events(df)
```

---

## Feature Engineering

### `feature_engine.FeatureEngine`

Build user-level and RFM features for ML models.

```python
from feature_engine import FeatureEngine

engine = FeatureEngine(config_path='config.yaml')
features = engine.build_user_features(df)
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `build_user_features(df, reference_date=None)` | `df`: DataFrame, `reference_date`: datetime | DataFrame | Build 50+ user features |
| `build_rfm_features(df, reference_date=None)` | `df`: DataFrame, `reference_date`: datetime | DataFrame | Build RFM scores and segments |
| `get_feature_summary(features_df)` | `features_df`: DataFrame | dict | Feature statistics |
| `print_feature_summary(features_df)` | `features_df`: DataFrame | None | Print formatted summary |

#### User Features Output

| Feature | Type | Description |
|---------|------|-------------|
| `amplitude_id` | int64 | User ID |
| `total_events` | int | Total events |
| `total_orders` | int | Order count |
| `total_revenue` | float | Total spend |
| `avg_order_value` | float | Average order |
| `days_since_first_event` | float | Account age |
| `days_since_last_event` | float | Recency |
| `days_since_last_order` | float | Order recency |
| `total_searches` | int | Search count |
| `total_product_views` | int | Product views |
| `total_sessions` | int | Session count |
| `days_active` | int | Active days |
| `primary_platform` | str | iOS/Android |
| `max_funnel_stage` | int | Deepest funnel stage |
| `user_conversion_rate` | float | Conversion rate |

#### RFM Features Output

| Feature | Type | Description |
|---------|------|-------------|
| `recency` | float | Days since last order |
| `frequency` | int | Order count |
| `monetary` | float | Total revenue |
| `R_score` | int | Recency score (1-5) |
| `F_score` | int | Frequency score (1-5) |
| `M_score` | int | Monetary score (1-5) |
| `RFM_score` | int | Combined score |
| `RFM_segment` | str | Segment name |

#### RFM Segments

| Segment | Criteria |
|---------|----------|
| Champions | R >= 4, F >= 4, M >= 4 |
| Loyal | F >= 4 |
| Recent | R >= 4 |
| Frequent | F >= 3 |
| At Risk | R <= 2, F >= 2 |
| Regular | All others |

---

## ML Models

### `models.churn.ChurnPredictor`

Predict customer churn probability.

```python
from models.churn import ChurnPredictor

predictor = ChurnPredictor(model_type='random_forest')
result = predictor.run_prediction(df_history, df_future)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | dict | None | Configuration dict |
| `model_type` | str | 'random_forest' | Model type: 'random_forest', 'gradient_boosting', 'logistic' |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `prepare_features(df)` | `df`: DataFrame | DataFrame | Build churn features |
| `prepare_labels(df_history, df_future)` | DataFrames | Series | Create churn labels |
| `run_prediction(df_history, df_future, test_size=0.2)` | DataFrames, float | dict | Full training pipeline |
| `predict(X)` | `X`: DataFrame | array | Predict on new data |
| `predict_proba(X)` | `X`: DataFrame | array | Get probabilities |
| `identify_at_risk_users(predictions, threshold=0.7)` | DataFrame, float | DataFrame | Filter high-risk users |
| `get_feature_importance(top_n=20)` | int | DataFrame | Top features |
| `save(path)` | str | None | Save model |
| `load(path)` | str | None | Load model |

#### Result Structure

```python
{
    'metrics': {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.78,
        'f1': 0.80,
        'auc_roc': 0.88
    },
    'feature_importance': DataFrame,
    'predictions': DataFrame  # amplitude_id, churn_probability, churn_prediction
}
```

---

### `models.conversion.ConversionPredictor`

Predict session/user conversion probability.

```python
from models.conversion import ConversionPredictor

predictor = ConversionPredictor(model_type='gradient_boosting')
result = predictor.run_prediction(df, level='user')
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | dict | None | Configuration |
| `model_type` | str | 'gradient_boosting' | Model type |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `prepare_features(df, level='user')` | DataFrame, str | DataFrame | Build features at user or session level |
| `prepare_labels(features, df=None)` | DataFrame | Series | Create conversion labels |
| `run_prediction(df, level='user', test_size=0.2)` | DataFrame, str, float | dict | Full pipeline |
| `score_realtime(session_events)` | DataFrame | float | Score single session |

#### Feature Categories

**User-Level Features:**
- Funnel depth (n_homepage_viewed, n_product_added, etc.)
- Session duration
- Search behavior
- Platform
- Time of day
- Merchant/product views

**Session-Level Features:**
- Events in session
- Unique event types
- Funnel progress markers
- Session duration

---

### `models.ltv.LTVPredictor`

Predict customer lifetime value.

```python
from models.ltv import LTVPredictor

predictor = LTVPredictor(model_type='gradient_boosting')
result = predictor.run_prediction(df)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | dict | None | Configuration |
| `model_type` | str | 'gradient_boosting' | 'random_forest', 'gradient_boosting', 'ridge' |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `prepare_features(df)` | DataFrame | DataFrame | Build LTV features |
| `prepare_labels(features)` | DataFrame | Series | Target = total_revenue |
| `run_prediction(df, test_size=0.2)` | DataFrame, float | dict | Full pipeline |
| `segment_by_ltv(predictions)` | DataFrame | dict | Create LTV tiers |

#### LTV Tiers

| Tier | Percentile | Description |
|------|------------|-------------|
| Diamond | Top 10% | Highest value |
| Gold | 70-90% | High value |
| Silver | 40-70% | Medium value |
| Bronze | Bottom 40% | Lower value |

#### Result Metrics

```python
{
    'mae': 25.50,      # Mean Absolute Error
    'rmse': 42.30,     # Root Mean Squared Error
    'r2': 0.65,        # R-squared
    'mean_actual': 85.00,
    'mean_predicted': 82.50
}
```

---

## Analytics Modules

All analytics modules are in the `analytics/` directory.

### `analytics.SessionAnalyzer`

Analyze user sessions.

```python
from analytics import SessionAnalyzer

analyzer = SessionAnalyzer(df, session_timeout_minutes=30)
summary = analyzer.get_summary()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_summary()` | dict | Session count, avg duration, bounce rate |
| `session_duration_distribution()` | DataFrame | Duration histogram |
| `bounce_rate_by_entry()` | DataFrame | Bounce rate by entry event |
| `bounce_rate_by_platform()` | DataFrame | Bounce rate by iOS/Android |
| `sessions_by_hour()` | DataFrame | Sessions by hour of day |
| `get_session_depth_analysis()` | DataFrame | Events per session |

---

### `analytics.FunnelAnalyzer`

Analyze conversion funnels.

```python
from analytics import FunnelAnalyzer

analyzer = FunnelAnalyzer(df)
funnel = analyzer.get_funnel()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_funnel()` | DataFrame | Step-by-step conversion rates |
| `get_funnel_by_segment(col)` | DataFrame | Funnel by segment |
| `get_drop_off_analysis()` | DataFrame | Drop-off between steps |
| `get_time_between_steps()` | DataFrame | Avg time between stages |
| `get_biggest_leaks()` | DataFrame | Biggest conversion drops |

---

### `analytics.PathAnalyzer`

Analyze user navigation paths.

```python
from analytics import PathAnalyzer

analyzer = PathAnalyzer(df, max_path_length=10)
paths = analyzer.get_common_paths()
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_common_paths(n=20, min_length=2)` | int, int | DataFrame | Most common sequences |
| `get_paths_to_conversion(n=20)` | int | DataFrame | Paths leading to checkout |
| `get_paths_to_drop_off(drop_after, n=15)` | str, int | DataFrame | Paths leading to abandonment |
| `get_entry_points()` | - | DataFrame | Common first events |
| `get_exit_points()` | - | DataFrame | Common last events |
| `get_event_flow()` | - | DataFrame | Event transitions |

---

### `analytics.MerchantIntelligence`

Analyze merchant performance.

```python
from analytics import MerchantIntelligence

intel = MerchantIntelligence(df)
health = intel.get_merchant_health_scores()
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_merchant_health_scores(top_n=50)` | int | DataFrame | Composite health score |
| `get_at_risk_merchants(min_orders=10)` | int | DataFrame | Declining merchants |
| `get_merchant_comparison(merchant_name)` | str | dict | Single merchant analysis |
| `get_category_performance()` | - | DataFrame | By restaurant category |
| `get_time_performance(merchant=None)` | str | DataFrame | By hour of day |
| `get_top_merchants_by_category(n=5)` | int | DataFrame | Category leaders |
| `get_new_merchants(days=7)` | int | DataFrame | Recently added |
| `get_merchant_customer_overlap(a, b)` | str, str | dict | Shared customers |
| `get_summary()` | - | dict | Overview metrics |

---

### `analytics.PromoAnalyzer`

Analyze promotional effectiveness.

```python
from analytics import PromoAnalyzer

promo = PromoAnalyzer(df)
effectiveness = promo.get_promo_effectiveness()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_promo_summary()` | dict | Promo usage overview |
| `get_promo_effectiveness()` | DataFrame | Revenue impact per code |
| `get_promo_user_segments()` | DataFrame | Promo vs non-promo users |
| `get_repeat_promo_users()` | DataFrame | Multi-promo users |
| `get_promo_timing()` | DataFrame | When promos are used |
| `get_promo_delivery_type()` | DataFrame | By delivery type |
| `get_promo_value_analysis()` | DataFrame | Discount amount impact |
| `estimate_promo_roi()` | dict | Revenue per $ spent |

---

### `analytics.SearchAnalyzer`

Analyze search behavior.

```python
from analytics import SearchAnalyzer

search = SearchAnalyzer(df)
popular = search.get_popular_searches()
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_search_summary()` | - | dict | Search overview |
| `get_popular_searches(top_n=20)` | int | DataFrame | Top searches |
| `get_search_conversion()` | - | DataFrame | Search to checkout rate |
| `get_search_to_checkout_time()` | - | DataFrame | Time from search to order |
| `get_search_bounce_rate()` | - | float | No follow-up rate |
| `get_trending_searches(hours=24)` | int | DataFrame | Recent trending |
| `get_search_success_rate()` | - | dict | Success metrics |
| `get_search_by_platform()` | - | DataFrame | iOS vs Android |

---

### `analytics.DeliveryAnalyzer`

Analyze delivery performance.

```python
from analytics import DeliveryAnalyzer

delivery = DeliveryAnalyzer(df)
summary = delivery.get_delivery_summary()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_delivery_summary()` | dict | Fulfillment rate, avg time |
| `get_delivery_time_distribution()` | DataFrame | Duration histogram |
| `get_delivery_performance_by_merchant()` | DataFrame | Per-merchant metrics |
| `get_delivery_zones()` | DataFrame | By zone/area |
| `get_delivery_time_of_day()` | DataFrame | By hour |
| `get_delivery_distance_analysis()` | DataFrame | Distance impact |
| `get_delivery_sla_compliance()` | dict | SLA adherence |
| `get_peak_delivery_hours()` | DataFrame | Busiest times |
| `get_delivery_cost_analysis()` | DataFrame | Fee impact |

---

### `analytics.CustomerScorer`

Multi-dimensional customer scoring.

```python
from analytics import CustomerScorer

scorer = CustomerScorer(df)
scores = scorer.get_customer_scores()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_customer_scores()` | DataFrame | All scores (0-100) |
| `segment_by_score()` | DataFrame | Customer tiers |
| `get_vip_customers()` | DataFrame | Top spenders |
| `get_high_engagement_low_spend()` | DataFrame | Upsell opportunities |
| `get_low_engagement_high_spend()` | DataFrame | Retention risks |
| `get_at_risk_customers()` | DataFrame | Churning customers |
| `get_customer_trajectories()` | DataFrame | Score changes over time |

#### Score Dimensions

| Score | Range | Based On |
|-------|-------|----------|
| Engagement | 0-100 | Session frequency, events/session |
| Value | 0-100 | Orders, revenue |
| Loyalty | 0-100 | Repeat rate, frequency |
| Health | 0-100 | Recency (inverse churn risk) |

---

### `analytics.AnomalyDetector`

Detect unusual patterns.

```python
from analytics import AnomalyDetector

detector = AnomalyDetector(df)
user_anomalies = detector.detect_user_anomalies()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `detect_user_anomalies()` | DataFrame | Unusual user behavior |
| `detect_merchant_anomalies()` | DataFrame | Unusual merchant activity |
| `detect_fraud_patterns()` | DataFrame | Potential fraud |
| `detect_systemic_issues()` | DataFrame | System-wide anomalies |

---

### `analytics.AttributionModeler`

Multi-touch attribution modeling.

```python
from analytics import AttributionModeler

modeler = AttributionModeler(df)
comparison = modeler.compare_models()
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `first_touch_attribution()` | - | DataFrame | Credit to first event |
| `last_touch_attribution()` | - | DataFrame | Credit to last event |
| `linear_attribution()` | - | DataFrame | Equal credit |
| `time_decay_attribution(half_life=6)` | float | DataFrame | More credit to recent |
| `position_based_attribution(first=0.4, last=0.4)` | float, float | DataFrame | Position-based |
| `compare_models()` | - | DataFrame | Compare all models |
| `get_journey_stats()` | - | dict | Journey statistics |
| `get_channel_influence()` | - | DataFrame | Channel contribution |

---

### `analytics.ReactivationTargeter`

Target dormant customers.

```python
from analytics import ReactivationTargeter

targeter = ReactivationTargeter(df, dormancy_days=14)
dormant = targeter.identify_dormant_users()
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Event data |
| `dormancy_days` | int | 14 | Days to consider dormant |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `identify_dormant_users()` | - | DataFrame | Dormant users |
| `score_reactivation_potential()` | - | DataFrame | Likelihood to return |
| `recommend_incentive(user)` | Series | dict | Suggested offer |
| `get_reactivation_campaign_segments()` | - | DataFrame | Campaign segments |
| `export_for_campaign(segment, limit)` | str, int | DataFrame | CRM export |
| `get_reactivation_summary()` | - | dict | Overview stats |

---

### `analytics.ProductAffinityAnalyzer`

Bundle and cross-sell analysis.

```python
from analytics import ProductAffinityAnalyzer

affinity = ProductAffinityAnalyzer(df)
bundles = affinity.get_bundle_recommendations()
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_merchant_affinity()` | - | DataFrame | Merchants ordered together |
| `get_product_pairs(min_support=2)` | int | DataFrame | Co-purchased products |
| `calculate_association_rules(support=0.01, confidence=0.1)` | float, float | DataFrame | Association rules |
| `get_bundle_recommendations(top_n=10)` | int | list | Top bundles |
| `get_cross_sell_recommendations(product, n=5)` | str, int | list | Cross-sell items |
| `get_merchant_top_bundles()` | - | DataFrame | Per-merchant bundles |
| `get_upsell_opportunities()` | - | DataFrame | Upsell suggestions |
| `get_affinity_summary()` | - | dict | Overview |

---

## Recommendations

### `recommendations.ItemItemRecommender`

Item-item collaborative filtering for merchants.

```python
from recommendations import ItemItemRecommender

recommender = ItemItemRecommender(min_support=5, n_neighbors=20)
recommender.fit(df)
recs = recommender.recommend(user_id=12345, n=10)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_support` | int | 5 | Min orders per merchant |
| `n_neighbors` | int | 20 | Similar merchants to consider |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit(df)` | DataFrame | self | Train on order data |
| `recommend(user_id, n=10, exclude=True)` | int, int, bool | list | Get recommendations |
| `recommend_batch(user_ids, n=10)` | list, int | dict | Batch recommendations |
| `get_similar_merchants(merchant_id, n=10)` | str, int | list | Similar merchants |
| `explain_recommendation(user_id, merchant_id)` | int, str | dict | Why recommended |
| `get_stats()` | - | dict | Model statistics |
| `save(path)` | str | None | Save model |
| `load(path)` | str | None | Load model |

---

### `recommendations.TrendingEngine`

Trending and popularity analysis.

```python
from recommendations import TrendingEngine

trending = TrendingEngine(df)
popular = trending.popular_now(n=15)
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_summary()` | - | dict | Overview stats |
| `popular_now(n=15)` | int | DataFrame | Top merchants |
| `popular_by_time(slot, n=5)` | str, int | DataFrame | By time slot |
| `popular_by_platform(platform, n=5)` | str, int | DataFrame | By platform |
| `trending_velocity(n=10)` | int | DataFrame | Fast-rising |
| `new_customer_favorites(n=10)` | int | DataFrame | New user favorites |
| `repeat_customer_favorites(n=10)` | int | DataFrame | High repeat rate |
| `weekend_vs_weekday(n=5)` | int | DataFrame | Weekend vs weekday |
| `contextual_recommendations(behavior, n=5)` | dict, int | DataFrame | Context-based |

#### Time Slots

| Slot | Hours |
|------|-------|
| `breakfast` | 6-11 |
| `lunch` | 11-14 |
| `afternoon` | 14-17 |
| `dinner` | 17-21 |
| `late_night` | 21-6 |

---

## Cohort & Segments

### `cohort_engine.CohortEngine`

Cohort analysis and retention.

```python
from cohort_engine import CohortEngine

engine = CohortEngine()
cohorts = engine.build_cohorts(df, cohort_type='first_order_week')
retention = engine.calculate_retention(df, cohorts)
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `build_cohorts(df, type)` | DataFrame, str | DataFrame | Assign cohorts |
| `calculate_retention(df, cohorts, period, max)` | DataFrame, DataFrame, str, int | DataFrame | Retention heatmap |
| `calculate_cumulative_ltv(df, cohorts, max)` | DataFrame, DataFrame, int | DataFrame | LTV curves |
| `calculate_time_to_nth_order(df, n=2)` | DataFrame, int | DataFrame | Days to Nth order |
| `calculate_order_frequency(df, cohorts)` | DataFrame, DataFrame | DataFrame | Orders per period |
| `get_cohort_summary(df, cohorts)` | DataFrame, DataFrame | DataFrame | Cohort metrics |

#### Cohort Types

| Type | Description |
|------|-------------|
| `first_order_week` | Week of first order |
| `first_order_month` | Month of first order |
| `first_visit_week` | Week of first visit |
| `platform` | iOS vs Android |
| `first_aov_tier` | First order AOV bucket |
| `first_order_type` | Delivery/takeaway/scheduled |

---

### `cohort_engine.SegmentBuilder`

Fluent API for custom segments.

```python
from cohort_engine import SegmentBuilder

segment = SegmentBuilder(df)
users = (segment
    .ordered_at_least(2)
    .platform('ios')
    .active_in_last_n_days(30)
    .get_users())
```

#### Methods (Chainable)

| Method | Parameters | Description |
|--------|------------|-------------|
| `did_event(type)` | str | Users who did event |
| `did_not_event(type)` | str | Users who didn't |
| `event_count(type, op, value)` | str, str, int | Filter by count |
| `ordered_exactly(n)` | int | Exactly N orders |
| `ordered_at_least(n)` | int | N+ orders |
| `ordered_less_than(n)` | int | < N orders |
| `never_ordered()` | - | Zero orders |
| `platform(p)` | str | iOS/Android |
| `active_in_last_n_days(n)` | int | Recent activity |
| `inactive_for_n_days(n)` | int | No recent activity |
| `first_order_between(start, end)` | date, date | Date range |
| `with_churn_risk(min)` | float | High churn probability |
| `with_ltv_tier(tier)` | str | LTV segment |

#### Terminal Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `count()` | int | User count |
| `get_users()` | list | User IDs |
| `export(path=None)` | DataFrame | Export to CSV |
| `describe()` | dict | Segment description |

---

### `cohort_engine.PredefinedSegments`

Common segments ready to use.

```python
from cohort_engine import PredefinedSegments

segments = PredefinedSegments()
cart_abandoners = segments.cart_abandoners(df)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `cart_abandoners(df)` | DataFrame | Added but no checkout |
| `one_time_buyers(df)` | DataFrame | Exactly 1 order |
| `repeat_buyers(df)` | DataFrame | 2+ orders |
| `power_users(df)` | DataFrame | 5+ orders |
| `browsers_not_buyers(df)` | DataFrame | Viewed but never ordered |
| `lapsed_customers(df, days=30)` | DataFrame | Ordered but inactive |
| `high_value_at_risk(df)` | DataFrame | Diamond + high churn |
| `engaged_non_converters(df)` | DataFrame | 3+ visits, no order |
| `ios_power_users(df)` | DataFrame | iOS + 3+ orders |
