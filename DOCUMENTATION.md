# Snoonu ML Framework Documentation

## Overview

The Snoonu ML Framework is a comprehensive machine learning and analytics platform for food delivery consumer analytics. It provides:

- **Data Processing**: Load, validate, and transform Amplitude event data
- **Feature Engineering**: 50+ user-level features for ML models
- **ML Predictions**: Churn, Conversion, and LTV prediction models
- **Analytics Suite**: 14 specialized analytics modules
- **Recommendation Engine**: Personalized merchant recommendations
- **Interactive Dashboard**: Streamlit-based UI with 20+ analysis pages
- **CLI Tools**: Command-line interface for batch processing

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Frontend (Dashboard)](#frontend-dashboard)
4. [Backend (Analytics & ML)](#backend-analytics--ml)
5. [Data Schema](#data-schema)
6. [Configuration](#configuration)
7. [CLI Reference](#cli-reference)
8. [Output Files](#output-files)

---

## Quick Start

### Run the Dashboard
```bash
streamlit run app.py
```

### Run Analysis via CLI
```bash
# Exploratory data analysis
python run.py --data data/dec_15.parquet --task eda

# All analyses
python run.py --data data/dec_15.parquet --task all

# Churn prediction (requires historical + future data)
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn
```

---

## Architecture

```
snoonu-ml-framework/
├── app.py                 # Streamlit dashboard (frontend)
├── run.py                 # CLI entry point
├── config.yaml            # Configuration & schema
├── data_loader.py         # Data loading & validation
├── feature_engine.py      # Feature engineering
├── cohort_engine.py       # Cohort & segment analysis
├── models/                # ML prediction models
│   ├── base.py           # Base predictor class
│   ├── churn.py          # Churn prediction
│   ├── conversion.py     # Conversion prediction
│   └── ltv.py            # Lifetime value prediction
├── analytics/             # Analytics modules
│   ├── session_analytics.py
│   ├── merchant_intelligence.py
│   ├── promo_analytics.py
│   ├── search_analytics.py
│   ├── delivery_analytics.py
│   ├── customer_scoring.py
│   ├── anomaly_detection.py
│   ├── attribution.py
│   ├── reactivation.py
│   └── product_affinity.py
├── recommendations/       # Recommendation engine
│   ├── engine.py         # ItemItem & Popularity recommenders
│   ├── evaluator.py      # Model evaluation
│   ├── trending.py       # Trending & popularity
│   └── segments.py       # CRM segment recommendations
├── outputs/               # Generated outputs
└── data/                  # Data files (gitignored)
```

---

## Frontend (Dashboard)

The dashboard is built with **Streamlit** and provides 20+ interactive analysis pages.

### Running the Dashboard
```bash
streamlit run app.py
```

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Key metrics, event distribution, platform split |
| **Session Analytics** | Session duration, depth, bounce rate analysis |
| **Funnel Analysis** | Conversion funnel with drop-off analysis |
| **User Segments** | RFM segmentation and segment builder |
| **Cohort Analysis** | Retention heatmaps and LTV curves |
| **Recommendations** | User recommendations and merchant similarity |
| **Trending** | Popular merchants by time, platform, customer type |
| **Merchant Intelligence** | Merchant health scores and comparisons |
| **Promo Analytics** | Promotional effectiveness and ROI |
| **Search Analytics** | Search behavior and conversion |
| **Delivery Analytics** | Fulfillment times and zone performance |
| **Customer Scoring** | Multi-dimensional customer scores |
| **Anomaly Detection** | Unusual pattern detection |
| **Attribution Modeling** | Multi-touch attribution analysis |
| **Reactivation Targeting** | Dormant user identification |
| **Product Affinity** | Bundle recommendations and cross-sell |
| **Order Analysis** | Order metrics and distributions |
| **Platform Analysis** | iOS vs Android comparison |
| **Hourly Trends** | Activity patterns by hour |
| **ML Predictions** | Churn, conversion, LTV predictions (if available) |

### Data Source Options

The sidebar allows selecting data sources:
- **Default (Dec 15)**: Pre-configured path
- **Dec 9 Data**: Alternative dataset
- **Upload File**: Upload parquet/CSV
- **Enter Path**: Custom file path

### Key UI Components

```python
# Metrics cards
st.metric("Total Users", f"{users:,}")

# Charts (Plotly)
fig = px.bar(df, x='category', y='count')
st.plotly_chart(fig, use_container_width=True)

# Data tables
st.dataframe(df.style.format({'revenue': 'QAR {:.0f}'}))

# Filters
selected = st.selectbox("Select:", options)
```

---

## Backend (Analytics & ML)

### Data Loading (`data_loader.py`)

```python
from data_loader import DataLoader

loader = DataLoader()
df = loader.load('data/dec_15.parquet')

# Get summary
loader.print_summary(df)

# Filter events
orders = loader.get_order_events(df)
funnel = loader.get_funnel_events(df)
```

**Key Methods:**
| Method | Description |
|--------|-------------|
| `load(path)` | Load parquet/CSV with validation |
| `get_summary(df)` | Dict with event counts, users, date range |
| `filter_events(df, types)` | Filter to specific event types |
| `get_funnel_events(df)` | Get funnel progression events |
| `get_order_events(df)` | Get checkout and delivery events |
| `parse_event_properties(df)` | Parse JSON properties column |

### Feature Engineering (`feature_engine.py`)

```python
from feature_engine import FeatureEngine

engine = FeatureEngine()
user_features = engine.build_user_features(df)
rfm_features = engine.build_rfm_features(df)
```

**User Features (50+ columns):**
- **Recency**: days_since_first_event, days_since_last_event, days_since_last_order
- **Frequency**: total_events, total_orders, total_sessions, days_active
- **Monetary**: total_revenue, avg_order_value, max_order_value
- **Behavioral**: total_searches, total_product_views, products_added
- **Platform**: primary_platform, pct_ios
- **Funnel**: max_funnel_stage, user_conversion_rate

**RFM Features:**
- R_score, F_score, M_score (1-5 scale)
- RFM_score (combined)
- RFM_segment (Champions, Loyal, At Risk, etc.)

### ML Models (`models/`)

#### Churn Prediction
```python
from models.churn import ChurnPredictor

predictor = ChurnPredictor(model_type='random_forest')
result = predictor.run_prediction(df_history, df_future)

# Result contains:
# - metrics: accuracy, precision, recall, f1, auc_roc
# - feature_importance: top features
# - predictions: user_id, churn_probability, churn_prediction
```

#### Conversion Prediction
```python
from models.conversion import ConversionPredictor

predictor = ConversionPredictor(model_type='gradient_boosting')
result = predictor.run_prediction(df, level='user')
```

#### LTV Prediction
```python
from models.ltv import LTVPredictor

predictor = LTVPredictor(model_type='gradient_boosting')
result = predictor.run_prediction(df)

# Includes LTV tiers: Diamond, Gold, Silver, Bronze
```

### Analytics Modules (`analytics/`)

All analytics modules follow a similar pattern:

```python
from analytics import SessionAnalyzer, MerchantIntelligence, PromoAnalyzer

# Session Analysis
analyzer = SessionAnalyzer(df)
summary = analyzer.get_summary()
bounce_rate = analyzer.bounce_rate_by_platform()

# Merchant Intelligence
intel = MerchantIntelligence(df)
health_scores = intel.get_merchant_health_scores()
at_risk = intel.get_at_risk_merchants()

# Promo Analytics
promo = PromoAnalyzer(df)
effectiveness = promo.get_promo_effectiveness()
roi = promo.estimate_promo_roi()
```

**Available Modules:**

| Module | Class | Key Methods |
|--------|-------|-------------|
| session_analytics | SessionAnalyzer | get_summary, bounce_rate_by_platform |
| session_analytics | FunnelAnalyzer | get_funnel, get_drop_off_analysis |
| session_analytics | PathAnalyzer | get_common_paths, get_paths_to_conversion |
| merchant_intelligence | MerchantIntelligence | get_merchant_health_scores, get_at_risk_merchants |
| promo_analytics | PromoAnalyzer | get_promo_effectiveness, estimate_promo_roi |
| search_analytics | SearchAnalyzer | get_popular_searches, get_search_conversion |
| delivery_analytics | DeliveryAnalyzer | get_delivery_time_distribution, get_sla_compliance |
| customer_scoring | CustomerScorer | get_customer_scores, segment_by_score |
| anomaly_detection | AnomalyDetector | detect_user_anomalies, detect_merchant_anomalies |
| attribution | AttributionModeler | first_touch_attribution, compare_models |
| reactivation | ReactivationTargeter | identify_dormant_users, score_reactivation_potential |
| product_affinity | ProductAffinityAnalyzer | get_bundle_recommendations, calculate_association_rules |

### Recommendations (`recommendations/`)

```python
from recommendations import ItemItemRecommender, TrendingEngine

# Item-Item Collaborative Filtering
recommender = ItemItemRecommender(min_support=5)
recommender.fit(df)
recs = recommender.recommend(user_id=12345, n=10)
similar = recommender.get_similar_merchants(merchant_id='m123')

# Trending Analysis
trending = TrendingEngine(df)
popular = trending.popular_now(n=15)
by_time = trending.popular_by_time('dinner', n=10)
velocity = trending.trending_velocity(n=10)
```

### Cohort Analysis (`cohort_engine.py`)

```python
from cohort_engine import CohortEngine, SegmentBuilder

# Cohort Analysis
engine = CohortEngine()
cohorts = engine.build_cohorts(df, cohort_type='first_order_week')
retention = engine.calculate_retention(df, cohorts)
ltv_curves = engine.calculate_cumulative_ltv(df, cohorts)

# Segment Builder (Fluent API)
segment = SegmentBuilder(df)
users = (segment
    .ordered_at_least(2)
    .platform('ios')
    .active_in_last_n_days(30)
    .export())
```

---

## Data Schema

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `amplitude_id` | int64 | Unique user identifier |
| `event_type` | string | Event name (see Event Types below) |
| `event_time` | datetime | Timestamp of event |
| `platform` | string | 'ios' or 'android' |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `device_family` | string | Device type (iPhone, Android, etc.) |
| `os_name` | string | Operating system |
| `os_version` | string | OS version |
| `country` | string | User country |
| `event_properties` | string (JSON) | Event-specific properties |

### Event Types

**Funnel Events (in order):**
1. `homepage_viewed` - App entry point
2. `category_page_viewed` - Browse categories
3. `merchant_page_viewed` - View restaurant
4. `product_page_viewed` - View item
5. `search_made` - Search action
6. `product_added` - Add to cart
7. `cart_created` - Cart initialized
8. `cart_page_viewed` - View cart
9. `checkout_button_pressed` - Start checkout
10. `payment_initiated` - Enter payment
11. `checkout_completed` - Order placed
12. `order_delivered` - Order fulfilled

**Auth Events:**
- `user_signed_up`
- `phone_entered`
- `otp_entered`

### Event Properties (JSON)

**checkout_completed:**
```json
{
  "order_id": "12345",
  "order_total": 45.50,
  "order_type": "delivery",
  "merchant_id": "m123",
  "merchant_name": "Pizza Palace",
  "products_quantity": 3,
  "payment_method": "credit_card",
  "delivery_fee": 3.50,
  "promo_code": "SAVE10",
  "group_order": false,
  "multi_cart": false
}
```

**order_delivered:**
```json
{
  "order_id": "12345",
  "total_delivered_orders": 5,
  "fulfillment_time": 28,
  "order_status": "delivered"
}
```

**homepage_viewed:**
```json
{
  "first_session": true,
  "customer_status": "logged_in",
  "hour": 12,
  "zone_name": "Doha Downtown"
}
```

**product_page_viewed:**
```json
{
  "product_id": "p456",
  "product_name": "Margherita Pizza",
  "product_price": 12.99,
  "merchant_id": "m123"
}
```

### Sample Data Format

```
amplitude_id | event_type          | event_time           | platform | event_properties
-------------|---------------------|----------------------|----------|------------------
12345        | homepage_viewed     | 2024-12-15 10:00:00  | ios      | {"first_session": false}
12345        | merchant_page_viewed| 2024-12-15 10:02:00  | ios      | {"merchant_id": "m1"}
12345        | product_added       | 2024-12-15 10:05:00  | ios      | {"product_id": "p1"}
12345        | checkout_completed  | 2024-12-15 10:10:00  | ios      | {"order_total": 35.00}
```

---

## Configuration

### config.yaml Structure

```yaml
schema:
  required_columns:
    - amplitude_id
    - event_type
    - event_time
    - platform
  optional_columns:
    - device_family
    - os_name
    - country

events:
  funnel:
    - homepage_viewed
    - category_page_viewed
    - merchant_page_viewed
    # ... etc
  order:
    - checkout_completed
    - order_delivered

models:
  churn:
    churn_window_days: 30
    lookback_days: 90
    model_params:
      n_estimators: 100
      max_depth: 10

  conversion:
    session_timeout_minutes: 30

  ltv:
    prediction_window_days: 365

output:
  format: csv
  save_models: true
```

---

## CLI Reference

### Basic Usage
```bash
python run.py --data <path> --task <task> [options]
```

### Options
| Option | Description |
|--------|-------------|
| `--data` | Path to data file (required) |
| `--future` | Path to future data (for temporal models) |
| `--task` | Analysis task to run |
| `--config` | Config file path (default: config.yaml) |
| `--output` | Output directory (default: outputs) |

### Available Tasks

| Task | Description |
|------|-------------|
| `eda` | Exploratory data analysis |
| `features` | Build user features |
| `segment` | RFM segmentation |
| `churn` | Simple churn analysis |
| `conversion` | Funnel analysis |
| `cohort` | Cohort analysis |
| `recommend` | Train recommendation model |
| `trending` | Trending analysis |
| `predict_churn` | ML churn prediction (needs --future) |
| `predict_conversion` | ML conversion prediction |
| `predict_ltv` | ML LTV prediction |
| `predict_all` | All ML predictions |
| `all` | Run all analyses |

### Examples

```bash
# Quick EDA
python run.py --data data/dec_15.parquet --task eda

# Full analysis
python run.py --data data/dec_15.parquet --task all

# Churn prediction with temporal split
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn

# Recommendations with evaluation
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task recommend
```

---

## Output Files

### Default Output Directory: `outputs/`

| File | Description |
|------|-------------|
| `eda_report.yaml` | Summary statistics |
| `user_features.csv` | User-level features (50+ columns) |
| `rfm_features.csv` | RFM scores and segments |
| `customer_segments.csv` | Segment assignments |
| `churn_predictions_ml.csv` | Churn probabilities per user |
| `churn_feature_importance.csv` | Top churn predictors |
| `churn_at_risk_users.csv` | High-risk users (>70%) |
| `conversion_predictions.csv` | Conversion probabilities |
| `ltv_predictions.csv` | LTV predictions with tier |
| `cohort_retention.csv` | Retention heatmap data |
| `cohort_ltv_curves.csv` | Cumulative LTV per cohort |

### Recommendation Outputs: `outputs/recommendations/`

| File | Description |
|------|-------------|
| `item_item_model.pkl` | Trained recommender |
| `user_recommendations.parquet` | All user recommendations |
| `model_metrics.json` | Evaluation metrics |

### Trending Outputs: `outputs/trending/`

| File | Description |
|------|-------------|
| `popular_now.csv` | Current popular merchants |
| `popular_by_time.csv` | Popular by time slot |
| `trending_velocity.csv` | Fast-rising merchants |

---

## Performance Notes

- **Data Volume**: Handles 3.6M events, ~130k users per day
- **Processing**: Vectorized pandas operations
- **Caching**: Streamlit caches data loading and feature building
- **Models**: Scikit-learn handles 100k+ samples efficiently
- **Sparse Matrices**: Used for large user-merchant interaction matrices

---

## Dependencies

```
pandas>=1.5.0
pyarrow>=10.0.0
numpy>=1.23.0
scikit-learn>=1.2.0
pyyaml>=6.0
streamlit>=1.28.0
plotly>=5.18.0
```

Install with:
```bash
pip install -r requirements.txt
```
