# Recommendations Engine Project

## Project Overview

Build a personalized merchant recommendation system for Snoonu that:
- Predicts which merchants a user is likely to order from
- Measures recommendation accuracy with industry-standard metrics
- Exports actionable user segments for CRM campaigns

## Business Goals

| Goal | Success Metric |
|------|----------------|
| Increase discovery | Users trying new merchants |
| Reduce churn | Personalized win-back campaigns |
| Boost order frequency | Relevant re-engagement |
| Support marketing | Exportable targeting lists |

---

## Data Requirements

### Input Data
- **Primary**: `checkout_completed` events with `merchant_id`, `merchant_name`, `order_total`
- **Secondary**: `merchant_page_viewed`, `product_page_viewed` for implicit signals
- **User IDs**: `amplitude_id` as unique identifier

### Data Schema (from checkout_completed)
```
amplitude_id    → User identifier
merchant_id     → Merchant identifier
merchant_name   → Merchant display name
order_total     → Order value (for weighting)
event_time      → Timestamp (for temporal split)
```

---

## Recommendation Algorithms

### Phase 1: Item-Item Collaborative Filtering
**Approach**: Merchants frequently ordered by the same users are "similar"

```
Similarity(Merchant_A, Merchant_B) =
    Users who ordered both / sqrt(Users who ordered A × Users who ordered B)
```

**Why start here**:
- Works well with sparse data
- Interpretable ("similar to your favorites")
- No cold-start for existing merchants
- Fast to compute

### Phase 2: Matrix Factorization (SVD/ALS)
**Approach**: Decompose user-merchant matrix into latent factors

```
R ≈ U × V^T
Where:
  R = User-Merchant interaction matrix
  U = User latent factors
  V = Merchant latent factors
```

**Benefits**:
- Captures hidden patterns
- Better generalization
- Handles sparsity well

### Phase 3: Hybrid Model
**Approach**: Combine multiple signals with learned weights

```
Final_Score = w1 × ItemItem_Score
            + w2 × MatrixFactor_Score
            + w3 × Popularity_Score
            + w4 × Recency_Boost
```

---

## Evaluation Metrics

### Offline Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Precision@5** | `(relevant ∩ top5) / 5` | > 0.15 |
| **Recall@10** | `(relevant ∩ top10) / total_relevant` | > 0.25 |
| **Hit Rate@5** | `users_with_hit / total_users` | > 0.40 |
| **NDCG@10** | Normalized Discounted Cumulative Gain | > 0.30 |
| **Coverage** | `unique_recommended / total_merchants` | > 0.50 |
| **Diversity** | Avg intra-list distance | > 0.60 |

### Evaluation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    TEMPORAL SPLIT                           │
├─────────────────────────────┬───────────────────────────────┤
│   TRAINING PERIOD           │   TEST PERIOD                 │
│   (Dec 1-9)                 │   (Dec 10-15)                 │
│                             │                               │
│   • Build user profiles     │   • Generate recommendations  │
│   • Learn merchant          │   • Check: Did user order     │
│     similarities            │     any recommended merchant? │
│   • Train models            │   • Calculate metrics         │
└─────────────────────────────┴───────────────────────────────┘
```

### Why Temporal Split?
- Simulates real-world deployment
- Prevents data leakage
- Tests actual predictive power

---

## CRM Export Segments

### Segment Definitions

| Segment Name | Logic | Export Fields | CRM Use Case |
|--------------|-------|---------------|--------------|
| **High Affinity Prospects** | Score > 0.7 for Merchant X, never ordered | user_id, merchant_id, merchant_name, affinity_score | Targeted new merchant promo |
| **Lapsed + Personalized** | No order in 14+ days + top 3 recs | user_id, days_since_order, rec_1, rec_2, rec_3 | Win-back with personalization |
| **Category Explorers** | Likely to enjoy Category X, never tried | user_id, category, top_merchants, score | Category expansion campaign |
| **Merchant Superfans** | Top N users by order count for Merchant X | user_id, merchant_id, order_count, total_spent | Loyalty program targeting |
| **Cross-sell Targets** | Ordered A, high affinity for B | user_id, ordered_merchant, recommended_merchant, score | Bundle/cross-sell promo |
| **New User Onboarding** | < 2 orders + personalized recs | user_id, order_count, rec_1, rec_2, rec_3 | Onboarding sequence |

### Export Format
```csv
amplitude_id,segment,recommended_merchant_id,recommended_merchant_name,score,export_date
123456,high_affinity_prospect,M001,Pizza Hut,0.85,2024-12-15
123456,high_affinity_prospect,M002,KFC,0.72,2024-12-15
```

---

## Implementation Plan

### Phase 1: Core Engine (MVP)
- [ ] Create `recommendations/` module structure
- [ ] Build user-merchant interaction matrix from checkout data
- [ ] Implement Item-Item similarity calculator
- [ ] Generate top-N recommendations per user
- [ ] Basic evaluation (Precision@K, Hit Rate)

### Phase 2: Evaluation & Tuning
- [ ] Implement full evaluation suite (NDCG, Coverage, Diversity)
- [ ] Temporal train/test split framework
- [ ] Hyperparameter tuning (similarity threshold, N neighbors)
- [ ] Baseline comparisons (popularity, random)

### Phase 3: CRM Integration
- [ ] Build segment definitions
- [ ] Export functionality with filters
- [ ] Dashboard integration (user lookup, segment builder)
- [ ] Scheduled export capability

### Phase 4: Advanced Models
- [ ] Matrix factorization (SVD/ALS)
- [ ] Implicit feedback weighting
- [ ] Hybrid model with learned weights
- [ ] Real-time scoring API (optional)

---

## File Structure

```
snoonu-ml-framework/
├── recommendations/
│   ├── __init__.py
│   ├── engine.py              # Core recommendation algorithms
│   │   ├── BaseRecommender
│   │   ├── ItemItemRecommender
│   │   ├── MatrixFactorizationRecommender
│   │   └── HybridRecommender
│   ├── evaluator.py           # Accuracy metrics
│   │   ├── precision_at_k()
│   │   ├── recall_at_k()
│   │   ├── ndcg_at_k()
│   │   ├── hit_rate_at_k()
│   │   ├── coverage()
│   │   └── diversity()
│   ├── segments.py            # CRM segment builders
│   │   ├── high_affinity_prospects()
│   │   ├── lapsed_with_recommendations()
│   │   ├── category_explorers()
│   │   ├── merchant_superfans()
│   │   └── cross_sell_targets()
│   └── utils.py               # Helper functions
├── run.py                     # Add: --task recommend
├── app.py                     # Add: Recommendations page
└── outputs/
    └── recommendations/
        ├── model_metrics.json
        ├── user_recommendations.parquet
        └── exports/
            └── {segment_name}_{date}.csv
```

---

## Dashboard Features

### Tab 1: Model Performance
- Precision/Recall curves at different K values
- Coverage and diversity stats
- Comparison vs baselines (popularity, random)
- Temporal performance trends

### Tab 2: User Lookup
- Enter amplitude_id → see personalized recommendations
- Show order history alongside recommendations
- Explain why each merchant was recommended

### Tab 3: Merchant Analysis
- Select merchant → see potential target users
- Affinity score distribution
- Competitor analysis (similar merchants)

### Tab 4: CRM Export
- Select segment type from dropdown
- Apply filters (min score, platform, etc.)
- Preview sample users
- Export to CSV with one click

---

## Success Criteria

### Technical
- [ ] Precision@5 > 0.15 (15% of top 5 recs are ordered)
- [ ] Hit Rate@5 > 0.40 (40% of users order at least 1 rec)
- [ ] Coverage > 0.50 (50% of merchants get recommended)
- [ ] Model training < 5 minutes on full dataset

### Business
- [ ] CRM team can export segments independently
- [ ] Segments are actionable (clear targeting logic)
- [ ] Recommendations are explainable to stakeholders

---

## Progress Log

| Date | Milestone | Status | Notes |
|------|-----------|--------|-------|
| 2024-12-23 | Project kickoff | Done | Created project spec |
| 2024-12-23 | Phase 1: Core Engine | Done | Item-Item CF + Popularity baseline |
| 2024-12-23 | Phase 2: Evaluation | Done | Precision, Recall, Hit Rate, NDCG, Coverage, Diversity |
| 2024-12-23 | Phase 3: CRM Integration | Done | 6 segment types with export |
| | Phase 4: Advanced Models | Pending | Matrix factorization, hybrid |

### Phase 1-3 Deliverables (Completed)

**Files Created:**
- `recommendations/__init__.py` - Module exports
- `recommendations/engine.py` - ItemItemRecommender, PopularityRecommender
- `recommendations/evaluator.py` - RecommendationEvaluator with all metrics
- `recommendations/segments.py` - RecommendationSegments for CRM exports

**CLI Task:**
```bash
# Train and evaluate
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task recommend

# Train only
python run.py --data data/dec_15.parquet --task recommend
```

**Dashboard:**
- Recommendations page with 4 tabs:
  1. Model Status - Training, metrics display
  2. User Lookup - Individual user recommendations
  3. Merchant Analysis - Similar merchants, prospects
  4. CRM Segments - Lapsed users, new users, cross-sell, superfans

**Test Results (Dec 15 data):**
- 54,413 users with orders
- 5,685 merchants
- 49,393 new users exported for onboarding
- Model saved to `outputs/recommendations/`

---

## References

- [Surprise Library](https://surpriselib.com/) - Python recommendation library
- [Implicit Library](https://github.com/benfred/implicit) - Fast collaborative filtering
- [Microsoft Recommenders](https://github.com/microsoft/recommenders) - Best practices
- [Evaluation Metrics Paper](https://arxiv.org/abs/2004.09817) - Beyond accuracy metrics
