# Snoonu ML Framework

## Project Overview
A reusable machine learning framework for consumer analytics at Snoonu (food delivery platform in Qatar). Built to be plug-and-play: drop in a new dataset, run analysis, get predictions.

## Business Context
- **Company**: Snoonu - food delivery app
- **Data Source**: Amplitude event data (exported as parquet)
- **Users**: ~130k unique users per day
- **Events**: ~3.6M events per day across 17 event types
- **Key Metrics**: Traffic, CVR (Conversion Rate), Orders, OPU (Orders Per User)

## Data Schema

### Core Columns
| Column | Type | Description |
|--------|------|-------------|
| `amplitude_id` | Int64 | Unique user identifier |
| `event_type` | string | Type of event (see below) |
| `event_time` | datetime | Timestamp of event |
| `event_properties` | JSON string | Event-specific properties |
| `platform` | string | iOS / Android |
| `device_family` | string | Device type |
| `os_name` | string | Operating system |
| `country` | string | User country (Qatar) |

### Event Types (Funnel Order)
1. `homepage_viewed` - Entry point
2. `category_page_viewed` - Browse categories
3. `merchant_page_viewed` - View restaurant
4. `product_page_viewed` - View item
5. `search_made` - Search action
6. `product_added` - Add to cart
7. `cart_created` - Cart initialized
8. `cart_page_viewed` - View cart
9. `checkout_button_pressed` - Start checkout
10. `payment_initiated` - Enter payment
11. `change_payment_method_viewed` - Payment friction
12. `checkout_completed` - ORDER PLACED
13. `order_delivered` - Order fulfilled
14. `order_info_page_viewed` - Track order
15. `user_signed_up` - New registration
16. `phone_entered` - Auth step
17. `otp_entered` - Auth verification

### Key Event Properties

#### checkout_completed
- `order_id` - Unique order ID
- `order_total` - Total order value
- `order_type` - delivery/now/scheduled/takeaway
- `merchant_id`, `merchant_name` - Restaurant
- `products_quantity` - Items in order
- `payment_method` - Payment type
- `delivery_fee` - Delivery cost
- `promo_code` - Discount code used
- `group_order` - Boolean, is group order
- `multi_cart` - Boolean, multiple merchants

#### homepage_viewed
- `first_session` - Boolean, new vs returning
- `customer_status` - logged_in, guest, etc.

#### order_delivered
- `total_delivered_orders` - User's lifetime order count
- `fulfillment_time` - Delivery duration

## Project Structure
```
snoonu-ml-framework/
├── CLAUDE.md           # This file - project context
├── config.yaml         # Configuration (schema, params)
├── run.py              # CLI entry point
├── data_loader.py      # Data loading & validation
├── feature_engine.py   # Feature generation
├── models/             # ML models
│   ├── churn.py
│   ├── segmentation.py
│   ├── conversion.py
│   └── recommendations.py
├── utils/              # Helper functions
├── outputs/            # Model outputs, reports
├── notebooks/          # Exploration notebooks
└── data/               # Data files (gitignored)
```

## Usage
```bash
# Basic Analysis
python run.py --data data/dec_15.parquet --task eda
python run.py --data data/dec_15.parquet --task segment
python run.py --data data/dec_15.parquet --task all

# ML Prediction Models
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn
python run.py --data data/dec_15.parquet --task predict_conversion
python run.py --data data/dec_15.parquet --task predict_ltv

# Dashboard
streamlit run app.py
```

## Key Metrics Formulas
- **CVR (user-based)**: Unique checkout users / Unique homepage users
- **CVR (order-based)**: Total orders / Unique homepage users
- **OPU**: Total orders / Unique checkout users

## ML Models

### 1. Churn Prediction
Predict which users are likely to stop ordering.
- **Features**: Recency, frequency, monetary, session patterns
- **Target**: No order in next 30 days
- **Output**: Churn probability per user

### 2. Customer Segmentation
Cluster users by behavior.
- **Method**: RFM + behavioral clustering
- **Output**: Segment labels, segment profiles

### 3. Conversion Prediction
Predict session conversion probability.
- **Features**: Funnel progress, time on site, platform
- **Target**: checkout_completed in session
- **Output**: Conversion probability

### 4. Recommendations (Future)
Personalized merchant/product recommendations.

## Development Notes
- Use pandas for data manipulation
- Use scikit-learn for ML models
- Keep models simple and interpretable first
- Config-driven: no hardcoded values
- All outputs should be reproducible

## Data Location
Default data path: `/Users/muhannadsaad/Desktop/investigation/dec_15/dec_15_25.parquet`
