# Data Schema Reference

Complete specification of the data format expected by the Snoonu ML Framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Required Columns](#required-columns)
3. [Optional Columns](#optional-columns)
4. [Event Types](#event-types)
5. [Event Properties](#event-properties)
6. [Data Examples](#data-examples)
7. [Validation Rules](#validation-rules)
8. [Data Quality](#data-quality)

---

## Overview

### Supported Formats
- **Parquet** (recommended) - Fast loading, compressed
- **CSV** - Universal compatibility

### Data Source
The framework is designed for **Amplitude** event data exports, but works with any event-based data following this schema.

### Typical Volume
- ~3.6M events per day
- ~130k unique users per day
- 17 event types

---

## Required Columns

These columns **must** exist for the framework to function.

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `amplitude_id` | `int64` | Unique user identifier | `1234567890` |
| `event_type` | `string` | Name of the event | `"checkout_completed"` |
| `event_time` | `datetime64` | Timestamp of event (UTC or local) | `2024-12-15 14:30:00` |
| `platform` | `string` | Device platform | `"ios"` or `"android"` |

### Column Details

#### `amplitude_id`
- Unique identifier for each user
- Should be consistent across sessions
- Type: 64-bit integer
- Cannot be null

#### `event_type`
- Name of the tracked event
- Case-sensitive
- Must match one of the [defined event types](#event-types)
- Cannot be null

#### `event_time`
- Timestamp when event occurred
- Should be timezone-naive (recommended) or UTC
- Format: `YYYY-MM-DD HH:MM:SS`
- Cannot be null

#### `platform`
- User's device platform
- Values: `"ios"`, `"android"` (lowercase)
- Used for platform-specific analytics
- Cannot be null

---

## Optional Columns

These columns enhance analysis but are not required.

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `device_family` | `string` | Device type/model | `"iPhone"`, `"Samsung"` |
| `os_name` | `string` | Operating system | `"iOS"`, `"Android"` |
| `os_version` | `string` | OS version | `"17.0"`, `"14"` |
| `country` | `string` | User's country | `"Qatar"` |
| `device_id` | `string` | Unique device ID | `"abc123..."` |
| `event_properties` | `string` (JSON) | Event-specific data | `{"order_total": 45.50}` |

### `event_properties` Column

This JSON column contains event-specific attributes. See [Event Properties](#event-properties) for details.

---

## Event Types

### Funnel Events (Conversion Path)

Events are listed in typical progression order:

| Order | Event Type | Description | Funnel Stage |
|-------|------------|-------------|--------------|
| 1 | `homepage_viewed` | App home screen loaded | Entry |
| 2 | `category_page_viewed` | Browse category page | Browse |
| 3 | `merchant_page_viewed` | Restaurant page viewed | Consideration |
| 4 | `product_page_viewed` | Menu item viewed | Consideration |
| 5 | `search_made` | Search performed | Discovery |
| 6 | `product_added` | Item added to cart | Intent |
| 7 | `cart_created` | Cart initialized | Intent |
| 8 | `cart_page_viewed` | Cart page opened | Intent |
| 9 | `checkout_button_pressed` | Checkout initiated | Checkout |
| 10 | `payment_initiated` | Payment flow started | Payment |
| 11 | `change_payment_method_viewed` | Payment method changed | Payment Friction |
| 12 | `checkout_completed` | **Order placed** | Conversion |
| 13 | `order_delivered` | Order fulfilled | Post-purchase |

### Authentication Events

| Event Type | Description |
|------------|-------------|
| `user_signed_up` | New user registration |
| `phone_entered` | Phone number submitted |
| `otp_entered` | OTP verification completed |

### Post-Purchase Events

| Event Type | Description |
|------------|-------------|
| `order_info_page_viewed` | Order tracking page viewed |
| `order_delivered` | Order successfully delivered |

---

## Event Properties

Each event type has specific properties stored as JSON in the `event_properties` column.

### `checkout_completed` Properties

The most important event with rich data:

```json
{
  "order_id": "ORD-12345",
  "order_total": 45.50,
  "order_type": "delivery",
  "merchant_id": "m123",
  "merchant_name": "Pizza Palace",
  "products_quantity": 3,
  "payment_method": "credit_card",
  "delivery_fee": 3.50,
  "promo_code": "SAVE10",
  "group_order": false,
  "multi_cart": false,
  "distance": 5.2,
  "estimated_delivery_time": 35,
  "products": [
    {
      "product_id": "p1",
      "product_name": "Margherita Pizza",
      "quantity": 1,
      "price": 25.00
    }
  ]
}
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `order_id` | string | Yes | Unique order identifier |
| `order_total` | float | Yes | Total order amount |
| `order_type` | string | Yes | `delivery`, `takeaway`, `scheduled` |
| `merchant_id` | string | Yes | Restaurant ID |
| `merchant_name` | string | Yes | Restaurant name |
| `products_quantity` | int | Yes | Number of items |
| `payment_method` | string | No | Payment type used |
| `delivery_fee` | float | No | Delivery charge |
| `promo_code` | string | No | Applied promo code |
| `group_order` | bool | No | Is group order |
| `multi_cart` | bool | No | Multiple merchants |
| `distance` | float | No | Delivery distance (km) |
| `estimated_delivery_time` | int | No | ETA in minutes |
| `products` | array | No | Product details |

### `order_delivered` Properties

```json
{
  "order_id": "ORD-12345",
  "total_delivered_orders": 5,
  "fulfillment_time": 28,
  "order_status": "delivered"
}
```

| Property | Type | Description |
|----------|------|-------------|
| `order_id` | string | Order identifier |
| `total_delivered_orders` | int | User's lifetime delivered orders |
| `fulfillment_time` | int | Actual delivery time (minutes) |
| `order_status` | string | Final status |

### `homepage_viewed` Properties

```json
{
  "first_session": true,
  "customer_status": "logged_in",
  "hour": 12,
  "zone_name": "Doha Downtown"
}
```

| Property | Type | Description |
|----------|------|-------------|
| `first_session` | bool | Is this user's first session |
| `customer_status` | string | `logged_in`, `guest`, etc. |
| `hour` | int | Hour of day (0-23) |
| `zone_name` | string | Delivery zone |

### `product_page_viewed` Properties

```json
{
  "product_id": "p456",
  "product_name": "Margherita Pizza",
  "product_price": 12.99,
  "merchant_id": "m123"
}
```

| Property | Type | Description |
|----------|------|-------------|
| `product_id` | string | Product identifier |
| `product_name` | string | Product name |
| `product_price` | float | Product price |
| `merchant_id` | string | Merchant identifier |

### `merchant_page_viewed` Properties

```json
{
  "merchant_id": "m123",
  "merchant_name": "Pizza Palace",
  "category": "Pizza"
}
```

### `search_made` Properties

```json
{
  "query": "pizza",
  "results_count": 15
}
```

### `product_added` Properties

```json
{
  "product_id": "p456",
  "product_name": "Margherita Pizza",
  "product_price": 12.99,
  "quantity": 1,
  "merchant_id": "m123"
}
```

---

## Data Examples

### Minimal Valid Dataset

```csv
amplitude_id,event_type,event_time,platform
12345,homepage_viewed,2024-12-15 10:00:00,ios
12345,merchant_page_viewed,2024-12-15 10:02:00,ios
12345,product_added,2024-12-15 10:05:00,ios
12345,checkout_completed,2024-12-15 10:10:00,ios
```

### Full Dataset with Properties

```csv
amplitude_id,event_type,event_time,platform,device_family,event_properties
12345,homepage_viewed,2024-12-15 10:00:00,ios,iPhone,"{""first_session"": false, ""customer_status"": ""logged_in""}"
12345,search_made,2024-12-15 10:01:00,ios,iPhone,"{""query"": ""pizza""}"
12345,merchant_page_viewed,2024-12-15 10:02:00,ios,iPhone,"{""merchant_id"": ""m123"", ""merchant_name"": ""Pizza Palace""}"
12345,product_page_viewed,2024-12-15 10:03:00,ios,iPhone,"{""product_id"": ""p1"", ""product_name"": ""Margherita"", ""product_price"": 25.00}"
12345,product_added,2024-12-15 10:05:00,ios,iPhone,"{""product_id"": ""p1"", ""quantity"": 1}"
12345,cart_page_viewed,2024-12-15 10:06:00,ios,iPhone,"{}"
12345,checkout_button_pressed,2024-12-15 10:07:00,ios,iPhone,"{}"
12345,payment_initiated,2024-12-15 10:08:00,ios,iPhone,"{""payment_method"": ""credit_card""}"
12345,checkout_completed,2024-12-15 10:10:00,ios,iPhone,"{""order_id"": ""ORD-001"", ""order_total"": 35.50, ""merchant_name"": ""Pizza Palace"", ""order_type"": ""delivery""}"
```

### Parquet Schema (PyArrow)

```python
import pyarrow as pa

schema = pa.schema([
    ('amplitude_id', pa.int64()),
    ('event_type', pa.string()),
    ('event_time', pa.timestamp('us')),
    ('platform', pa.string()),
    ('device_family', pa.string()),
    ('os_name', pa.string()),
    ('os_version', pa.string()),
    ('country', pa.string()),
    ('device_id', pa.string()),
    ('event_properties', pa.string()),  # JSON string
])
```

---

## Validation Rules

The framework applies these validation rules when loading data:

### Required Column Validation

```python
REQUIRED_COLUMNS = ['amplitude_id', 'event_type', 'event_time', 'platform']

# Check presence
missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
```

### Data Type Validation

| Column | Expected Type | Conversion |
|--------|---------------|------------|
| `amplitude_id` | int64 | Auto-convert from string |
| `event_time` | datetime64 | Parse from string |
| `platform` | string | Lowercase normalization |

### Platform Normalization

```python
# Normalize platform values
df['platform'] = df['platform'].str.lower()
# Valid values: 'ios', 'android'
```

### Event Time Parsing

```python
# Auto-parse various formats
df['event_time'] = pd.to_datetime(df['event_time'])

# Supported formats:
# - 2024-12-15 14:30:00
# - 2024-12-15T14:30:00
# - 2024-12-15T14:30:00Z
# - 2024-12-15T14:30:00+00:00
```

### Sorting

Data should be sorted by user and time:

```python
df = df.sort_values(['amplitude_id', 'event_time'])
```

---

## Data Quality

### Recommended Quality Checks

#### 1. Completeness
```python
# Check for nulls in required columns
null_counts = df[REQUIRED_COLUMNS].isnull().sum()
if null_counts.any():
    print("Warning: Null values found")
```

#### 2. Event Distribution
```python
# Ensure all expected events exist
expected_events = [
    'homepage_viewed', 'checkout_completed', 'order_delivered'
]
missing_events = set(expected_events) - set(df['event_type'].unique())
```

#### 3. Time Range
```python
# Check date range makes sense
date_range = df['event_time'].max() - df['event_time'].min()
print(f"Data spans {date_range.days} days")
```

#### 4. User Counts
```python
# Verify user count is reasonable
unique_users = df['amplitude_id'].nunique()
print(f"Unique users: {unique_users:,}")
```

#### 5. Conversion Sanity
```python
# Check conversion rate is reasonable (0.5% - 20%)
homepage = df[df['event_type'] == 'homepage_viewed']['amplitude_id'].nunique()
checkout = df[df['event_type'] == 'checkout_completed']['amplitude_id'].nunique()
cvr = checkout / homepage * 100
assert 0.5 <= cvr <= 20, f"Unusual CVR: {cvr:.2f}%"
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing events | Low CVR, broken funnel | Verify event tracking |
| Duplicate events | Inflated counts | Deduplicate by event_id |
| Wrong timezone | Off-by-hours patterns | Normalize to UTC |
| Invalid JSON | Parse errors | Validate JSON format |
| Missing order_total | Zero revenue | Check checkout properties |

### Sample Data Check Script

```python
from data_loader import DataLoader

loader = DataLoader()
df = loader.load('data/your_file.parquet')

# Print summary
loader.print_summary(df)

# Check event distribution
print("\nEvent Distribution:")
print(df['event_type'].value_counts())

# Check platform distribution
print("\nPlatform Distribution:")
print(df['platform'].value_counts())

# Check for nulls
print("\nNull Counts:")
print(df.isnull().sum())

# Sample checkout properties
checkout = df[df['event_type'] == 'checkout_completed'].head(1)
if len(checkout) > 0:
    print("\nSample checkout_completed properties:")
    print(checkout['event_properties'].iloc[0])
```

---

## Quick Reference

### Minimum Required Data

```
amplitude_id | event_type         | event_time          | platform
-------------|--------------------|--------------------|----------
int64        | string             | datetime           | string
```

### Key Events for Analytics

| Analysis Type | Required Events |
|---------------|-----------------|
| Conversion | homepage_viewed, checkout_completed |
| Funnel | All funnel events |
| Revenue | checkout_completed with order_total |
| Delivery | order_delivered with fulfillment_time |
| Search | search_made |
| Recommendations | checkout_completed with merchant_id |

### Key Properties for ML

| Model | Required Properties |
|-------|---------------------|
| Churn | order dates, recency |
| Conversion | funnel events |
| LTV | order_total, order counts |
| Recommendations | merchant_id, order history |
