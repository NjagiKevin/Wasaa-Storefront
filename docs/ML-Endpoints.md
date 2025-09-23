# Wasaa Storefront ML API Endpoints

This document lists ML-related API endpoints for Recommendations, Forecasting, Fraud Scoring, and Metrics. It includes example curl commands and JSON payloads you can paste into Postman.

## Base URL

Set a base URL according to how you run the API:

- FastAPI locally: http://localhost:8000
- Docker compose service (wasaa_storefront_api): http://localhost:8001

```bash
# choose one
BASE_URL=http://localhost:8000
# BASE_URL=http://localhost:8001
```

---

## 1) Recommendations

Get recommended products for a user. Uses Redis cache when available.

- Method: GET
- Path: /api/v1/recommendations/products/{user_id}

Example:
```bash
curl -sS "$BASE_URL/api/v1/recommendations/products/USER_123"
```
Sample response shape (your data may differ):
```json
{
  "user_id": "USER_123",
  "recommended_products": [
    {"product_id": "P001", "name": "Sneakers", "score": 0.92},
    {"product_id": "P002", "name": "Backpack", "score": 0.88}
  ],
  "count": 2
}
```

---

## 2) Demand Forecasting

Get demand scores for a list of product IDs (stub demand logic for now).

- Method: POST
- Path: /api/v1/forecasting/demand

Request body:
```json
{
  "product_ids": ["P001", "P002", "P003"]
}
```
Example:
```bash
curl -sS -X POST "$BASE_URL/api/v1/forecasting/demand" \
  -H "Content-Type: application/json" \
  -d '{
    "product_ids": ["P001", "P002", "P003"]
  }'
```
Sample response:
```json
{
  "P001": 0.73,
  "P002": 0.41,
  "P003": 0.50
}
```

---

## 3) Fraud & Risk Scoring

Score a transaction for fraud risk (stub model now; uses amount heuristic).

- Method: POST
- Path: /api/v1/fraud/score

Request body (feel free to add more fields; only "amount" is used right now):
```json
{
  "transaction_id": "T-1001",
  "user_id": "USER_123",
  "amount": 259.99,
  "currency": "KES",
  "channel": "web",
  "timestamp": "2025-09-23T04:45:00Z",
  "features": {
    "ip_country": "KE",
    "device_type": "mobile",
    "num_failed_payments_7d": 1
  }
}
```
Example:
```bash
curl -sS -X POST "$BASE_URL/api/v1/fraud/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "T-1001",
    "user_id": "USER_123",
    "amount": 259.99,
    "currency": "KES",
    "channel": "web",
    "timestamp": "2025-09-23T04:45:00Z",
    "features": { "ip_country": "KE", "device_type": "mobile", "num_failed_payments_7d": 1 }
  }'
```
Sample response:
```json
{ "score": 0.59 }
```

---

## 4) Metrics & Observability

Prometheus scrape endpoint with service metrics (includes recommendation request counters and latency histograms).

- Method: GET
- Path: /metrics

Example:
```bash
curl -sS "$BASE_URL/metrics"
```
Note: Output is Prometheus text format, not JSON.

---

## 5) Health

Basic service health check.

- Method: GET
- Path: /api/v1/health

Example:
```bash
curl -sS "$BASE_URL/api/v1/health"
```

---

### Notes
- The current endpoints do not require authentication in this setup.
- For Recommendations caching, set REDIS_URL for the API process if you want caching enabled.
- For DB-backed operations, ensure DATABASE_URL is set for the API process.
- Forecasting and Fraud scoring currently return stub/heuristic values until full model training/persistence is integrated.
