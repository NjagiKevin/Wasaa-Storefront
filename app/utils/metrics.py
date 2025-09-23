from prometheus_client import Counter, Histogram

# Requests
recommendation_requests_total = Counter(
    'recommendation_requests_total',
    'Total recommendation requests',
    ['endpoint']
)

# Latency
recommendation_request_latency_seconds = Histogram(
    'recommendation_request_latency_seconds',
    'Recommendation request latency in seconds',
    ['endpoint']
)
