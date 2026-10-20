"""
Custom Prometheus metrics and monitoring telemetry for Heart Disease API.
"""
from prometheus_client import Counter, Histogram

PREDICTION_REQUESTS_TOTAL = Counter(
    "heart_prediction_requests_total",
    "Total count of patient prediction requests processed",
    ["result_class", "risk_level"]
)

PREDICTION_CONFIDENCE_HISTOGRAM = Histogram(
    "heart_prediction_confidence_histogram",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "heart_prediction_latency_seconds",
    "Time taken to execute model inference in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
)
