from prometheus_client import Counter


successful_predictions = Counter(
    "used_car_predictions_total",
    "Total number of successful used car price predictions"
)