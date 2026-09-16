# Integration and Load Testing

## Environment

- Application: Used Car Price Prediction API
- Framework: FastAPI
- Container: Docker
- Metrics: Prometheus
- Test environment: Local Docker Compose
- Base URL: http://127.0.0.1:8000

## Integration Testing

The application was started using:

docker compose up --build

The running container was tested over HTTP.

### Endpoints tested

- GET /health
- GET /metrics
- POST /api/v1/predict
- POST /api/v1/predict-batch

### Results

All integration tests passed against the running Docker container.

## Load Testing

A Python asyncio/httpx load test was used.

Number of concurrent requests:

50

Results:

- Successful requests: [FILL ACTUAL RESULT]
- Failed requests: [FILL ACTUAL RESULT]
- Total time: [FILL ACTUAL RESULT]
- Average response time: [FILL ACTUAL RESULT]
- Fastest response: [FILL ACTUAL RESULT]
- Slowest response: [FILL ACTUAL RESULT]

## Logs and Metrics

The application logs were monitored during the load test.

The /metrics endpoint was checked before and after the load test.

## Bug Found

[Describe the actual issue found.]

## Fix

[Describe the actual change made.]

## Verification

The integration tests and load test were executed again after the fix.

[Record the actual result.]

## Conclusion

The containerized API was tested end-to-end using real HTTP requests.
The integration and load testing results were recorded for Task 19.