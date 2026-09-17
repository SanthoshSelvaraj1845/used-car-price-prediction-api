# Used Car Price Prediction API

A production-oriented Machine Learning API that predicts the selling price of a used car based on its vehicle and ownership details.

The project combines a trained Machine Learning model with a FastAPI service, input validation, API versioning, authentication, structured logging, Docker containerization, Prometheus monitoring, automated testing, and integration/load testing.

---

## Project Overview

The goal of this project is to build a complete ML-powered API rather than only training a Machine Learning model.

The system takes used-car information through an HTTP API and returns a predicted selling price.

## Architecture

                         ┌──────────────────────┐
                         │       Client         │
                         │ curl / Browser / App │
                         └──────────┬───────────┘
                                    │
                                    │ HTTP Request
                                    ▼
                         ┌──────────────────────┐
                         │    Docker Container  │
                         │                      │
                         │      FastAPI         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   API Key Security   │
                         │      x-api-key       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Pydantic Validation  │
                         │   Request Schema     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────┐
                 │             API Router              │
                 │                                    │
                 │ /api/v1/predict                    │
                 │ /api/v1/predict-batch              │
                 │ /api/v1/health                     │
                 │ /api/v1/model-info                 │
                 │ /api/v2/predict                    │
                 └────────────────┬───────────────────┘
                                  │
                                  ▼
                         ┌──────────────────────┐
                         │   Saved ML Model     │
                         │    model.joblib      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Price Prediction   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    JSON Response     │
                         └──────────────────────┘


       ┌──────────────────────┐          ┌──────────────────────┐
       │  Structured Logging  │          │ Prometheus Metrics   │
       │     logs/app.log     │          │      /metrics        │
       └──────────────────────┘          └──────────────────────┘

## ML training

                  ┌──────────────────────┐
                  │  Used Car Dataset    │
                  │  data/used_cars.csv  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Preprocessing   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Train/Test Split     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Random Forest Model  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Model Evaluation     │
                  │ MAE / MSE / R²       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ model.joblib         │
                  │ Saved Model          │
                  └──────────┬───────────┘
                             │
                             ▼
                       FastAPI API
                       

### Input Features

The model uses the following features:

* Car name
* Manufacturing year
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Owner type

### Output

The API returns:

* Prediction
* Request ID
* Model version
* Additional API response information depending on the API version

---

## Technology Stack

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| Python                  | Main programming language |
| FastAPI                 | REST API framework        |
| Pydantic                | Request validation        |
| Scikit-learn            | Machine Learning          |
| Random Forest Regressor | Price prediction model    |
| Pandas                  | Data processing           |
| Joblib                  | Model serialization       |
| Uvicorn                 | ASGI server               |
| Docker                  | Containerization          |
| Docker Compose          | Local deployment          |
| Pytest                  | Automated testing         |
| HTTPX                   | Integration testing       |
| Prometheus              | API monitoring            |
| Git/GitHub              | Version control           |

---

## Architecture

```text
                        Client
                          |
                          | HTTP Request
                          v
                  +------------------+
                  |    FastAPI API   |
                  +------------------+
                          |
                          v
                  +------------------+
                  | API Key Security |
                  +------------------+
                          |
                          v
                  +------------------+
                  | Pydantic         |
                  | Validation       |
                  +------------------+
                          |
                          v
                  +------------------+
                  | API Router       |
                  | /api/v1          |
                  | /api/v2          |
                  +------------------+
                          |
                          v
                  +------------------+
                  | Saved ML Pipeline|
                  | / Model          |
                  +------------------+
                          |
                          v
                  +------------------+
                  | Price Prediction |
                  +------------------+
                          |
             +------------+-------------+
             |                          |
             v                          v
      Structured Logs              Prometheus
      logs/app.log                  /metrics
```

---

## Project Structure

```text
used-car-price-prediction-api/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logging_config.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── routers/
│       └── v1.py
│
├── ml/
│   ├── train.py
│   ├── predict.py
│   └── saved_model/
│       ├── model.joblib
│       └── model_info.json
│
├── tests/
│   ├── test_integration.py
│   └── ...
│
├── data/
│   └── used_cars.csv
│
├── logs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env
├── .gitignore
└── README.md
```

> `.env` contains local configuration and secrets and should not be committed to GitHub.

---

# Machine Learning Pipeline

The Machine Learning workflow is:

```text
Used Car Dataset
       |
       v
Data Preparation
       |
       v
Feature Selection
       |
       v
Train/Test Split
       |
       v
Preprocessing
       |
       v
Random Forest Regressor
       |
       v
Model Evaluation
       |
       v
Saved Model
       |
       v
FastAPI Prediction Service
```

The trained model is saved using Joblib so that the API can load the model without retraining it for every request.

---

# API Endpoints

## Root

### `GET /`

Checks that the API application is running.

Example:

```bash
curl http://localhost:8000/
```

---

## Health Check

### `GET /api/v1/health`

Checks API and model availability.

Example:

```bash
curl http://localhost:8000/api/v1/health \
  -H "x-api-key: YOUR_API_KEY"
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Single Prediction

### `POST /api/v1/predict`

Predicts the selling price of one used car.

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "name": "Maruti Swift VXI",
    "year": 2020,
    "km_driven": 45000,
    "fuel": "Diesel",
    "seller_type": "Dealer",
    "transmission": "Manual",
    "owner": "First Owner"
  }'
```

Example response:

```json
{
  "request_id": "example-request-id",
  "prediction": 770000.0,
  "confidence_score": null,
  "model_version": "1.0.0"
}
```

The exact prediction depends on the trained model and input data.

---

# Batch Prediction

### `POST /api/v1/predict-batch`

Predicts prices for multiple cars in one request.

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/predict-batch" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "cars": [
      {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
      },
      {
        "name": "Hyundai i20",
        "year": 2019,
        "km_driven": 30000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
      }
    ]
  }'
```

---

# API Version 2

### `POST /api/v2/predict`

Provides the version 2 prediction response.

Example:

```bash
curl -X POST "http://localhost:8000/api/v2/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "name": "Maruti Swift VXI",
    "year": 2020,
    "km_driven": 45000,
    "fuel": "Diesel",
    "seller_type": "Dealer",
    "transmission": "Manual",
    "owner": "First Owner"
  }'
```

API versioning allows the service to introduce new response formats or functionality while maintaining the existing version.

---

# Model Information

### `GET /api/v1/model-info`

Returns information about the loaded Machine Learning model.

Example:

```bash
curl http://localhost:8000/api/v1/model-info \
  -H "x-api-key: YOUR_API_KEY"
```

---

# Prometheus Metrics

### `GET /metrics`

Exposes application metrics for Prometheus.

Example:

```bash
curl http://localhost:8000/metrics
```

The endpoint exposes metrics related to application requests and API activity.

---

# Interactive API Documentation

FastAPI automatically provides interactive documentation.

After starting the application, open:

```text
http://localhost:8000/docs
```

Alternative OpenAPI documentation:

```text
http://localhost:8000/redoc
```

---

# Authentication

Protected API endpoints require an API key through the request header:

```text
x-api-key
```

Example:

```text
x-api-key: YOUR_API_KEY
```

The API key is configured through an environment variable.

Never commit real API keys or other secrets to GitHub.

---

# Configuration

Configuration is managed through environment variables.

Example local `.env`:

```env
MODEL_PATH=ml/saved_model/model.joblib
LOG_LEVEL=INFO
MAX_BATCH_SIZE=100
API_TITLE=Used Car Price Prediction API
API_VERSION=1.0.0
MODEL_INFO_PATH=ml/saved_model/model_info.json
LOG_FILE_PATH=logs/app.log
API_KEY=YOUR_LOCAL_API_KEY
```

The actual `.env` file should remain local and should be excluded through `.gitignore`.

---

# Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/SanthoshSelvaraj1845/used-car-price-prediction-api.git
cd used-car-price-prediction-api
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a local `.env` file and configure the required values.

## 5. Start the API

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# Running with Docker Compose

The recommended reproducible way to run the application is Docker Compose.

```bash
docker compose up --build
```

After the container starts:

```text
API:   http://localhost:8000
Docs:  http://localhost:8000/docs
```

To stop the application:

```bash
docker compose down
```

---

# Testing

The project uses Pytest for automated testing.

Run the complete test suite:

```bash
pytest -v
```

Integration tests communicate with the running API over HTTP instead of directly calling FastAPI application functions.

Run integration tests:

```bash
pytest tests/test_integration.py -v
```

Before running integration tests, make sure the Docker container or local API server is running.

---

# Load Testing

A basic asynchronous HTTP load test was created to send concurrent prediction requests to the API.

The load test was used to evaluate:

* Request success rate
* Response time
* Fastest request
* Slowest request
* Average response time
* HTTP status codes

The load test also helped identify configuration/application issues before finalizing the project.

---

# Logging

The API uses structured application logging.

Logs include information such as:

* Timestamp
* Log level
* Logger name
* Request processing information
* Request ID
* API activity

Application logs are written to:

```text
logs/app.log
```

Log rotation is configured to prevent a single log file from growing indefinitely.

---

# Monitoring

Prometheus-compatible metrics are exposed through:

```text
GET /metrics
```

This allows monitoring tools such as Prometheus and Grafana to collect and visualize application metrics.

---

# Error Handling and Validation

Pydantic models validate incoming API requests.

Examples of invalid input that can be rejected include:

* Missing required fields
* Invalid numeric values
* Negative `km_driven`
* Invalid year values
* Unexpected request fields
* Invalid API keys
* Oversized batch requests

This prevents invalid data from reaching the Machine Learning model.

---

# Docker

The application is containerized using Docker.

The container packages:

* Python runtime
* Application code
* Machine Learning dependencies
* Saved model
* API configuration

This makes the application reproducible across different environments.

---

# Deployment

The application is designed to be deployed as a Docker container to a cloud hosting platform.

The deployment configuration uses environment variables for configuration and secrets rather than storing sensitive values in source code.

A public deployment URL should be added here after deployment:

```text
Live API:
YOUR_DEPLOYED_URL
```

Swagger documentation:

```text
YOUR_DEPLOYED_URL/docs
```

Metrics:

```text
YOUR_DEPLOYED_URL/metrics
```

---

# Independent Extension

## GitHub Actions CI

As an independent extension, GitHub Actions can be used to automatically run the project's test suite whenever code is pushed to GitHub.

The workflow performs the following:

```text
Git Push
   |
   v
GitHub Actions
   |
   v
Set up Python
   |
   v
Install Dependencies
   |
   v
Run Pytest
   |
   v
Pass / Fail
```

This adds an automated quality check to the project and helps detect regressions before changes are merged or deployed.

---

# What I Learned

During this project I learned how to build an ML application as a complete API service instead of stopping after model training.

I learned how to:

* Train and save a Machine Learning model.
* Load a saved model without retraining it.
* Build REST APIs using FastAPI.
* Validate API input using Pydantic.
* Organize APIs using routers and API versions.
* Handle configuration using environment variables.
* Add API-key based security.
* Implement structured application logging.
* Add Prometheus metrics.
* Write automated tests using Pytest.
* Perform integration testing against a running API.
* Perform basic concurrent load testing.
* Package the application using Docker.
* Run the application using Docker Compose.
* Debug API errors such as authentication, validation, routing, and server errors.
* Document an application so another developer can understand and run it.
* Use Git and GitHub to manage the project.

The biggest lesson was understanding the complete flow:

```text
Client Request
      ↓
Docker Container
      ↓
FastAPI
      ↓
Authentication
      ↓
Pydantic Validation
      ↓
API Router
      ↓
Saved ML Model
      ↓
Prediction
      ↓
Response
      ↓
Logs + Prometheus Metrics
```

---

# Future Improvements

Possible future improvements include:

* Continuous deployment
* Model retraining pipeline
* Database storage for prediction history
* Grafana monitoring dashboard
* More advanced model evaluation
* Response caching
* Model version management
* Improved authentication and authorization
* Cloud-based model storage

---

# Final Project Status

The project demonstrates an end-to-end Machine Learning API workflow:

```text
Dataset
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Serialization
  ↓
FastAPI Service
  ↓
Validation
  ↓
Security
  ↓
Logging
  ↓
Monitoring
  ↓
Automated Testing
  ↓
Integration Testing
  ↓
Load Testing
  ↓
Docker
  ↓
Cloud Deployment
```

The final goal is a reproducible, tested, documented, and deployable Machine Learning API.

---