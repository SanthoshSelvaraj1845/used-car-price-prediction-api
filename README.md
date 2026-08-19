# used-car-price-prediction-api
Machine learning REST API that predicts used car prices using Python, Scikit-learn/XGBoost, and FastAPI.

# Used Car Price Prediction API

## Project Overview

This project is a Machine Learning REST API that predicts the estimated price of a used car based on its details.

## ML Problem

* **Problem:** Used Car Price Prediction
* **Type:** Supervised Learning
* **Task:** Regression
* **Target:** Selling Price

## Dataset

The project uses a tabular used-car dataset containing features such as:

* Year
* Kilometers driven
* Fuel type
* Transmission
* Owner

The model uses these features to predict the car's selling price.

## API Contract

The `/predict` endpoint accepts used-car details and returns the estimated selling price.

### Example Request

```json
{
  "year": 2020,
  "km_driven": 45000,
  "fuel": "Diesel",
  "transmission": "Manual",
  "owner": "First Owner"
}
```

### Example Response

```json
{
  "predicted_price": 625000
}
```

## API Flow

```text
Client
   ↓
POST /predict
   ↓
FastAPI
   ↓
Pydantic Validation
   ↓
Data Preprocessing
   ↓
ML Model
   ↓
Predicted Price
   ↓
JSON Response
```

## Technology Stack

* Python
* FastAPI
* Pydantic
* Scikit-learn / XGBoost
* Uvicorn
* Docker
* Pytest
* Prometheus
* Structured Logging
* Git
* GitHub

## MVP Scope

The first version will focus on:

1. Preparing the used-car dataset
2. Training a regression model
3. Saving the trained model
4. Creating the FastAPI `/predict` endpoint
5. Validating input using Pydantic
6. Returning the predicted price

Advanced features such as authentication, frontend, payments, and user accounts are outside the initial MVP scope.

## Project Goal

The goal is to build a simple, reliable, and testable Machine Learning API that can predict used-car prices.
