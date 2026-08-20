# used-car-price-prediction-api
Machine learning REST API that predicts used car prices using Python, Scikit-learn/XGBoost, and FastAPI.

# Used Car Price Prediction API

## 1. Project Overview

This project is a machine learning-powered REST API that predicts the estimated selling price of a used car based on its features. The project combines a tabular machine learning model with FastAPI to provide predictions through a REST API. The main goal is to learn how to build and serve a machine learning model as a reliable API service.

## 2. ML Problem

**Problem:** Used Car Price Prediction

**Machine Learning Type:** Supervised Learning

**ML Task:** Regression

The model will learn the relationship between used-car features and their selling prices. Given the details of a used car, the model will predict its estimated selling price.

## 3. Dataset

The project will use a tabular used-car dataset containing information such as:

- name
- year
- km_driven
- fuel
- seller_type
- transmission
- owner

The target variable is the **selling price** of the used car.

## 4. Model

The project will use a regression model from **Scikit-learn and/or XGBoost**.

The model will be trained using the selected used-car dataset and evaluated using appropriate regression metrics such as MAE, RMSE, and R².

The initial focus is not on building a highly complex model. The main goal is to understand how a machine learning model can be integrated into a production-style API.

## 5. API Contract

The `/predict` endpoint will accept used-car information such as the manufacturing year, kilometers driven, fuel type, transmission type, and ownership information.

The API will validate the received input using Pydantic. Valid data will then be passed through the required preprocessing steps and sent to the trained machine learning model.

The model will return an estimated used-car selling price, which the API will return to the client as a JSON response.

### Example Request

```json
{
  "name": "Maruti Swift VXI",
  "year": 2020,
  "km_driven": 45000,
  "fuel": "Diesel",
  "seller_type": "Dealer",
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

## 6. REST API

The main MVP endpoint will be:

| Method | Endpoint   | Purpose                                   |
| ------ | ---------- | ----------------------------------------- |
| POST   | `/predict` | Predict the estimated price of a used car |

A successful prediction will return HTTP status code `200`.

Invalid request data will be rejected through Pydantic/FastAPI validation.

## 7. Request Flow

```text
Client
   |
   | POST /predict
   ↓
FastAPI
   |
   ↓
Pydantic Validation
   |
   ↓
Data Preprocessing
   |
   ↓
Trained ML Model
   |
   ↓
Predicted Car Price
   |
   ↓
JSON Response
```

## 8. Technology Stack

Python
FastAPI
Pydantic
Pandas
Scikit-learn
Joblib
Uvicorn
Docker
Pytest
Prometheus
Structured Logging
Git
GitHub

## 9. MVP Scope

The first version of the project will focus only on:

1. Preparing the used-car dataset
2. Training a regression model
3. Saving the trained model
4. Creating a FastAPI application
5. Creating the `/predict` endpoint
6. Validating request data using Pydantic
7. Returning the predicted car price

Advanced features such as authentication, databases, frontend applications, model retraining pipelines, and cloud deployment are outside the initial MVP scope.

## 10. Project Architecture

```text
                Used Car Dataset
                       |
                       ↓
               Data Preprocessing
                       |
                       ↓
                ML Model Training
                       |
                       ↓
                 Trained Model
                       |
                       ↓
Client → FastAPI → Pydantic → Prediction → JSON Response
                       |
             ┌─────────┴─────────┐
             ↓                   ↓
        Structured           Prometheus
         Logging              Metrics

Docker packages the application.
Pytest tests the application.
Git/GitHub manages the source code.
```

## 11. Future Tasks

The project will be developed incrementally:

* Project understanding and architecture planning
* Project folder structure and Python environment
* Dataset preparation and exploratory analysis
* Model training and evaluation
* FastAPI implementation
* Pydantic validation
* API testing with Pytest
* Dockerization
* Structured logging
* Prometheus monitoring
* GitHub documentation and version control

## 12. Goal

The goal of this project is to demonstrate how a machine learning model can be transformed into a usable, testable, monitorable, and containerized REST API service.

