from contextlib import asynccontextmanager
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI

from app.models.schemas import PredictionInput


# Store the loaded model
model = None


# Load model when FastAPI starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load(
        "ml/saved_model/model.joblib"
    )

    print("ML model loaded successfully")

    yield


# Create FastAPI application
app = FastAPI(
    title="Used Car Price Prediction API",
    description="API for predicting used car prices using Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "ML API is alive"
    }


# Health check endpoint
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict_car_price(car: PredictionInput):

    # Create a request ID
    request_id = str(uuid.uuid4())

    # Convert validated Pydantic data to DataFrame
    input_df = pd.DataFrame([
        car.model_dump()
    ])

    # Make prediction
    prediction = model.predict(input_df)

    # RandomForestRegressor does not support predict_proba
    confidence_score = None

    return {
        "request_id": request_id,
        "prediction": float(prediction[0]),
        "confidence_score": confidence_score
    }