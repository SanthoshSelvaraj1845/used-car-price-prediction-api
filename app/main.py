from contextlib import asynccontextmanager

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
    lifespan=lifespan
)


# Home endpoint
@app.get("/")
def root():
    return {
        "message": "ML API is alive"
    }


# Prediction endpoint
@app.post("/predict")
def predict(car: PredictionInput):

    # Convert validated Pydantic data into DataFrame
    input_df = pd.DataFrame([
        car.model_dump()
    ])

    # Make prediction
    prediction = model.predict(input_df)

    # Return prediction
    return {
        "predicted_price": float(prediction[0])
    }