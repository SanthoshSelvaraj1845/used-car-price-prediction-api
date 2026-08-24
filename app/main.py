from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI


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


# Create FastAPI app
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
def predict():

    # Car details
    car = pd.DataFrame([
        {
            "name": "Maruti Swift VXI",
            "year": 2020,
            "km_driven": 45000,
            "fuel": "Diesel",
            "seller_type": "Dealer",
            "transmission": "Manual",
            "owner": "First Owner"
        }
    ])

    # Make prediction
    prediction = model.predict(car)

    # Return prediction
    return {
        "predicted_price": float(prediction[0])
    }