from contextlib import asynccontextmanager
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import PredictionInput, PredictionOutput


# Store loaded model
model = None


# Load model once when application starts
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


# Custom ValueError handler
@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid value",
            "message": str(exc)
        }
    )


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "ML API is alive"
    }


# Health endpoint
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict_car_price(car: PredictionInput):

    request_id = str(uuid.uuid4())

    try:

        # Convert Pydantic input to DataFrame
        input_df = pd.DataFrame([
            car.model_dump()
        ])

        # Make ML prediction
        prediction = model.predict(input_df)

        # Return validated response
        return PredictionOutput(
            request_id=request_id,
            prediction=float(prediction[0]),
            confidence_score=None,
            model_version="1.0.0"
        )

    except Exception as e:

        # Internal error information
        print(
            f"Prediction failed for request "
            f"{request_id}: {e}"
        )

        # Safe error for client
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )