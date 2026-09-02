import json
import time

import pandas as pd

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)

from app.logging_config import setup_logger

# Router

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

# Logger

logger = setup_logger()

# Model information file

MODEL_INFO_PATH = "ml/saved_model/model_info.json"

# Load model information

def load_model_info():

    with open(MODEL_INFO_PATH, "r") as file:

        return json.load(file)

# Health Endpoint

@router.get("/health")
def health(request: Request):

    model_loaded = hasattr(
        request.app.state,
        "model"
    )

    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

# Single Prediction Endpoint

@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    data: PredictionInput,
    request: Request
):

    request_id = request.state.request_id

    model = request.app.state.model

    try:

        # Convert input into DataFrame

        input_data = pd.DataFrame([
            data.model_dump()
        ])

        # Make prediction

        prediction = model.predict(
            input_data
        )

        # Load model information

        model_info = load_model_info()

        model_version = model_info[
            "model_version"
        ]

        logger.info(
            f"Prediction successful | "
            f"request_id={request_id}"
        )

        return PredictionOutput(

            request_id=request_id,

            prediction=float(
                prediction[0]
            ),

            confidence_score=None,

            model_version=model_version
        )

    except Exception as e:

        logger.exception(
            f"Prediction failed | "
            f"request_id={request_id} | "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

# Batch Prediction Endpoint


@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    batch: PredictionBatchInput,
    request: Request
):

    request_id = request.state.request_id

    model = request.app.state.model

    start_time = time.perf_counter()

    batch_size = len(
        batch.cars
    )

    try:

        # Convert all cars into dictionaries

        cars = [
            car.model_dump()
            for car in batch.cars
        ]

        # Convert complete batch to DataFrame

        input_df = pd.DataFrame(
            cars
        )

        # Predict entire batch at once

        predictions = model.predict(
            input_df
        )

        # Load model information

        model_info = load_model_info()

        model_version = model_info[
            "model_version"
        ]

        # Create output

        results = []

        for prediction in predictions:

            results.append(

                PredictionOutput(

                    request_id=request_id,

                    prediction=float(
                        prediction
                    ),

                    confidence_score=None,

                    model_version=model_version
                )
            )

        # Calculate duration

        duration = (
            time.perf_counter()
            - start_time
        )

        # Log batch information

        logger.info(
            f"Batch prediction successful | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"duration={duration:.4f}s"
        )

        return PredictionBatchOutput(
            predictions=results
        )

    except Exception as e:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.exception(
            f"Batch prediction failed | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"duration={duration:.4f}s | "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )


# =================================
# Model Information Endpoint
# =================================

@router.get("/model-info")
def model_info():

    try:

        info = load_model_info()

        return info

    except Exception as e:

        logger.exception(
            f"Failed to load model information | "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load model information"
        )