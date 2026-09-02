import json
import time

import pandas as pd

from fastapi import APIRouter, HTTPException, Request

from app.config import settings

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)

from app.logging_config import setup_logger


# ---------------------------------
# Router
# ---------------------------------

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)


# ---------------------------------
# Logger
# ---------------------------------

logger = setup_logger()


# =================================
# Load Model Information
# =================================

def load_model_info():

    with open(
        settings.MODEL_INFO_PATH,
        "r"
    ) as file:

        return json.load(file)


# =================================
# Health Endpoint
# =================================

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


# =================================
# Single Prediction
# =================================

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

        input_data = pd.DataFrame([
            data.model_dump()
        ])

        prediction = model.predict(
            input_data
        )

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


# =================================
# Batch Prediction
# =================================

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

    batch_size = len(
        batch.cars
    )

    start_time = time.perf_counter()

    # ---------------------------------
    # Check maximum batch size
    # ---------------------------------

    if batch_size > settings.MAX_BATCH_SIZE:

        logger.warning(
            f"Batch size exceeded | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"max_batch_size={settings.MAX_BATCH_SIZE}"
        )

        raise HTTPException(

            status_code=400,

            detail=(
                f"Batch size cannot exceed "
                f"{settings.MAX_BATCH_SIZE} cars. "
                f"Received {batch_size} cars."
            )
        )

    try:

        # ---------------------------------
        # Convert cars to dictionaries
        # ---------------------------------

        cars = [
            car.model_dump()
            for car in batch.cars
        ]

        # ---------------------------------
        # Create DataFrame
        # ---------------------------------

        input_df = pd.DataFrame(
            cars
        )

        # ---------------------------------
        # Predict complete batch at once
        # ---------------------------------

        predictions = model.predict(
            input_df
        )

        # ---------------------------------
        # Get model version
        # ---------------------------------

        model_info = load_model_info()

        model_version = model_info[
            "model_version"
        ]

        # ---------------------------------
        # Create results
        # ---------------------------------

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

        # ---------------------------------
        # Calculate duration
        # ---------------------------------

        duration = (
            time.perf_counter()
            - start_time
        )

        # ---------------------------------
        # Log batch prediction
        # ---------------------------------

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
# Model Information
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