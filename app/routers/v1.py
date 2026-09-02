import pandas as pd

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import setup_logger


router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

logger = setup_logger()


@router.get("/health")
def health(request: Request):

    model = request.app.state.model

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict_car_price(
    car: PredictionInput,
    request: Request
):

    request_id = request.state.request_id

    model = request.app.state.model

    try:

        input_df = pd.DataFrame([
            car.model_dump()
        ])

        prediction = model.predict(input_df)

        predicted_price = float(prediction[0])

        logger.info(
            f"Prediction successful | "
            f"request_id={request_id} | "
            f"prediction={predicted_price}"
        )

        return PredictionOutput(
            request_id=request_id,
            prediction=predicted_price,
            confidence_score=None,
            model_version="1.0.0"
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


# Future v2 plan:
# /api/v2/predict can use a separate Pydantic response schema,
# for example PredictionOutputV2, with extra fields.
# v1 should remain unchanged so existing clients do not break.