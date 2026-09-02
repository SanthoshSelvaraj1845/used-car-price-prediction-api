import json

import pandas as pd

from fastapi import APIRouter, HTTPException, Request

from app.config import settings

from app.models.schemas import (
    PredictionInput,
    PredictionV2Output
)

from app.logging_config import setup_logger


# ---------------------------------
# V2 Router
# ---------------------------------

router = APIRouter(
    prefix="/api/v2",
    tags=["v2"]
)


logger = setup_logger()


# ---------------------------------
# Load Model Information
# ---------------------------------

def load_model_info():

    with open(
        settings.MODEL_INFO_PATH,
        "r"
    ) as file:

        return json.load(file)


# ---------------------------------
# V2 Prediction Endpoint
# ---------------------------------

@router.post(
    "/predict",
    response_model=PredictionV2Output
)
def predict_v2(
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
            f"V2 prediction successful | "
            f"request_id={request_id}"
        )

        return PredictionV2Output(
            request_id=request_id,
            predicted_price=float(
                prediction[0]
            ),
            model_version=model_version
        )

    except Exception as e:

        logger.exception(
            f"V2 prediction failed | "
            f"request_id={request_id} | "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="V2 prediction failed"
        )