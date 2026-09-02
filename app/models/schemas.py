from typing import List

from pydantic import BaseModel, Field


# Single Prediction Input


class PredictionInput(BaseModel):
    name: str

    year: int = Field(
        ...,
        gt=1900
    )

    km_driven: float = Field(
        ...,
        ge=0
    )

    fuel: str

    seller_type: str

    transmission: str

    owner: str


# Single Prediction Output

class PredictionOutput(BaseModel):
    request_id: str

    prediction: float

    confidence_score: float | None = None

    model_version: str

# Batch Prediction Input

class PredictionBatchInput(BaseModel):
    cars: List[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=100
    )

# Batch Prediction Output


class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]