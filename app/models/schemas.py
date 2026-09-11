from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    year: int = Field(..., gt=1900)
    km_driven: float = Field(..., ge=0)
    fuel: str = Field(..., min_length=1)
    seller_type: str = Field(..., min_length=1)
    transmission: str = Field(..., min_length=1)
    owner: str = Field(..., min_length=1)


class PredictionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    prediction: float
    confidence_score: float | None = None
    model_version: str


class PredictionBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cars: List[PredictionInput] = Field(..., min_length=1)


class PredictionBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: List[PredictionOutput]


class PredictionV2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    predicted_price: float
    model_version: str