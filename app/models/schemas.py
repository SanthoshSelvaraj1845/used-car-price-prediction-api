from pydantic import BaseModel, Field


class PredictionInput(BaseModel):

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Car name"
    )

    year: int = Field(
        ...,
        ge=1900,
        le=2026,
        description="Manufacturing year"
    )

    km_driven: int = Field(
        ...,
        ge=0,
        description="Kilometers driven"
    )

    fuel: str = Field(
        ...,
        min_length=1,
        description="Fuel type"
    )

    seller_type: str = Field(
        ...,
        min_length=1,
        description="Seller type"
    )

    transmission: str = Field(
        ...,
        min_length=1,
        description="Transmission type"
    )

    owner: str = Field(
        ...,
        min_length=1,
        description="Owner type"
    )


class PredictionOutput(BaseModel):
    request_id: str
    prediction: float
    confidence_score: float | None = None
    model_version: str