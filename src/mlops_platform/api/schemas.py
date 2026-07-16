from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=6, max_length=6)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    model_version: int


class DriftCheckRequest(BaseModel):
    current: list[list[float]] = Field(..., min_length=10)


class DriftCheckResponse(BaseModel):
    has_drift: bool
    drifted_features: list[str]
    alerted: bool


class ModelInfo(BaseModel):
    name: str
    version: int
    stage: str
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    production_model: str
