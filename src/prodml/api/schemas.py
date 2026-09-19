from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    trip_distance: float = Field(gt=0, lt=200)
    PU_DO: str

    model_config = {
        "json_schema_extra": {"example": {"trip_distance": 5.0, "PU_DO": "132_264"}}
    }


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float
