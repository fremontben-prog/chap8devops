from pydantic import BaseModel, Field
from typing import Dict

class PredictionInput(BaseModel):
    AMT_CREDIT: float = Field(..., gt=0)
    DAYS_BIRTH: int = Field(...)
    model_config = {"extra": "allow"}

class PredictionOutput(BaseModel):
    probability: float
    prediction: int
