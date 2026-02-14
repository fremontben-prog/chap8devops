from pydantic import BaseModel, Field
from typing import Dict

class PredictionInput(BaseModel):
    model_config = {"extra": "allow"}

class PredictionOutput(BaseModel):
    probability: float
    prediction: int
