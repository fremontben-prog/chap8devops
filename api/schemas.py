from pydantic import BaseModel, Field
from typing import Dict

class PredictionInput(BaseModel):
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    probability: float
    prediction: int
