from pydantic import BaseModel, Field
from typing import Dict

class PredictionInput(BaseModel):
    pass

class PredictionOutput(BaseModel):
    probability: float
    prediction: int
