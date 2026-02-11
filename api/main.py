from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import numpy as np

from api.schemas import PredictionInput, PredictionOutput
from api.model_loader import model

app = FastAPI(title="Credit Default API")

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
)

BEST_THRESHOLD = 0.5  # idéalement charger depuis MLflow

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    try:
        df = pd.DataFrame([data.features])

        proba = model.predict_proba(df)[:, 1][0]
        prediction = int(proba >= BEST_THRESHOLD)

        return {
            "probability": float(proba),
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
