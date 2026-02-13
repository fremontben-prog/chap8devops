from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import numpy as np
import os

from api.schemas import PredictionInput, PredictionOutput
from api.model_loader import get_model_and_threshold


# Chargement des variables de .env
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Credit Default API")

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
)


@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    model, BEST_THRESHOLD = get_model_and_threshold()
    try:
        df = pd.DataFrame([data.model_dump()])
        

        proba = model.predict_proba(df)[:, 1][0]
        prediction = int(proba >= BEST_THRESHOLD)

        return {
            "probability": float(proba),
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
