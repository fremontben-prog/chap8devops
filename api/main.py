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
        # 1️⃣ Transformer en DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 2️⃣ Forcer exactement les colonnes vues à l'entraînement
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # 3️⃣ Prédiction
        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= BEST_THRESHOLD)

        return {
            "probability": float(proba),
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))