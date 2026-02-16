from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import numpy as np
import os
import logging


from api.schemas import PredictionInput, PredictionOutput
from api.model_loader import get_model_and_threshold

from elasticsearch import Elasticsearch
from datetime import datetime
import time

# 
es = Elasticsearch("http://elasticsearch:9200")


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
    
    start = time.time()
    try:
        # 1️⃣ JSON -> DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 2️⃣ Colonnes exactes
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # 3️⃣ Conversion numérique
        df = df.apply(pd.to_numeric, errors="coerce")

        # 4️⃣ Remplacer NaN
        df = df.fillna(0)

        # 5️⃣ Predict
        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= BEST_THRESHOLD)
        
        # Temps d'exécution
        exec_time = (time.time() - start) * 1000

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        # Log
        log_data = {
            "timestamp": datetime.utcnow(),
            "model_version": "v1.0",
            "input_features": data.model_dump_json(),
            "prediction": prediction,
            "probability": float(proba),
            "execution_time_ms": exec_time,
            "status_code": 200
        }
    
        logger = logging.getLogger(__name__)
        es.index(index="api-logs", document=log_data)
    except Exception as e:
        logger.warning(f"Elasticsearch unavailable: {e}")
        

        return {
            "probability": float(proba),
            "prediction": prediction
        }

    