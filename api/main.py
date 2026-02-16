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

# ElasticSarch avec
es = Elasticsearch("http://elasticsearch:9200")



# Chargement des variables de .env
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Credit Default API")

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
)

print("Elasticsearch ping:", es.ping())

@app.get("/")
def health():
    return {"status": "API running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    model, BEST_THRESHOLD = get_model_and_threshold()
    start = time.time()
    
    try:
        # 1. Préparation des données
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # 2. Prédiction
        proba = float(model.predict_proba(df)[0][1])
        prediction = int(proba >= BEST_THRESHOLD)
        exec_time = (time.time() - start) * 1000

        # 3. Tentative de Log (Elasticsearch) - Ne doit pas bloquer le retour
        try:
            log_data = {
                "timestamp": datetime.now(), # Vérifiez votre import ici !
                "model_version": "v1.0",
                "input_features": input_dict,
                "prediction": prediction,
                "probability": proba,
                "execution_time_ms": exec_time,
                "status_code": 200
            }
            es.index(index="api-logs", document=log_data)
        except Exception as log_err:
            print(f"Logging failed: {log_err}")

        # 4. Retour de la réponse
        return {
            "probability": proba,
            "prediction": prediction
        }

    except Exception as e:
        # Transforme l'erreur 500 en 400 avec un message explicite
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {str(e)}")