from fastapi import FastAPI, HTTPException
from api.schemas import PredictionInput, PredictionOutput
from api.model_loader import get_model_and_threshold
from api.middleware import setup_middleware

import os
from elasticsearch import Elasticsearch
from contextlib import asynccontextmanager
import time
import cProfile
import numpy as np

from pathlib import Path

import onnxruntime as ort


from monitoring.drift_monitor import run_global_monitoring

ES_HOST = os.getenv("ES_HOST", "localhost")
INDEX_PROD = os.getenv("INDEX_PROD", "predictions")

es = Elasticsearch(f"http://{ES_HOST}:9200")


# ========================
# LIFESPAN
#=========================
@asynccontextmanager
async def lifespan(app: FastAPI):

    # ---- WAIT FOR ELASTICSEARCH ----
    max_retries = 10
    for attempt in range(max_retries):
        try:
            if es.ping():
                print("OK - Elasticsearch connecté")
                break
        except Exception:
            pass

        print(f"Attente Elasticsearch... ({attempt+1}/{max_retries})")
        time.sleep(3)
    else:
        raise RuntimeError("KO - Impossible de se connecter à Elasticsearch")

    # ---- CREATE INDEX ----
    if not es.indices.exists(index=INDEX_PROD):
        es.indices.create(index=INDEX_PROD)
        print(f"OK - Index {INDEX_PROD} créé.")
    else:
        print(f"Index {INDEX_PROD} existe déjà.")

    # ---- LOAD MODEL ----
    model, BEST_THRESHOLD = get_model_and_threshold()
    
    # BOOSTER XGBOOST
    booster = model.get_booster()   
    app.state.model = model
    app.state.booster = booster     
    app.state.threshold = BEST_THRESHOLD
    
 
    yield  # Permet de gérer le cycle de lifespan (avant/après)

    # ---- SHUTDOWN (optionnel) ----
    print("API shutdown")


app = FastAPI(title="Credit Default API", lifespan=lifespan)

# Middleware
setup_middleware(app)

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # Lancement du profiler pour la prédiction
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Récupération du model et du seuil chargés au lancement de l'API et non à chaque prédiction
    model = app.state.model
    BEST_THRESHOLD = app.state.threshold
    print(f"PREDICT Model - {model.__class__.__name__}")
        
    try:
        input_dict = data.model_dump()
        # En commentaire suite à optimisation 1
        # df = pd.DataFrame([input_dict])
        # df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        # df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        
        # Optimisation 1
        ordered_values = [
            float(input_dict.get(col) or 0)
            for col in model.feature_names_in_
        ]
        array = np.array([ordered_values], dtype=np.float32)

        # Optimisation 1 proba = float(model.predict_proba(df)[0][1])
        # Optimisation 1 => proba = float(model.predict_proba(array)[0][1])
        # Optimisation 2
        proba = app.state.booster.inplace_predict(array)

        prediction = int(proba >= BEST_THRESHOLD)
        
        # Fin du profile
        profiler.disable()

        # Enregistrement des valeurs du profile
        profiler.dump_stats("predict_profile.prof")


        return {"probability": proba, "prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {str(e)}")    

# Endpoint pour exécution du monitoring
@app.post("/run-drift")
def run_drift():
    run_global_monitoring()
    return {"status": "Drift monitoring executed"}


