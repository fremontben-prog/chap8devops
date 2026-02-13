import os
import mlflow
import json
from mlflow.tracking import MlflowClient
import joblib  # pour charger le modèle pré-enregistré localement

from pathlib import Path

MODEL_NAME = "CreditDefaultModel"
MODEL_ALIAS = "staging"

OUTPUT_DIR = Path("outputs")
LOCAL_MODEL_FILE = OUTPUT_DIR / "model.pkl"
LOCAL_THRESHOLD_FILE = OUTPUT_DIR / "best_threshold.json"

class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        # retourne toujours la même prédiction pour le test
        return np.array([[0.3, 0.7]])

def load_model_and_threshold():
    env = os.getenv("ENV", "prod")

    # --------------------
    # Mode test : dummy ou modèle local
    # --------------------
    if env == "test":
        # Si on a dumpé le modèle localement
        if os.path.exists(LOCAL_MODEL_FILE):
            print("=== MODE TEST ===")
            print(f"Model path: {LOCAL_MODEL_FILE}")
            print(f"Threshold path: {LOCAL_THRESHOLD_FILE}")

            model = joblib.load(LOCAL_MODEL_FILE)

            print(f"Model type: {type(model)}")
            print(f"Model attributes: {dir(model)}")

            with open(LOCAL_THRESHOLD_FILE, "r") as f:
                threshold_data = json.load(f)
                print(f"Threshold file content: {threshold_data}")

                best_threshold = threshold_data["best_threshold"]
                print(f"Best threshold loaded: {best_threshold}")
            return model, best_threshold
        # Sinon fallback Dummy
        return DummyModel(), 0.5

    # --------------------
    # Mode prod : MLflow réel
    # --------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001"))

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = model_version.run_id
    run = client.get_run(run_id)
    best_threshold = float(run.data.params.get("best_threshold", 0.5))

    return model, best_threshold

# Charger le modèle une seule fois au démarrage
model, BEST_THRESHOLD = load_model_and_threshold()