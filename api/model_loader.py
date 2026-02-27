import os
import mlflow
import json
import joblib
from pathlib import Path
from mlflow.tracking import MlflowClient

MODEL_NAME = "CreditDefaultModel"
MODEL_ALIAS = "staging"

OUTPUT_DIR = Path("outputs")
LOCAL_MODEL_FILE = OUTPUT_DIR / "final_model.pkl"
LOCAL_THRESHOLD_FILE = OUTPUT_DIR / "best_threshold.json"


model = None
BEST_THRESHOLD = None


def get_model_and_threshold():
    # Lazy loading : charge le modèle une seule fois.
    
    global model, BEST_THRESHOLD

    if model is None:
        model, BEST_THRESHOLD = load_model_and_threshold()

    return model, BEST_THRESHOLD


def load_model_and_threshold():
    env = os.getenv("ENV", "prod")
    print(f"ENV value: {env}")

    # ==========================
    # MODE TEST (CI / local dev)
    # ==========================
    if env == "test":
        print("MODE TEST")

        if not LOCAL_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model file not found: {LOCAL_MODEL_FILE}"
            )

        if not LOCAL_THRESHOLD_FILE.exists():
            raise FileNotFoundError(
                f"Threshold file not found: {LOCAL_THRESHOLD_FILE}"
            )

        print(f"Loading model from {LOCAL_MODEL_FILE}")
        model = joblib.load(LOCAL_MODEL_FILE)

        with open(LOCAL_THRESHOLD_FILE, "r") as f:
            threshold_data = json.load(f)

        best_threshold = float(threshold_data["best_threshold"])

        print(f"Loaded threshold: {best_threshold}")

        return model, best_threshold

    # ==========================
    # MODE PROD (MLflow)
    # ==========================
    print("MODE PROD (MLflow)")

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
    )

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = model_version.run_id
    run = client.get_run(run_id)

    best_threshold = float(run.data.params.get("best_threshold", 0.5))

    print(f"Loaded MLflow threshold: {best_threshold}")

    return model, best_threshold