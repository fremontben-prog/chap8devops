import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "CreditDefaultModel"
MODEL_ALIAS = "staging"

def load_model_and_threshold():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
    )

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
     
    client = MlflowClient()

    # Récupérer la version via alias
    model_version = client.get_model_version_by_alias(
        MODEL_NAME, MODEL_ALIAS
    )

    run_id = model_version.run_id

    # Charger le modèle
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    # Récupérer le run
    run = client.get_run(run_id)

    # Récupérer le threshold
    best_threshold = float(run.data.params.get("best_threshold", 0.5))

    return model, best_threshold


# Charger le modèle une seule fois au démarrage
model, BEST_THRESHOLD = load_model_and_threshold()




