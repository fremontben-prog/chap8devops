import mlflow
import os

MODEL_NAME = "CreditDefaultModel"
MODEL_ALIAS = "staging"   # ou production

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Chargement au démarrage
model = load_model()
