from fastapi.testclient import TestClient
from api.model_loader import BEST_THRESHOLD
import json
import os

# Charge les fichiers JSON depuis la racine
file_path = os.path.join(os.path.dirname(__file__), "../donnees_test_true.json")
with open(file_path, "r") as f:
    test_data_true = json.load(f)
    
file_path = os.path.join(os.path.dirname(__file__), "../donnees_test_false.json")
with open(file_path, "r") as f:
    test_data_false = json.load(f)

client = TestClient(app)

# Tests pour des valeurs de clients à true
def test_prediction_true():
    for row in test_data_true:
        response = client.post("/predict", json={"features": row["features"]})
        assert response.status_code == 200

        json_resp = response.json()

        # Vérifie structure
        assert "probability" in json_resp
        assert "prediction" in json_resp

        proba = json_resp["probability"]
        pred = json_resp["prediction"]

        # Vérifie types
        assert isinstance(proba, float)
        assert isinstance(pred, int)

        # Vérifie cohérence logique
        assert pred == row["expected_prediction"]

        # Vérifie cohérence seuil
        if proba >= BEST_THRESHOLD:
            assert pred == 1
        else:
            assert pred == 0
        
# Tests pour des valeurs de clients à false
def test_prediction_false():
    for row in test_data_false:
        response = client.post("/predict", json={"features": row["features"]})
        assert response.status_code == 200

        json_resp = response.json()

        # Vérifie structure
        assert "probability" in json_resp
        assert "prediction" in json_resp

        proba = json_resp["probability"]
        pred = json_resp["prediction"]

        # Vérifie types
        assert isinstance(proba, float)
        assert isinstance(pred, int)

        # Vérifie cohérence logique
        assert pred == row["expected_prediction"]

        # Vérifie cohérence seuil
        if proba < BEST_THRESHOLD:
            assert pred == 0
        else:
            assert pred == 1
