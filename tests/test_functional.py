from fastapi.testclient import TestClient
from api.model_loader import BEST_THRESHOLD
from api.main import app
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
    # Charge les fichiers JSON depuis la racine
    file_path = os.path.join(os.path.dirname(__file__), "../donnees_test_true.json")
    with open(file_path, "r") as f:
        test_data_true = json.load(f)
        
    for row in test_data_true:

        row_copy = row.copy()
        expected = row_copy.pop("TARGET")

        response = client.post("/predict", json=row_copy)
        assert response.status_code == 200

        json_resp = response.json()

        assert "probability" in json_resp
        assert "prediction" in json_resp
        
        # Vérifie structure
        proba = json_resp["probability"]
        pred = json_resp["prediction"]

        assert isinstance(proba, float)
        assert isinstance(pred, int)


        if proba >= BEST_THRESHOLD:
            assert pred == 1
        else:
            assert pred == 0
        
# Tests pour des valeurs de clients à false
def test_prediction_false():
    file_path = os.path.join(os.path.dirname(__file__), "../donnees_test_false.json")
    with open(file_path, "r") as f:
        test_data_false = json.load(f)
        
    for row in test_data_false:
        row_copy = row.copy()
        expected = row_copy.pop("TARGET")
        
        response = client.post("/predict", json=row_copy)
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

        # Cohérence seuil
        if proba < BEST_THRESHOLD:
            assert pred == 1
        else:
            assert pred == 0
