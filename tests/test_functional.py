from fastapi.testclient import TestClient
from api.model_loader import get_model_and_threshold
from api.main import app
from src.feature import (clean_feature_names, clean_object_type)
import json
import os
import logging


client = TestClient(app)

_, BEST_THRESHOLD = get_model_and_threshold()

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
        print(f"BFR reponse json {response.json()}")
        assert response.status_code == 200

        json_resp = response.json()

        assert "probability" in json_resp
        assert "prediction" in json_resp
        
        # Vérifie structure
        proba = json_resp["probability"]
        pred = json_resp["prediction"]

        assert isinstance(proba, float)
        assert isinstance(pred, int)

        logger = logging.getLogger(__name__)
        # Cette ligne s'affichera automatiquement dans GitHub Actions
        logger.info(f"Vérification true : Proba={proba:.4f} | Pred={pred} | Seuil={BEST_THRESHOLD} | Expected ={expected}")
        print(f"Vérification : Proba={proba:.4f} | Pred={pred} | Seuil={BEST_THRESHOLD} | Expected ={expected}")

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

        # Log des valeurs de prédiction
        logger = logging.getLogger(__name__)
        # Cette ligne s'affichera automatiquement dans GitHub Actions
        logger.info(f"Vérification false : Proba={proba:.4f} | Pred={pred} | Seuil={BEST_THRESHOLD} | Expected ={expected}")
        #
        print(f"Vérification false : Proba={proba:.4f} | Pred={pred} | Seuil={BEST_THRESHOLD} | Expected ={expected}")
        
        # Vérifie types
        assert isinstance(proba, float)
        assert isinstance(pred, int)

        # Cohérence seuil
        if proba < BEST_THRESHOLD:
            assert pred == 0
        else:
            assert pred == 1
