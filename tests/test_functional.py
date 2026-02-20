from fastapi.testclient import TestClient
from api.model_loader import get_model_and_threshold
from api.main import app

import json
import os
import logging
import pytest
import time

from pathlib import Path 


client = TestClient(app)

# Récupération du seuil
_, BEST_THRESHOLD = get_model_and_threshold()

# Chemin des fichiers racine projet
PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)    

# --------------------------
# Fonctions utilitaires
# --------------------------
def load_test_data(filename):
    file_path = PROJECT_ROOT / "outputs" / filename
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(file_path, "r") as f:
        return json.load(f)
    
def check_prediction(row, json_resp, expected=None):
    assert "probability" in json_resp
    assert "prediction" in json_resp

    proba = json_resp["probability"]
    pred = json_resp["prediction"]

    # Types
    assert isinstance(proba, float)
    assert isinstance(pred, int)
    # Probabilité valide
    assert 0.0 <= proba <= 1.0
    # Cohérence seuil
    if proba >= BEST_THRESHOLD:
        assert pred == 1
    else:
        assert pred == 0
    # Optionnel : vérifier avec valeur attendue
    if expected is not None:
        assert expected in [0, 1]

    logger.info(f"Proba={proba:.4f} | Pred={pred} | Expected={expected}")
    
       
###########################################################

# --------------------------
# 1️⃣ Tests rapides (smoke) – pour CI rapide
# --------------------------
@pytest.mark.smoke
@pytest.mark.parametrize("filename", ["donnees_test_true.json", "donnees_test_false.json"])
def test_smoke_prediction(filename):
    """Test rapide pour CI, vérifie structure et cohérence sur un petit échantillon"""
    test_data = load_test_data(OUTPUT_DIR / filename)
    for row in test_data:  
        row_copy = row.copy()
        expected = row_copy.pop("TARGET", None)
        response = client.post("/predict", json=row_copy)
        assert response.status_code == 200
        check_prediction(row, response.json(), expected)

# --------------------------
# 2️⃣ Tests longs / robustesse – moins fréquents
# --------------------------
@pytest.mark.long
@pytest.mark.parametrize("filename", ["donnees_test_full_true.json", "donnees_test_full_false.json"])
def test_full_prediction(filename):
    """Test complet sur toutes les données pour non-régression"""
    test_data = load_test_data(filename)
    for row in test_data:
        row_copy = row.copy()
        expected = row_copy.pop("TARGET", None)
        response = client.post("/predict", json=row_copy)
        assert response.status_code == 200
        check_prediction(row, response.json(), expected)

# --------------------------
# 3️⃣ Tests de validité / erreurs
# --------------------------
@pytest.mark.smoke
@pytest.mark.parametrize("invalid_row", [
    {},  # tout vide
    {"AMT_CREDIT": "invalid_string"},  # type incorrect
    {"AMT_CREDIT": None, "DAYS_BIRTH": None},  # valeurs nulles
])
def test_invalid_data(invalid_row):
    response = client.post("/predict", json=invalid_row)
    # L'API doit renvoyer une erreur 400 ou 422
    assert response.status_code in [400, 422]

# --------------------------
# 4️⃣ Tests performance
# --------------------------
@pytest.mark.smoke
def test_prediction_performance():
    test_data = load_test_data("donnees_test_true.json")
    for row in test_data[:5]:
        start = time.time()
        response = client.post("/predict", json=row)
        end = time.time()
        assert response.status_code == 200
        assert (end - start) < 0.5  # réponse rapide pour CI

# --------------------------
# 5️⃣ Tests limites / valeurs extrêmes
# --------------------------
@pytest.mark.long
@pytest.mark.parametrize("extreme_row", [
    {"AMT_CREDIT": 1e9, "AMT_INCOME_TOTAL": 1e8, "DAYS_BIRTH": -20000},
    {"AMT_CREDIT": 0, "AMT_INCOME_TOTAL": 0, "DAYS_BIRTH": -1},
])
def test_extreme_values(extreme_row):
    response = client.post("/predict", json=extreme_row)
    assert response.status_code == 200
    check_prediction(extreme_row, response.json())

# --------------------------
# 6️⃣ Tests golden set – non-régression
# --------------------------
@pytest.mark.long
def test_golden_set():
    golden_data = load_test_data("donnees_test_true.json")
    for row in golden_data:
        row_copy = row.copy()
        expected = row_copy.pop("TARGET", None)
        response = client.post("/predict", json=row_copy)
        check_prediction(row, response.json(), expected)