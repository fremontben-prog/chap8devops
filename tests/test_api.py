import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Charge les fichiers JSON depuis la racine
file_path = os.path.join(os.path.dirname(__file__), "../donnees_test_true.json")
with open(file_path, "r") as f:
    test_data_true = json.load(f)
    
file_path = os.path.join(os.path.dirname(__file__), "../donnees_test_false.json")
with open(file_path, "r") as f:
    test_data_false = json.load(f)

# Tests pour des valeurs de clients à true
def test_prediction_true():
    for row in test_data_true:
        response = client.post("/predict", json={"features": row})
        assert response.status_code == 200

        json_resp = response.json()
        assert "prediction" in json_resp
        pred = json_resp["prediction"]
        assert isinstance(pred, (float, int))
        assert 1 == pred 
        
# Tests pour des valeurs de clients à false
def test_prediction_false():
    for row in test_data_false:
        response = client.post("/predict", json={"features": row})
        assert response.status_code == 200

        json_resp = response.json()
        assert "prediction" in json_resp
        pred = json_resp["prediction"]
        assert isinstance(pred, (float, int))
        assert 0 == pred 

# Test de l’état de santé de l’API
def test_health():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json={
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7
        }
    })
    assert response.status_code in [200, 400]
