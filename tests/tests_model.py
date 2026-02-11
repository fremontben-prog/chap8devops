from src.model import model

def test_prediction():
    result = model.predict({"surface": 50, "rooms": 2})
    assert result[0] == 50000
