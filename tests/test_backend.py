from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    """Vérifie que le point de terminaison santé répond bien."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_success():
    """Vérifie une prédiction réussie avec des données valides."""
    payload = {
        "MedInc": 3.5,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


def test_predict_invalid_data():
    """Vérifie que l'API renvoie une erreur 422 si les données sont incomplètes."""
    response = client.post("/predict", json={"MedInc": 3.5})
    assert response.status_code == 422
