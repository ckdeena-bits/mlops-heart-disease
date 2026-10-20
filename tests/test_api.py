import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info_endpoint(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "author" in data

def test_predict_endpoint_valid(client):
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "confidence" in data
    assert "risk_level" in data
    assert res.headers.get("X-Request-ID") is not None

def test_predict_endpoint_invalid_range(client):
    payload = {
        "age": 200, # Invalid age
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422 # Pydantic validation error

def test_predict_batch_endpoint(client):
    patient1 = {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
    }
    patient2 = {
        "age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    }
    res = client.post("/predict/batch", json={"patients": [patient1, patient2]})
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2

def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "heart_prediction_requests_total" in res.text
