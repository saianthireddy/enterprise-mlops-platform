import os

os.environ["REGISTRY_ROOT"] = "artifacts/test_registry"

from fastapi.testclient import TestClient

from mlops_platform.api.main import app
from mlops_platform.training.pipeline import make_synthetic_dataset

client = TestClient(app)


def test_health_reports_production_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["production_model"].startswith("churn-classifier/v")


def test_models_lists_registry():
    response = client.get("/models")
    assert response.status_code == 200
    assert any(m["stage"] == "production" for m in response.json())


def test_predict_returns_probability():
    response = client.post("/predict", json={"features": [0.5, -1.2, 0.3, 0.0, 1.1, -0.4]})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_validates_feature_count():
    assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422


def test_drift_endpoint_flags_shifted_window():
    shifted, _ = make_synthetic_dataset(300, seed=8, shift=1.5)
    response = client.post("/drift/check", json={"current": shifted.tolist()})
    assert response.status_code == 200
    assert response.json()["has_drift"] is True
