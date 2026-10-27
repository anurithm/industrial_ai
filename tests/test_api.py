"""
Tests for the FastAPI backend endpoints.
Uses httpx TestClient to avoid needing a live server.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_EXISTS = (MODELS_DIR / "best_model.joblib").exists()

VALID_PAYLOAD = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.database import init_db
    init_db()
    from src.api import app
    return TestClient(app)


def test_health_endpoint(client):
    """Health endpoint returns 200 with expected fields."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_ready" in body
    assert "timestamp" in body


def test_health_status_ok(client):
    """Health endpoint reports status='ok'."""
    r = client.get("/health")
    assert r.json()["status"] == "ok"


def test_predict_invalid_type(client):
    """Invalid product type returns 422 validation error."""
    payload = {**VALID_PAYLOAD, "Type": "X"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_negative_rotational_speed(client):
    """Negative RPM returns validation error."""
    payload = {**VALID_PAYLOAD, "Rotational speed [rpm]": -100}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_field(client):
    """Missing required field returns 422."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Torque [Nm]"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained")
def test_predict_valid_input(client):
    """Valid input returns structured prediction response."""
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_failure" in body
    assert "failure_probability" in body
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert "risk_level" in body
    assert "model_name" in body
    assert "disclaimer" in body


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained")
def test_history_endpoint(client):
    """History endpoint returns list of records."""
    r = client.get("/history")
    assert r.status_code == 200
    body = r.json()
    assert "records" in body
    assert "count" in body
    assert isinstance(body["records"], list)


def test_maintenance_ask_valid(client):
    """Maintenance ask endpoint returns an answer."""
    r = client.post(
        "/maintenance/ask",
        json={"question": "What is the normal operating temperature range?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["answer"]) > 0
    assert "sources" in body


def test_maintenance_ask_short_question(client):
    """Too-short question returns validation error."""
    r = client.post("/maintenance/ask", json={"question": "x"})
    assert r.status_code == 422
