"""
Tests for model training, evaluation, and prediction functions.
"""
import json
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"


def _make_training_data(n=300):
    """Generate minimal synthetic data for unit tests."""
    np.random.seed(42)
    X = np.random.randn(n, 7)
    y = np.random.choice([0, 1], n, p=[0.97, 0.03])
    # Ensure at least 2 positive samples in each split
    y[0] = 1
    y[1] = 1
    y[n // 2] = 1
    y[n // 2 + 1] = 1
    y[n - 1] = 1
    y[n - 2] = 1
    return X, y


def test_logistic_regression_trains():
    """Logistic Regression fits without error."""
    from src.train_model import train_logistic_regression
    X, y = _make_training_data()
    model = train_logistic_regression(X, y)
    preds = model.predict_proba(X[:5])
    assert preds.shape == (5, 2)
    assert all(0.0 <= p <= 1.0 for p in preds[:, 1])


def test_random_forest_trains():
    """Random Forest fits without error."""
    from src.train_model import train_random_forest
    X, y = _make_training_data()
    model = train_random_forest(X, y)
    preds = model.predict_proba(X[:5])
    assert preds.shape == (5, 2)


def test_compute_metrics_structure():
    """compute_metrics returns all required fields."""
    from src.train_model import compute_metrics
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.4, 0.1])
    metrics = compute_metrics(y_true, y_pred, y_prob, latency_ms=1.0)
    for key in ["accuracy", "precision", "recall", "f1_score",
                "true_positives", "false_positives", "false_negatives",
                "roc_auc", "pr_auc", "confusion_matrix"]:
        assert key in metrics, f"Missing metric: {key}"


def test_threshold_tuning():
    """Best threshold should be between 0 and 1."""
    from src.train_model import _find_best_threshold
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.25])
    threshold = _find_best_threshold(y_true, y_prob)
    assert 0.0 <= threshold <= 1.0


@pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="Model not trained — skipping prediction tests"
)
def test_predict_normal_input():
    """Predict returns valid structured output for normal inputs."""
    from src.predict import predict
    data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    result = predict(data)
    assert "predicted_failure" in result
    assert "failure_probability" in result
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"}
    assert "disclaimer" in result


@pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="Model not trained — skipping"
)
def test_predict_returns_timestamp():
    """Prediction includes a timestamp."""
    from src.predict import predict
    data = {
        "Type": "L",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 2000,
        "Torque [Nm]": 30.0,
        "Tool wear [min]": 50,
    }
    result = predict(data)
    assert "prediction_timestamp" in result
    assert len(result["prediction_timestamp"]) > 10


@pytest.mark.skipif(
    not (MODELS_DIR / "anomaly_detector.joblib").exists(),
    reason="Anomaly detector not trained — skipping"
)
def test_anomaly_detection_output():
    """detect_anomaly returns required fields."""
    from src.anomaly_detection import detect_anomaly
    data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    result = detect_anomaly(data)
    assert "is_anomaly" in result
    assert "anomaly_score_raw" in result
    assert "anomaly_score_normalised" in result
    assert isinstance(result["is_anomaly"], bool)
    assert 0.0 <= result["anomaly_score_normalised"] <= 1.0
