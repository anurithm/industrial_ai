"""
ML model training: Logistic Regression, Random Forest, and XGBoost.
Handles class imbalance, threshold tuning, and saves best model.
"""
import logging
import json
import time
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Will train only LR and RF.")

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_SEED, MODEL_VERSION

logger = logging.getLogger(__name__)

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "lr_model.joblib",
    "random_forest": MODELS_DIR / "rf_model.joblib",
    "xgboost": MODELS_DIR / "xgb_model.joblib",
    "best": MODELS_DIR / "best_model.joblib",
}

THRESHOLD_PATHS = {
    "logistic_regression": MODELS_DIR / "lr_threshold.json",
    "random_forest": MODELS_DIR / "rf_threshold.json",
    "xgboost": MODELS_DIR / "xgb_threshold.json",
    "best": MODELS_DIR / "best_threshold.json",
}


def _compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute class weight ratio for imbalanced data."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    return n_neg / max(n_pos, 1)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find threshold that maximises F1 on validation set.
    Used to tune decision boundary without touching the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = np.argmax(f1_scores[:-1])  # exclude last (trivial) threshold
    best_threshold = float(thresholds[best_idx])
    logger.info(f"Best threshold: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
    return best_threshold


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
    latency_ms: float = 0.0,
) -> Dict:
    """Compute comprehensive evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "inference_latency_ms": round(latency_ms, 3),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return metrics


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    scale = _compute_scale_pos_weight(y_train)
    model = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: scale},
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    scale = _compute_scale_pos_weight(y_train)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight={0: 1.0, 1: scale},
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> Optional[object]:
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, skipping.")
        return None
    scale = _compute_scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model, model_name: str,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Tuple[Dict, float]:
    """Evaluate model on validation set, tune threshold, return metrics and threshold."""
    start = time.perf_counter()
    y_prob = model.predict_proba(X_val)[:, 1]
    latency = (time.perf_counter() - start) * 1000 / len(y_val)

    threshold = _find_best_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_val, y_pred, y_prob, latency_ms=latency)
    metrics["threshold"] = threshold
    metrics["model_name"] = model_name
    metrics["n_val"] = len(y_val)

    logger.info(
        f"{model_name}: F1={metrics['f1_score']:.4f}, "
        f"PR-AUC={metrics.get('pr_auc', 'N/A')}, "
        f"ROC-AUC={metrics.get('roc_auc', 'N/A')}"
    )
    return metrics, threshold


def train_and_compare(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Dict:
    """Train all models, evaluate on validation set, select best model."""
    all_metrics = {}
    models = {}

    logger.info("Training Logistic Regression (baseline)...")
    lr = train_logistic_regression(X_train, y_train)
    lr_metrics, lr_threshold = evaluate_model(lr, "logistic_regression", X_val, y_val)
    all_metrics["logistic_regression"] = lr_metrics
    models["logistic_regression"] = (lr, lr_threshold)
    joblib.dump(lr, MODEL_PATHS["logistic_regression"])

    logger.info("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_metrics, rf_threshold = evaluate_model(rf, "random_forest", X_val, y_val)
    all_metrics["random_forest"] = rf_metrics
    models["random_forest"] = (rf, rf_threshold)
    joblib.dump(rf, MODEL_PATHS["random_forest"])

    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost...")
        xgb = train_xgboost(X_train, y_train)
        if xgb is not None:
            xgb_metrics, xgb_threshold = evaluate_model(xgb, "xgboost", X_val, y_val)
            all_metrics["xgboost"] = xgb_metrics
            models["xgboost"] = (xgb, xgb_threshold)
            joblib.dump(xgb, MODEL_PATHS["xgboost"])

    # Select best model based on PR-AUC (primary) then F1 (secondary)
    # PR-AUC is preferred for imbalanced binary classification
    best_name = max(
        all_metrics.keys(),
        key=lambda k: (
            all_metrics[k].get("pr_auc") or 0,
            all_metrics[k]["f1_score"],
        ),
    )
    best_model, best_threshold = models[best_name]
    logger.info(f"Best model: {best_name}")

    # Save best model
    joblib.dump(best_model, MODEL_PATHS["best"])
    with open(THRESHOLD_PATHS["best"], "w") as f:
        json.dump({
            "threshold": best_threshold,
            "model_name": best_name,
            "model_version": MODEL_VERSION,
        }, f, indent=2)

    # Save all metrics
    results = {
        "model_comparison": all_metrics,
        "best_model": best_name,
        "model_version": MODEL_VERSION,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    metrics_path = REPORTS_DIR / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Validation metrics saved to {metrics_path}")

    return results


def evaluate_on_test(
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict:
    """Final evaluation on the held-out test set using the best model."""
    model = joblib.load(MODEL_PATHS["best"])
    with open(THRESHOLD_PATHS["best"]) as f:
        threshold_info = json.load(f)
    threshold = threshold_info["threshold"]

    start = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    latency = (time.perf_counter() - start) * 1000 / len(y_test)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob, latency_ms=latency)
    metrics["threshold"] = threshold
    metrics["model_name"] = threshold_info.get("model_name", "best")
    metrics["model_version"] = MODEL_VERSION
    metrics["n_test"] = len(y_test)
    metrics["dataset"] = "AI4I 2020 Predictive Maintenance (UCI) — synthetic"

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")
    logger.info(
        f"Test — F1={metrics['f1_score']:.4f}, "
        f"PR-AUC={metrics.get('pr_auc')}, ROC-AUC={metrics.get('roc_auc')}"
    )
    return metrics
