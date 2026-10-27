"""
SHAP-based explainability for the IndustrialAI platform.
Provides global feature importance and local (per-prediction) explanations.
"""
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Using permutation importance as fallback.")

from src.config import MODELS_DIR
from src.preprocessing import load_preprocessor, get_feature_names
from src.predict import load_model

_explainer_cache = None


def _get_explainer(model, X_background: np.ndarray):
    """Create or return cached SHAP explainer."""
    global _explainer_cache
    if _explainer_cache is not None:
        return _explainer_cache

    if not SHAP_AVAILABLE:
        return None

    try:
        # TreeExplainer for tree-based models (RF, XGBoost) - fast and exact
        if hasattr(model, "estimators_") or "xgb" in type(model).__name__.lower():
            explainer = shap.TreeExplainer(model)
        else:
            # For LR, use LinearExplainer with background data
            background = shap.sample(X_background, min(100, len(X_background)))
            explainer = shap.LinearExplainer(model, background)
        _explainer_cache = explainer
        return explainer
    except Exception as e:
        logger.warning(f"Could not create TreeExplainer: {e}. Falling back to KernelExplainer.")
        try:
            background = shap.sample(X_background, min(50, len(X_background)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            _explainer_cache = explainer
            return explainer
        except Exception as e2:
            logger.error(f"SHAP explainer creation failed: {e2}")
            return None


def get_global_importance(X_background: np.ndarray) -> Dict:
    """
    Compute global feature importance using SHAP mean absolute values.
    Falls back to model's built-in feature_importances_ if SHAP fails.
    """
    model, _, _ = load_model()
    preprocessor = load_preprocessor()
    feature_names = get_feature_names(preprocessor)

    if SHAP_AVAILABLE:
        try:
            explainer = _get_explainer(model, X_background)
            if explainer is not None:
                sample = X_background[:min(200, len(X_background))]
                shap_values = explainer.shap_values(sample)

                # For binary classifiers, shap_values may be a list [class0, class1]
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                else:
                    sv = shap_values

                mean_abs_shap = np.abs(sv).mean(axis=0)
                importance = dict(zip(feature_names, mean_abs_shap.tolist()))
                sorted_importance = dict(
                    sorted(importance.items(), key=lambda x: x[1], reverse=True)
                )
                return {
                    "method": "SHAP mean |value|",
                    "feature_importance": sorted_importance,
                    "top_feature": max(importance, key=importance.get),
                    "disclaimer": (
                        "Feature importance reflects model behaviour, not physical causation."
                    ),
                }
        except Exception as e:
            logger.warning(f"SHAP global importance failed: {e}")

    # Fallback: model built-in importance
    return _fallback_importance(model, feature_names)


def _fallback_importance(model, feature_names: List[str]) -> Dict:
    """Use model's built-in feature importance if SHAP fails."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

        importance = dict(zip(feature_names, importances.tolist()))
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        return {
            "method": "Model built-in importance (SHAP unavailable)",
            "feature_importance": sorted_importance,
            "top_feature": max(importance, key=importance.get),
            "disclaimer": "Feature importance reflects model behaviour, not physical causation.",
        }
    except Exception as e:
        logger.error(f"Fallback importance failed: {e}")
        return {"method": "unavailable", "feature_importance": {}}


def explain_prediction(
    sensor_data: Dict, X_background: np.ndarray
) -> Dict:
    """
    Generate local (per-prediction) SHAP explanation for a single record.

    Returns:
        dict with feature contributions and a plain-language summary.
    """
    import pandas as pd
    from src.config import FEATURE_COLUMNS
    from src.preprocessing import preprocess_single

    model, _, _ = load_model()
    preprocessor = load_preprocessor()
    feature_names = get_feature_names(preprocessor)

    X_instance = preprocess_single(sensor_data, preprocessor)

    if SHAP_AVAILABLE:
        try:
            explainer = _get_explainer(model, X_background)
            if explainer is not None:
                shap_values = explainer.shap_values(X_instance)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]

                contributions = dict(zip(feature_names, sv.tolist()))
                sorted_contribs = dict(
                    sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                )

                # Plain-language summary
                top_pos = [k for k, v in sorted_contribs.items() if v > 0][:2]
                top_neg = [k for k, v in sorted_contribs.items() if v < 0][:2]

                summary = _build_summary(top_pos, top_neg, sorted_contribs)

                return {
                    "method": "SHAP",
                    "feature_contributions": sorted_contribs,
                    "summary": summary,
                    "top_risk_factors": top_pos,
                    "top_protective_factors": top_neg,
                    "disclaimer": (
                        "SHAP values explain this model's prediction, not physical causation. "
                        "Positive values increase predicted failure probability."
                    ),
                }
        except Exception as e:
            logger.warning(f"SHAP local explanation failed: {e}")

    # Fallback
    return _fallback_importance(model, feature_names)


def _build_summary(
    top_pos: List[str], top_neg: List[str], contribs: Dict
) -> str:
    """Generate a plain-language summary of feature contributions."""
    lines = ["Key factors increasing failure risk:"]
    for feat in top_pos:
        lines.append(f"  • {feat}: +{contribs[feat]:.4f} (increases risk prediction)")
    if top_neg:
        lines.append("Key factors reducing failure risk:")
        for feat in top_neg:
            lines.append(f"  • {feat}: {contribs[feat]:.4f} (reduces risk prediction)")
    lines.append(
        "\nNote: These values reflect the model's internal reasoning and do not "
        "establish physical causation between sensor readings and failures."
    )
    return "\n".join(lines)
