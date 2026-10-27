"""
Preprocessing pipeline for the AI4I predictive maintenance dataset.
Ensures reproducibility and prevents target leakage.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    FEATURE_COLUMNS, TARGET_COLUMN,
    PROCESSED_DATA_DIR, MODELS_DIR,
)
from src.data_loader import prepare_features, get_target

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def build_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for consistent preprocessing."""
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
            pd.Series, pd.Series, pd.Series]:
    """
    Stratified train/val/test split.
    Processing order: full → hold out test → split remainder into train/val.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Val is a fraction of the remaining train+val
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    logger.info(
        f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    logger.info(
        f"Failure rates — Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessor ONLY on training data."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocessor fitted and saved to {PREPROCESSOR_PATH}")
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Load a previously fitted preprocessor."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. Run training first."
        )
    return joblib.load(PREPROCESSOR_PATH)


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Apply the fitted preprocessor to a feature DataFrame."""
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Get feature names after preprocessing (for SHAP)."""
    num_names = NUMERIC_FEATURES.copy()
    try:
        cat_enc = preprocessor.named_transformers_["cat"]["onehot"]
        cat_names = list(cat_enc.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_names = []
    return num_names + cat_names


def prepare_pipeline(df: pd.DataFrame):
    """
    Full preprocessing pipeline: select features, split, and fit preprocessor.
    Returns all splits and the fitted preprocessor.
    """
    X = prepare_features(df)
    y = get_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Fit ONLY on training data
    preprocessor = fit_preprocessor(X_train)

    X_train_t = transform_features(preprocessor, X_train)
    X_val_t = transform_features(preprocessor, X_val)
    X_test_t = transform_features(preprocessor, X_test)

    # Save processed splits
    np.save(PROCESSED_DATA_DIR / "X_train.npy", X_train_t)
    np.save(PROCESSED_DATA_DIR / "X_val.npy", X_val_t)
    np.save(PROCESSED_DATA_DIR / "X_test.npy", X_test_t)
    np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train.values)
    np.save(PROCESSED_DATA_DIR / "y_val.npy", y_val.values)
    np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test.values)

    # Save raw feature splits for anomaly detection (unscaled)
    X_train.to_parquet(PROCESSED_DATA_DIR / "X_train_raw.parquet")
    X_val.to_parquet(PROCESSED_DATA_DIR / "X_val_raw.parquet")
    X_test.to_parquet(PROCESSED_DATA_DIR / "X_test_raw.parquet")

    logger.info("All preprocessed splits saved.")
    return X_train_t, X_val_t, X_test_t, y_train, y_val, y_test, preprocessor


def preprocess_single(data: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Preprocess a single input dictionary for inference.
    Used by the prediction API.
    """
    df = pd.DataFrame([data])
    # Select only feature columns
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[feature_cols]
    return preprocessor.transform(df)
