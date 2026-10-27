"""
Test suite for data loading and preprocessing.
"""
import json
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    """Create a minimal valid dataset for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"L{i:05d}" for i in range(n)],
        "Type": np.random.choice(["L", "M", "H"], n),
        "Air temperature [K]": np.random.uniform(295, 305, n),
        "Process temperature [K]": np.random.uniform(305, 315, n),
        "Rotational speed [rpm]": np.random.randint(1200, 2800, n),
        "Torque [Nm]": np.random.uniform(20, 65, n),
        "Tool wear [min]": np.random.randint(0, 250, n),
        "Machine failure": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "TWF": np.zeros(n, dtype=int),
        "HDF": np.zeros(n, dtype=int),
        "PWF": np.zeros(n, dtype=int),
        "OSF": np.zeros(n, dtype=int),
        "RNF": np.zeros(n, dtype=int),
    })
    return df


def test_validate_schema_valid(sample_df):
    """Schema validation passes on clean dataset."""
    from src.data_loader import validate_schema
    result = validate_schema(sample_df)
    assert result["valid"] is True
    assert len(result["issues"]) == 0


def test_validate_schema_missing_column(sample_df):
    """Schema validation fails on missing required columns."""
    from src.data_loader import validate_schema
    df_bad = sample_df.drop(columns=["Torque [Nm]"])
    result = validate_schema(df_bad)
    assert "Torque [Nm]" in str(result["issues"])


def test_prepare_features_excludes_target(sample_df):
    """prepare_features must not include target or failure mode columns."""
    from src.data_loader import prepare_features, LEAKAGE_COLUMNS
    X = prepare_features(sample_df)
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns, f"Leak column {col} found in features!"


def test_prepare_features_only_allowed(sample_df):
    """prepare_features should only return permitted feature columns."""
    from src.data_loader import prepare_features
    from src.config import FEATURE_COLUMNS
    X = prepare_features(sample_df)
    for col in X.columns:
        assert col in FEATURE_COLUMNS


def test_split_is_stratified(sample_df):
    """Stratified split preserves failure rate across splits."""
    from src.data_loader import prepare_features, get_target
    from src.preprocessing import split_data
    X = prepare_features(sample_df)
    y = get_target(sample_df)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
    # Failure rates should be roughly similar (within 2x)
    overall_rate = y.mean()
    for split_y, name in [(y_tr, "train"), (y_v, "val"), (y_te, "test")]:
        rate = split_y.mean()
        assert abs(rate - overall_rate) < 0.05 or overall_rate < 0.01, \
            f"{name} failure rate {rate:.3f} differs too much from {overall_rate:.3f}"


def test_no_leakage_in_preprocessor(sample_df):
    """Preprocessor must be fit only on training data columns (not target)."""
    from src.data_loader import prepare_features, get_target
    from src.preprocessing import split_data, build_preprocessor
    X = prepare_features(sample_df)
    y = get_target(sample_df)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_tr)  # Should only see training data
    # Should transform test data without seeing its labels
    X_test_t = preprocessor.transform(X_te)
    assert X_test_t.shape[0] == len(X_te)


def test_validate_single_input_valid():
    """Validates that a normal sensor reading passes validation."""
    from src.data_loader import validate_single_input
    data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    is_valid, warnings = validate_single_input(data)
    assert is_valid is True


def test_validate_single_input_extreme():
    """Out-of-range readings should produce warnings."""
    from src.data_loader import validate_single_input
    data = {
        "Type": "M",
        "Air temperature [K]": 350.0,  # Way too high
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    is_valid, warnings = validate_single_input(data)
    assert len(warnings) > 0  # Should warn about high temp


def test_validate_invalid_type():
    """Invalid product type should produce a warning."""
    from src.data_loader import validate_single_input
    data = {
        "Type": "X",  # invalid
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    _, warnings = validate_single_input(data)
    assert any("Type" in w for w in warnings)
