"""
Data loading and validation for the UCI AI4I 2020 Predictive Maintenance dataset.
Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from src.config import (
    DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN,
    FAILURE_MODE_COLUMNS, LEAKAGE_COLUMNS
)

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = {
    "Type": "object",
    "Air temperature [K]": "float64",
    "Process temperature [K]": "float64",
    "Rotational speed [rpm]": "int64",
    "Torque [Nm]": "float64",
    "Tool wear [min]": "int64",
    "Machine failure": "int64",
}

FEATURE_RANGES = {
    "Air temperature [K]": (290.0, 310.0),
    "Process temperature [K]": (295.0, 320.0),
    "Rotational speed [rpm]": (1000, 3000),
    "Torque [Nm]": (0.0, 100.0),
    "Tool wear [min]": (0, 300),
}

VALID_TYPES = {"L", "M", "H"}


COLUMN_RENAME_MAP = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UCI column shortnames to include units, as expected by the schema."""
    return df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})


def download_dataset() -> bool:
    """
    Attempt to download the dataset using the ucimlrepo package.
    Returns True if successful, False otherwise.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Downloading UCI AI4I 2020 dataset...")
        dataset = fetch_ucirepo(id=601)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df = _normalize_columns(df)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Dataset saved to {DATASET_PATH} ({len(df)} rows)")
        return True
    except Exception as e:
        logger.warning(f"Auto-download failed: {e}")
        return False


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw CSV dataset. Attempts auto-download if file not found.

    Returns:
        DataFrame with raw data.
    Raises:
        FileNotFoundError if dataset cannot be found or downloaded.
    """
    path = path or DATASET_PATH

    if not path.exists():
        logger.info(f"Dataset not found at {path}. Attempting auto-download...")
        if not download_dataset():
            raise FileNotFoundError(
                f"Dataset not found at {path}.\n"
                "Please download it manually:\n"
                "  1. Visit https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
                "  2. Download the CSV and place it at: data/raw/ai4i2020.csv\n"
                "  OR run: python -c \"from ucimlrepo import fetch_ucirepo; "
                "import pandas as pd; d=fetch_ucirepo(id=601); "
                "pd.concat([d.data.features,d.data.targets],axis=1).to_csv('data/raw/ai4i2020.csv',index=False)\""
            )

    df = pd.read_csv(path)
    df = _normalize_columns(df)  # Normalize column names in case CSV has short names
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate dataset schema and report issues.

    Returns:
        dict with validation results and any warnings.
    """
    issues = []
    warnings_list = []

    # Check required columns are present
    all_expected = list(EXPECTED_SCHEMA.keys()) + FAILURE_MODE_COLUMNS
    missing = [c for c in all_expected if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check for target leakage risk
    feature_cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    leakage_check = [c for c in LEAKAGE_COLUMNS if c in df.columns and c in feature_cols_present]
    if leakage_check:
        issues.append(f"LEAKAGE RISK: These columns are in features AND leakage list: {leakage_check}")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings_list.append(f"Found {dup_count} duplicate rows")

    # Missing values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        warnings_list.append(f"Missing value columns: {cols_with_nulls.to_dict()}")

    # Target distribution
    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts()
        failure_rate = target_dist.get(1, 0) / len(df)
        warnings_list.append(f"Target distribution: {target_dist.to_dict()} ({failure_rate:.1%} failure rate)")

    # Type column values
    if "Type" in df.columns:
        invalid_types = set(df["Type"].unique()) - VALID_TYPES
        if invalid_types:
            issues.append(f"Invalid 'Type' values: {invalid_types}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings_list,
        "shape": df.shape,
        "columns": list(df.columns),
    }


def validate_single_input(data: dict) -> Tuple[bool, list]:
    """
    Validate a single sensor reading for prediction.

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings_list = []

    for col, (low, high) in FEATURE_RANGES.items():
        if col in data:
            val = data[col]
            if val < low or val > high:
                warnings_list.append(
                    f"{col} = {val} is outside expected range [{low}, {high}]"
                )

    if "Type" in data and data["Type"] not in VALID_TYPES:
        warnings_list.append(f"Type must be one of {VALID_TYPES}, got: {data['Type']}")

    temp_air = data.get("Air temperature [K]")
    temp_proc = data.get("Process temperature [K]")
    if temp_air and temp_proc:
        diff = temp_proc - temp_air
        if diff < 0 or diff > 20:
            warnings_list.append(
                f"Process temp minus Air temp = {diff:.1f} K; expected 8–10 K difference typically"
            )

    return True, warnings_list


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only permitted feature columns and prevent target leakage.
    Drops identifier and leakage columns.
    """
    # Only keep columns that are in FEATURE_COLUMNS and actually present
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available) != len(FEATURE_COLUMNS):
        missing = set(FEATURE_COLUMNS) - set(available)
        logger.warning(f"Some feature columns missing: {missing}")
    return df[available].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    """Extract the target column."""
    return df[TARGET_COLUMN].copy()
