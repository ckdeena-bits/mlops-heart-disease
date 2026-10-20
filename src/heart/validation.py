"""
Data validation schema for Heart Disease UCI Dataset.
Validates input types, allowed ranges, and missing value constraints.
"""
from typing import Any, Dict, List

import pandas as pd

FEATURE_RANGES: Dict[str, Dict[str, Any]] = {
    "age": {"type": "numeric", "min": 1, "max": 120},
    "sex": {"type": "categorical", "allowed": [0, 1]},
    "cp": {"type": "categorical", "allowed": [1, 2, 3, 4]},
    "trestbps": {"type": "numeric", "min": 50, "max": 250},
    "chol": {"type": "numeric", "min": 80, "max": 600},
    "fbs": {"type": "categorical", "allowed": [0, 1]},
    "restecg": {"type": "categorical", "allowed": [0, 1, 2]},
    "thalach": {"type": "numeric", "min": 50, "max": 250},
    "exang": {"type": "categorical", "allowed": [0, 1]},
    "oldpeak": {"type": "numeric", "min": 0.0, "max": 10.0},
    "slope": {"type": "categorical", "allowed": [1, 2, 3]},
    "ca": {"type": "categorical", "allowed": [0, 1, 2, 3]},
    "thal": {"type": "categorical", "allowed": [3, 6, 7]},
}

EXPECTED_FEATURES: List[str] = list(FEATURE_RANGES.keys())

def validate_dataframe(df: pd.DataFrame, is_training: bool = True) -> bool:
    """Validates dataframe against schema constraints."""
    missing_cols = set(EXPECTED_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if is_training and "target" not in df.columns:
        raise ValueError("Training data must contain 'target' column")

    for col, rule in FEATURE_RANGES.items():
        series = df[col].dropna()
        if rule["type"] == "numeric":
            if (series < rule["min"]).any() or (series > rule["max"]).any():
                out_of_bounds = series[(series < rule["min"]) | (series > rule["max"])]
                raise ValueError(f"Column '{col}' has values outside [{rule['min']}, {rule['max']}]: {out_of_bounds.values[:5]}")
        elif rule["type"] == "categorical":
            invalid = series[~series.isin(rule["allowed"])]
            if not invalid.empty:
                raise ValueError(f"Column '{col}' has invalid categories (allowed {rule['allowed']}): {invalid.values[:5]}")

    if is_training:
        target_series = df["target"].dropna()
        if not target_series.isin([0, 1]).all():
            raise ValueError(f"Target column must be binary (0 or 1), found: {target_series.unique()}")

    return True
