"""
Data acquisition, cleaning, preprocessing and loading.
"""
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from heart.config import load_config
from heart.validation import validate_dataframe

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def download_raw_data(force_online: bool = False) -> Path:
    config = load_config()
    raw_path = REPO_ROOT / config["data"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force_online:
        print(f"Using committed raw data at {raw_path}")
        return raw_path

    url = config["data"]["raw_url"]
    print(f"Fetching from {url}...")
    try:
        urllib.request.urlretrieve(url, raw_path)
    except Exception as e:
        print(f"Online download failed: {e}. Falling back to local copy.")
        if not raw_path.exists():
            raise RuntimeError(f"No local data and online download failed: {e}")
    return raw_path

def clean_data(raw_path: Optional[Path] = None, output_path: Optional[Path] = None) -> pd.DataFrame:
    config = load_config()
    if raw_path is None:
        raw_path = REPO_ROOT / config["data"]["raw_path"]
    if output_path is None:
        output_path = REPO_ROOT / config["data"]["clean_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = config["data"]["feature_names"]

    # Raw data uses '?' for missing values
    df = pd.read_csv(raw_path, names=names, na_values="?", skipinitialspace=True)

    # Impute missing values for ca and thal (the only 2 attributes with ? in Cleveland dataset)
    # ca has 4 missing, thal has 2 missing.
    # We use median/mode imputation for data cleaning so the processed dataset is complete and clean.
    if df["ca"].isnull().any():
        df["ca"] = df["ca"].fillna(df["ca"].median())
    if df["thal"].isnull().any():
        df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

    # Convert num (0 to 4) to binary target: 0 = absence (healthy), 1 = presence (heart disease risk)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Cast integer columns to int
    int_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "slope", "ca", "thal", "target"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    # Validate against in-house contract
    validate_dataframe(df, is_training=True)

    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path} ({len(df)} rows, {df.shape[1]} columns)")
    return df

def load_clean_data(clean_path: Optional[Path] = None) -> pd.DataFrame:
    config = load_config()
    if clean_path is None:
        clean_path = REPO_ROOT / config["data"]["clean_path"]
    if not clean_path.exists():
        print("Clean data not found, generating from raw data...")
        return clean_data(output_path=clean_path)
    df = pd.read_csv(clean_path)
    validate_dataframe(df, is_training=True)
    return df

if __name__ == "__main__":
    download_raw_data()
    clean_data()
