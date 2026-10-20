"""
Feature engineering pipelines and column transformers.
"""
from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import load_config


def build_preprocessor() -> Tuple[ColumnTransformer, List[str], List[str]]:
    config = load_config()
    numeric_features = config["data"]["numeric_features"]
    categorical_features = config["data"]["categorical_features"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_features, categorical_features
