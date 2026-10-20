"""
Prediction engine for inference using saved production model.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd

from heart.config import load_config
from heart.validation import validate_dataframe

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

class ModelPredictor:
    def __init__(self, model_path: Union[str, Path] = None, metadata_path: Union[str, Path] = None):
        config = load_config()
        if model_path is None:
            model_path = REPO_ROOT / config["models"]["model_path"]
        if metadata_path is None:
            metadata_path = REPO_ROOT / config["models"]["metadata_path"]

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")

        self.pipeline = joblib.load(self.model_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def predict_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([data])
        validate_dataframe(df, is_training=False)
        proba = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = int(proba >= 0.5)

        # Risk levels based on clinical risk stratification
        if proba < 0.35:
            risk_level = "Low Risk"
        elif proba < 0.70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "prediction": prediction,
            "disease_present": bool(prediction == 1),
            "confidence": round(proba if prediction == 1 else (1.0 - proba), 4),
            "probability_disease": round(proba, 4),
            "risk_level": risk_level
        }

    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_one(item) for item in data_list]
