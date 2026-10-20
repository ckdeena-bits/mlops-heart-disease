from pathlib import Path

from heart.config import load_config
from heart.predict import ModelPredictor


def test_model_artifact_exists():
    config = load_config()
    model_path = Path(config["models"]["model_path"])
    metadata_path = Path(config["models"]["metadata_path"])
    assert model_path.exists()
    assert metadata_path.exists()

def test_model_predictor_single_inference():
    predictor = ModelPredictor()
    sample = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }
    result = predictor.predict_one(sample)
    assert "prediction" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk_level"] in ("Low Risk", "Moderate Risk", "High Risk")

def test_model_quality_gate_auc():
    predictor = ModelPredictor()
    roc_auc = predictor.metadata.get("metrics", {}).get("roc_auc", 0.0)
    assert roc_auc >= 0.80, f"Model ROC-AUC ({roc_auc}) is below production threshold of 0.80"
