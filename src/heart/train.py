"""
Training pipeline with Hyperparameter Tuning, Cross-Validation, MLflow tracking, and model export.
"""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from heart.config import load_config
from heart.data import load_clean_data
from heart.evaluate import evaluate_predictions, plot_confusion_matrix, plot_roc_curve
from heart.features import build_preprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def train_and_evaluate(online_download: bool = False) -> None:
    config = load_config()
    figures_dir = REPO_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir = REPO_ROOT / config["models"]["save_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data()
    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]].values

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv_folds = config["training"]["cv_folds"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Set up MLflow with sqlite backend (recommended for production tracking) or fallback
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlruns_dir = REPO_ROOT / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    db_path = mlruns_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(config["models"]["mlflow_experiment_name"])

    preprocessor, num_cols, cat_cols = build_preprocessor()

    candidates = {
        "LogisticRegression": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=random_state, max_iter=1000))
            ]),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__penalty": ["l2"]
            }
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=random_state))
            ]),
            "params": {
                "classifier__n_estimators": [50, 100, 150, 200],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 4, 6],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        }
    }

    best_overall_score = -1.0
    best_overall_model_name = ""
    best_overall_pipeline = None
    best_overall_metrics = {}

    for name, spec in candidates.items():
        with mlflow.start_run(run_name=name):
            print(f"\n================ Training & Tuning {name} ================")
            search = RandomizedSearchCV(
                estimator=spec["pipeline"],
                param_distributions=spec["params"],
                n_iter=8,
                scoring=config["training"]["scoring"],
                cv=cv,
                random_state=random_state,
                n_jobs=-1
            )
            search.fit(X_train, y_train)

            best_pipeline = search.best_estimator_
            best_cv_score = float(search.best_score_)
            print(f"Best CV {config['training']['scoring']}: {best_cv_score:.4f}")
            print(f"Best Hyperparameters: {search.best_params_}")

            # Predict on Holdout Test Set
            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate_predictions(y_test, y_pred, y_proba)
            metrics["cv_mean_roc_auc"] = best_cv_score
            for k, v in metrics.items():
                print(f"Test {k}: {v:.4f}")

            # Generate Plots
            cm_path = figures_dir / f"confusion_matrix_{name.lower()}.png"
            roc_path = figures_dir / f"roc_curve_{name.lower()}.png"
            plot_confusion_matrix(y_test, y_pred, cm_path, model_name=name)
            plot_roc_curve(y_test, y_proba, roc_path, model_name=name)

            # Log to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.log_artifact(str(roc_path), artifact_path="figures")
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
                input_example=X_train.iloc[:2]
            )

            if metrics["roc_auc"] > best_overall_score:
                best_overall_score = metrics["roc_auc"]
                best_overall_model_name = name
                best_overall_pipeline = best_pipeline
                best_overall_metrics = metrics

    print(f"\nChampion Model: {best_overall_model_name} with ROC-AUC {best_overall_score:.4f}")

    # Enforce quality gate
    quality_gate = config["training"]["quality_gate_auc"]
    if best_overall_score < quality_gate:
        raise ValueError(f"Model ROC-AUC ({best_overall_score:.4f}) failed quality gate ({quality_gate})")

    # Save champion model locally
    final_model_path = REPO_ROOT / config["models"]["model_path"]
    joblib.dump(best_overall_pipeline, final_model_path)
    print(f"Champion pipeline saved to {final_model_path}")

    # Save metadata
    metadata = {
        "model_name": best_overall_model_name,
        "metrics": best_overall_metrics,
        "input_features": list(X.columns),
        "target": config["data"]["target"],
        "random_state": random_state,
        "author": "Deenadayalan C K (2025AE05402)",
        "dataset": "UCI Heart Disease (Cleveland Clinic)"
    }
    metadata_path = REPO_ROOT / config["models"]["metadata_path"]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    train_and_evaluate()
