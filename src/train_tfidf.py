"""Train TF-IDF + classifier baselines, pick the best on validation, save it, test it once.

All candidates share one TF-IDF representation (word 1-2 grams): a calibrated linear SVM,
multinomial naive Bayes and a random forest. Models are fitted on the train split only.
The classifier is chosen on the validation split by macro-F1 (accuracy breaks ties).
The test split is scored once, for the chosen model only. No test result is used to
choose anything.

Everything is tracked in the MLflow experiment "reviewpulse-tfidf": one run per candidate
(validation metrics) and one run for the selected model (validation and test metrics,
the saved model and its metadata). Browse the runs with `mlflow ui`.

Run:  python -m src.train_tfidf
      python -m src.train_tfidf --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.common import (
    DATASET_REVISION,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SEED,
    compute_metrics,
    load_split,
    write_predictions,
)
from src.tfidf_model import MODEL_FILE, TfidfPredictor

MLFLOW_TFIDF_EXPERIMENT = "reviewpulse-tfidf"
TFIDF_MODEL_DIR = MODELS_DIR / "tfidf_v1"

TFIDF_PARAMS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
}

# Fixed classifier settings. They are logged to MLflow with every run.
CLASSIFIER_PARAMS = {
    "linear_svc": {"C": 0.5, "calibration_cv": 3},
    "multinomial_nb": {"alpha": 0.5},
    "random_forest": {"n_estimators": 100},
}


def build_candidates(tfidf_params: dict | None = None) -> dict[str, Pipeline]:
    """One untrained pipeline per candidate classifier. Every one supports predict_proba."""
    params = {**TFIDF_PARAMS, **(tfidf_params or {})}
    svc = CLASSIFIER_PARAMS["linear_svc"]
    nb = CLASSIFIER_PARAMS["multinomial_nb"]
    forest = CLASSIFIER_PARAMS["random_forest"]

    def make(classifier) -> Pipeline:
        return Pipeline([("tfidf", TfidfVectorizer(**params)), ("clf", classifier)])

    return {
        "linear_svc": make(
            CalibratedClassifierCV(
                LinearSVC(C=svc["C"], random_state=SEED), cv=svc["calibration_cv"]
            )
        ),
        "multinomial_nb": make(MultinomialNB(alpha=nb["alpha"])),
        "random_forest": make(
            RandomForestClassifier(
                n_estimators=forest["n_estimators"], n_jobs=-1, random_state=SEED
            )
        ),
    }


def log_to_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict | None = None,
    artifacts=(),
    dicts: dict | None = None,
) -> None:
    """Log one run to the TF-IDF experiment: params, metrics, tags, files and JSON dicts."""
    import mlflow  # imported here so the rest of the module works without MLflow

    mlflow.set_experiment(MLFLOW_TFIDF_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags or {})
        for path in artifacts:
            mlflow.log_artifact(str(path))
        for file_name, content in (dicts or {}).items():
            mlflow.log_dict(content, file_name)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the TF-IDF baselines.")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    train = load_split("train")
    validation = load_split("validation")
    print(f"Train: {len(train)} reviews. Validation: {len(validation)} reviews.")

    base_params = {
        "tfidf": json.dumps(TFIDF_PARAMS),
        "train_size": len(train),
        "validation_size": len(validation),
        "seed": SEED,
        "dataset_revision": DATASET_REVISION,
    }
    base_tags = {"model_family": "tfidf", "sklearn_version": sklearn.__version__}

    # 1. Fit every candidate on train and score it on validation.
    fitted = {}
    validation_metrics = {}
    for name, pipeline in build_candidates().items():
        print(f"Fitting {name}...")
        started = time.perf_counter()
        pipeline.fit(train["review_text"], train["label"])
        fit_seconds = time.perf_counter() - started

        predicted = [int(p) for p in pipeline.predict(validation["review_text"])]
        metrics = compute_metrics(validation["label"].tolist(), predicted)

        fitted[name] = pipeline
        validation_metrics[name] = metrics
        print(
            f"  validation accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}"
        )

        if not args.no_mlflow:
            log_to_mlflow(
                run_name=f"tfidf_{name}",
                params={
                    **base_params,
                    "classifier": name,
                    "classifier_params": json.dumps(CLASSIFIER_PARAMS[name]),
                },
                metrics={
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "fit_seconds": fit_seconds,
                },
                tags={**base_tags, "stage": "candidate"},
                dicts={"validation_metrics.json": metrics},
            )

    # 2. Choose on validation only.
    best = max(
        validation_metrics,
        key=lambda n: (validation_metrics[n]["macro_f1"], validation_metrics[n]["accuracy"]),
    )
    print(f"Selected on validation: {best}")

    # 3. Save the chosen model with its metadata.
    TFIDF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], TFIDF_MODEL_DIR / MODEL_FILE)
    metadata = {
        "classifier": best,
        "classifier_params": CLASSIFIER_PARAMS[best],
        "artifact_name": TFIDF_MODEL_DIR.name,
        "tfidf_params": {
            k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF_PARAMS.items()
        },
        "train_size": len(train),
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "validation_accuracy": validation_metrics[best]["accuracy"],
        "validation_macro_f1": validation_metrics[best]["macro_f1"],
    }
    metadata_path = TFIDF_MODEL_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 4. Score the held-out test split once, through the saved artifact.
    predictor = TfidfPredictor(TFIDF_MODEL_DIR)
    test = load_split("test")
    predictions = predictor.predict(test["review_text"].tolist())
    labels = [prediction.label for prediction in predictions]
    scores = [prediction.score for prediction in predictions]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = write_predictions(
        PREDICTIONS_DIR / "tfidf_test.csv",
        test["row_id"].tolist(),
        test["label"].tolist(),
        labels,
        scores,
    )
    test_metrics = compute_metrics(test["label"].tolist(), labels)
    test_metrics["model"] = predictor.name
    test_metrics["split"] = "test"
    metrics_path = PREDICTIONS_DIR / "tfidf_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=4), encoding="utf-8")

    metadata["test_accuracy"] = test_metrics["accuracy"]
    metadata["test_macro_f1"] = test_metrics["macro_f1"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 5. Log the selected model: validation and test metrics, model file and metadata.
    if not args.no_mlflow:
        log_to_mlflow(
            run_name=f"tfidf_selected_{best}",
            params={
                **base_params,
                "classifier": best,
                "classifier_params": json.dumps(CLASSIFIER_PARAMS[best]),
                "artifact_path": str(TFIDF_MODEL_DIR),
            },
            metrics={
                "val_accuracy": metadata["validation_accuracy"],
                "val_macro_f1": metadata["validation_macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            tags={**base_tags, "stage": "selected"},
            artifacts=[TFIDF_MODEL_DIR / MODEL_FILE, metadata_path, metrics_path],
            dicts={"validation_metrics.json": validation_metrics[best]},
        )

    print(f"\nHeld-out test results for {predictor.name}")
    print(f"  accuracy {test_metrics['accuracy']:.4f}, macro-F1 {test_metrics['macro_f1']:.4f}")
    print(f"  confusion matrix: {test_metrics['confusion_matrix']}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model stored in {TFIDF_MODEL_DIR}/ (kept out of Git).")
    if not args.no_mlflow:
        print(f"MLflow experiment: {MLFLOW_TFIDF_EXPERIMENT} (view with: mlflow ui)")


if __name__ == "__main__":
    main()
