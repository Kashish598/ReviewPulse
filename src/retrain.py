"""Retrain candidate models on the original training data plus the labelled requests (M4).

The loop this belongs to: users send reviews to /predict, the request store saves them, the LLM
labeller labels them in the background (src/labeler.py), and this script trains new candidate
models on the original 10,000 reviews plus those labelled requests.

What it does
- Reads the labelled requests from the database (DATABASE_URL) or from a CSV (--new-data), so it
  can be tried and tested without real traffic.
- Training set: the original train split (human labels) plus the labelled requests whose split is
  "train" (LLM labels).
- Validation: the original validation split (human labels, the number to choose on) and, as extra
  validation, the labelled requests whose split is "validation" (LLM labels, never trained on).
- Candidates are TF-IDF variants that train on a CPU in seconds (BERT candidates are trained on
  Kaggle by hand). Every candidate is logged to MLflow ("reviewpulse-retrain") with its parameters,
  metrics and the saved model, so the best one can be registered later.
- One extra run, "baseline_original_only", trains the default candidate on the original data alone,
  so the effect of the new data can be read off the same table.
- Each candidate is saved in models/candidates/<name>/ (model.joblib and metadata.json) and
  models/candidates/summary.json lists them, for the promotion step (M5).

What it does NOT do
- It never promotes anything and never scores the test split. The test split is read only to
  exclude its texts from the new data, so a test review can never end up in training.
- It does not touch the model the API serves.

Run:  python -m src.retrain                          (reads labelled requests from DATABASE_URL)
      python -m src.retrain --new-data requests.csv  (columns: text, llm_label, optional split)
      python -m src.retrain --no-mlflow --min-new 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.common import (
    DATASET_REVISION,
    LABEL_TO_ID,
    MODELS_DIR,
    SEED,
    compute_metrics,
    load_split,
    to_label_name,
)
from src.tfidf_model import MODEL_FILE
from src.train_tfidf import CLASSIFIER_PARAMS, TFIDF_PARAMS

logger = logging.getLogger("reviewpulse.retrain")

MLFLOW_RETRAIN_EXPERIMENT = "reviewpulse-retrain"
CANDIDATES_DIR = MODELS_DIR / "candidates"
SUMMARY_FILE = "summary.json"
BASELINE_NAME = "baseline_original_only"

DEFAULT_MIN_NEW = 50  # fewer labelled requests than this is too little to learn from
MIN_NEW_VALIDATION = 10  # fewer than this and the extra validation number is not reported


class RetrainError(Exception):
    """Something the person running the script can fix (no data, bad CSV, too few rows)."""


# ------------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    """One model to train: a TF-IDF setting plus a classifier and its settings."""

    name: str
    classifier: str  # "linear_svc" or "multinomial_nb"
    classifier_params: dict
    tfidf_overrides: dict = field(default_factory=dict)

    @property
    def tfidf_params(self) -> dict:
        return {**TFIDF_PARAMS, **self.tfidf_overrides}


UNIGRAMS = {"ngram_range": (1, 1)}

CANDIDATES: tuple[Candidate, ...] = (
    # The same settings train_tfidf.py uses, so this one is the like-for-like retrain.
    Candidate("svc_bigram_c0.5", "linear_svc", {"C": 0.5, "calibration_cv": 3}),
    Candidate("svc_bigram_c1", "linear_svc", {"C": 1.0, "calibration_cv": 3}),
    Candidate("svc_unigram_c0.5", "linear_svc", {"C": 0.5, "calibration_cv": 3}, UNIGRAMS),
    Candidate("nb_bigram", "multinomial_nb", dict(CLASSIFIER_PARAMS["multinomial_nb"])),
    Candidate("nb_unigram", "multinomial_nb", dict(CLASSIFIER_PARAMS["multinomial_nb"]), UNIGRAMS),
)
BASELINE_CANDIDATE = CANDIDATES[0]


def build_pipeline(candidate: Candidate) -> Pipeline:
    """An untrained TF-IDF + classifier pipeline. Every candidate supports predict_proba."""
    params = candidate.classifier_params
    if candidate.classifier == "linear_svc":
        classifier = CalibratedClassifierCV(
            LinearSVC(C=params["C"], random_state=SEED), cv=params["calibration_cv"]
        )
    elif candidate.classifier == "multinomial_nb":
        classifier = MultinomialNB(alpha=params["alpha"])
    else:
        raise ValueError(f"Unknown classifier {candidate.classifier!r}")
    return Pipeline([("tfidf", TfidfVectorizer(**candidate.tfidf_params)), ("clf", classifier)])


def pick_candidates(names: Sequence[str] | None) -> list[Candidate]:
    if not names:
        return list(CANDIDATES)
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise RetrainError(f"Unknown candidate(s) {unknown}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


# ------------------------------------------------------------------------------- new data


@dataclass
class NewData:
    train: pd.DataFrame  # columns review_text, label (0 or 1)
    validation: pd.DataFrame
    report: dict


def rows_from_csv(path: str | Path) -> list[dict]:
    """Read labelled requests from a CSV with columns text, llm_label and optionally split."""
    path = Path(path)
    if not path.is_file():
        raise RetrainError(f"{path} not found")
    frame = pd.read_csv(path)
    label_column = "llm_label" if "llm_label" in frame.columns else "label"
    if "text" not in frame.columns or label_column not in frame.columns:
        raise RetrainError(
            f"{path} needs the columns 'text' and 'llm_label' (and optionally 'split')"
        )
    rows = []
    for _, record in frame.iterrows():
        rows.append(
            {
                "text": record["text"],
                "llm_label": record[label_column],
                "split": record["split"] if "split" in frame.columns else None,
            }
        )
    return rows


def rows_from_store() -> list[dict]:
    """Read the labelled requests from the database named by DATABASE_URL."""
    if not os.environ.get("DATABASE_URL"):
        raise RetrainError(
            "DATABASE_URL is not set, so there is no request database to read. Set it, or pass "
            "--new-data requests.csv."
        )
    from src import feedback_store

    store = feedback_store.get_store()
    if store is None:
        raise RetrainError("The request store could not start, see the log above.")
    return store.labelled_rows()


def _clean(value) -> str:
    return (
        "" if value is None or (isinstance(value, float) and value != value) else str(value).strip()
    )


def prepare_new_data(rows: Sequence[dict], known_texts: set[str]) -> NewData:
    """Turn labelled requests into training and validation frames, and say what was dropped.

    A request is dropped when its text is empty, its label is not exactly negative or positive
    (or 0/1), it repeats an earlier request, or it already appears in the original train,
    validation or test split (the last one is what keeps test reviews out of training).
    """
    from src.feedback_store import assign_split, text_hash

    report = {
        "received": len(rows),
        "dropped_empty": 0,
        "dropped_bad_label": 0,
        "dropped_duplicate": 0,
        "dropped_in_original_data": 0,
        "used_train": 0,
        "used_validation": 0,
    }
    seen: set[str] = set()
    train, validation = [], []
    for row in rows:
        text = _clean(row.get("text"))
        if not text:
            report["dropped_empty"] += 1
            continue
        try:
            label = LABEL_TO_ID[to_label_name(row.get("llm_label"))]
        except (ValueError, KeyError):
            report["dropped_bad_label"] += 1
            continue
        if text in seen:
            report["dropped_duplicate"] += 1
            continue
        seen.add(text)
        if text in known_texts:
            report["dropped_in_original_data"] += 1
            continue
        split = _clean(row.get("split"))
        if split not in {"train", "validation"}:
            split = assign_split(text_hash(text))
        (validation if split == "validation" else train).append((text, label))

    report["used_train"] = len(train)
    report["used_validation"] = len(validation)
    columns = ["review_text", "label"]
    return NewData(
        train=pd.DataFrame(train, columns=columns),
        validation=pd.DataFrame(validation, columns=columns),
        report=report,
    )


def original_texts(*frames: pd.DataFrame) -> set[str]:
    return {str(text).strip() for frame in frames for text in frame["review_text"]}


# ------------------------------------------------------------------------------- training


def score(pipeline: Pipeline, frame: pd.DataFrame) -> dict | None:
    """Accuracy and macro-F1 on a frame, or None when there is nothing to score."""
    if frame is None or len(frame) == 0:
        return None
    predicted = [int(p) for p in pipeline.predict(frame["review_text"])]
    return compute_metrics(frame["label"].tolist(), predicted)


@dataclass
class Result:
    name: str
    val_accuracy: float
    val_macro_f1: float
    new_val_accuracy: float | None
    new_val_macro_f1: float | None
    fit_seconds: float
    run_id: str | None = None
    path: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _metrics_dict(result: Result) -> dict:
    return {
        "val_accuracy": result.val_accuracy,
        "val_macro_f1": result.val_macro_f1,
        "new_val_accuracy": result.new_val_accuracy,
        "new_val_macro_f1": result.new_val_macro_f1,
        "fit_seconds": result.fit_seconds,
    }


def fit_and_score(
    candidate: Candidate,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    new_validation: pd.DataFrame,
) -> tuple[Pipeline, Result]:
    pipeline = build_pipeline(candidate)
    started = time.perf_counter()
    pipeline.fit(train["review_text"], train["label"])
    fit_seconds = time.perf_counter() - started

    main = score(pipeline, validation)
    extra = score(pipeline, new_validation) if len(new_validation) >= MIN_NEW_VALIDATION else None
    result = Result(
        name=candidate.name,
        val_accuracy=main["accuracy"],
        val_macro_f1=main["macro_f1"],
        new_val_accuracy=extra["accuracy"] if extra else None,
        new_val_macro_f1=extra["macro_f1"] if extra else None,
        fit_seconds=round(fit_seconds, 3),
    )
    return pipeline, result


def save_candidate(
    pipeline: Pipeline, candidate: Candidate, result: Result, context: dict, out_dir: Path
) -> Path:
    """Save a candidate where TfidfPredictor can load it: model.joblib plus metadata.json."""
    target = out_dir / candidate.name
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target / MODEL_FILE)
    metadata = {
        "classifier": candidate.name,  # TfidfPredictor names the model "tfidf-<this>"
        "classifier_type": candidate.classifier,
        "classifier_params": candidate.classifier_params,
        "tfidf_params": {
            k: list(v) if isinstance(v, tuple) else v for k, v in candidate.tfidf_params.items()
        },
        "artifact_name": candidate.name,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": "retrain_candidate",
        "validation_accuracy": result.val_accuracy,
        "validation_macro_f1": result.val_macro_f1,
        "new_validation_accuracy": result.new_val_accuracy,
        "new_validation_macro_f1": result.new_val_macro_f1,
        **context,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    model_dir: Path | None = None,
    dicts: dict | None = None,
) -> str:
    """Log one run to the retraining experiment and return its run id."""
    import mlflow  # imported here so the rest of the module works without MLflow

    mlflow.set_experiment(MLFLOW_RETRAIN_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
        mlflow.set_tags(tags)
        if model_dir is not None:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
        for file_name, content in (dicts or {}).items():
            mlflow.log_dict(content, file_name)
        return run.info.run_id


def run_retraining(
    original_train: pd.DataFrame,
    original_validation: pd.DataFrame,
    new: NewData,
    candidates: Sequence[Candidate],
    out_dir: Path = CANDIDATES_DIR,
    *,
    track: bool = True,
    source: str = "",
) -> dict:
    """Train the baseline and every candidate, log them, save them and write the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    combined = pd.concat([original_train[["review_text", "label"]], new.train], ignore_index=True)
    context = {
        "retrain_batch": batch,
        "new_data_source": source,
        "n_train_original": len(original_train),
        "n_new_train": len(new.train),
        "n_new_validation": len(new.validation),
    }
    base_params = {
        **context,
        "validation_size": len(original_validation),
        "seed": SEED,
        "dataset_revision": DATASET_REVISION,
    }
    base_tags = {"model_family": "tfidf", "sklearn_version": sklearn.__version__}

    def log(candidate: Candidate, result: Result, stage: str, model_dir: Path | None) -> str | None:
        if not track:
            return None
        return log_run(
            run_name=f"{stage}_{candidate.name}" if stage != "baseline" else BASELINE_NAME,
            params={
                **base_params,
                "candidate": candidate.name,
                "classifier": candidate.classifier,
                "classifier_params": json.dumps(candidate.classifier_params),
                "tfidf": json.dumps(candidate.tfidf_params),
                "n_train": len(original_train) if stage == "baseline" else len(combined),
            },
            metrics=_metrics_dict(result),
            tags={**base_tags, "stage": stage, "retrain_batch": batch},
            model_dir=model_dir,
            dicts={"validation_metrics.json": _metrics_dict(result)},
        )

    logger.info(
        "Training on %d original + %d new labelled requests; validating on %d original (+ %d new)",
        len(original_train),
        len(new.train),
        len(original_validation),
        len(new.validation),
    )

    # The baseline: same settings as the default candidate, original data only.
    logger.info("Fitting %s ...", BASELINE_NAME)
    _, baseline = fit_and_score(
        BASELINE_CANDIDATE,
        original_train[["review_text", "label"]],
        original_validation,
        new.validation,
    )
    baseline.name = BASELINE_NAME
    baseline.run_id = log(BASELINE_CANDIDATE, baseline, "baseline", None)

    results: list[Result] = []
    for candidate in candidates:
        logger.info("Fitting %s ...", candidate.name)
        pipeline, result = fit_and_score(candidate, combined, original_validation, new.validation)
        result.path = str(save_candidate(pipeline, candidate, result, context, out_dir))
        result.run_id = log(candidate, result, "candidate", Path(result.path))
        results.append(result)
        logger.info(
            "  validation accuracy %.4f, macro-F1 %.4f%s",
            result.val_accuracy,
            result.val_macro_f1,
            ""
            if result.new_val_accuracy is None
            else f", new-request validation accuracy {result.new_val_accuracy:.4f}",
        )

    best = max(results, key=lambda r: (r.val_macro_f1, r.val_accuracy))
    summary = {
        **context,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "new_data": new.report,
        "tracked_in_mlflow": track,
        "baseline": baseline.as_dict(),
        "candidates": [result.as_dict() for result in results],
        "best_on_validation": best.name,
        "note": "Chosen on validation only. The test split has not been used. Promotion is M5.",
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_table(summary: dict) -> str:
    rows = [summary["baseline"], *summary["candidates"]]
    lines = [
        f"{'model':<22}{'val acc':>9}{'val F1':>9}{'new-val acc':>13}{'fit s':>8}",
    ]
    for row in rows:
        extra = "-" if row["new_val_accuracy"] is None else f"{row['new_val_accuracy']:.4f}"
        lines.append(
            f"{row['name']:<22}{row['val_accuracy']:>9.4f}{row['val_macro_f1']:>9.4f}"
            f"{extra:>13}{row['fit_seconds']:>8.1f}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train candidate models on the original data plus labelled requests."
    )
    parser.add_argument("--new-data", type=Path, help="CSV with text, llm_label, optional split")
    parser.add_argument(
        "--min-new",
        type=int,
        default=DEFAULT_MIN_NEW,
        help=f"stop if fewer usable labelled requests than this (default {DEFAULT_MIN_NEW})",
    )
    parser.add_argument("--candidates", help="comma-separated names (default: all)")
    parser.add_argument("--out-dir", type=Path, default=CANDIDATES_DIR)
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        names = [n.strip() for n in args.candidates.split(",")] if args.candidates else None
        candidates = pick_candidates(names)

        if args.new_data:
            rows, source = rows_from_csv(args.new_data), f"csv:{args.new_data.name}"
        else:
            rows, source = rows_from_store(), "database"

        train = load_split("train")
        validation = load_split("validation")
        test = load_split("test")  # only to keep its texts out of the new data
        new = prepare_new_data(rows, original_texts(train, validation, test))
        del test

        logger.info("Labelled requests (%s): %s", source, json.dumps(new.report))
        usable = new.report["used_train"] + new.report["used_validation"]
        if usable < args.min_new:
            raise RetrainError(
                f"Only {usable} usable labelled requests, need at least {args.min_new} "
                "(--min-new lowers it). Send more traffic and let the labeller run, or use "
                "--new-data."
            )

        if not args.no_mlflow:
            import mlflow

            logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
            if not os.environ.get("MLFLOW_TRACKING_URI"):
                logger.warning(
                    "MLFLOW_TRACKING_URI is not set. See the README (MLflow Tracking) for the "
                    "sqlite:///reviewpulse_mlflow.db setting that 'mlflow ui' needs."
                )

        summary = run_retraining(
            train,
            validation,
            new,
            candidates,
            args.out_dir,
            track=not args.no_mlflow,
            source=source,
        )
    except (RetrainError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 2

    logger.info("\n%s", format_table(summary))
    logger.info(
        "Best on validation: %s (no promotion, no test score).", summary["best_on_validation"]
    )
    logger.info("Candidates saved in %s, summary in %s", args.out_dir, args.out_dir / SUMMARY_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
