"""Shared contracts for ReviewPulse. Everyone imports from here.

Do not redefine labels, metrics, paths or file schemas in other modules.
Any change to this file needs a PR that all three members review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
MODELS_DIR = ROOT / "models"

SEED = 42
DATASET_NAME = "fancyzhx/amazon_polarity"
# Hugging Face commit hash of the dataset. Fill it in from data_report.json after the first
# run so every machine builds the splits from exactly the same data.
DATASET_REVISION: str | None = "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289"

ID_TO_LABEL = {0: "negative", 1: "positive"}
LABEL_TO_ID = {name: idx for idx, name in ID_TO_LABEL.items()}
LABELS = ["negative", "positive"]

SPLITS = ("train", "validation", "test")
DATA_COLUMNS = ["row_id", "review_text", "label"]
PREDICTION_COLUMNS = ["row_id", "label_true", "pred", "score"]
TEST_ORDER_FILE = "test_order.csv"

MLFLOW_BERT_EXPERIMENT = "reviewpulse-bert"
MLFLOW_LLM_EXPERIMENT = "reviewpulse-llm"


# ---------------------------------------------------------------- labels


def to_label_name(value) -> str:
    """Return "negative" or "positive" for 0/1 or for those exact strings.

    Anything else raises ValueError. This is deliberate: an LLM output that is not
    exactly one of the two labels must be flagged by the caller, not guessed.
    """
    if isinstance(value, bool):
        pass  # True and False are not labels
    elif isinstance(value, str):
        if value in LABEL_TO_ID:
            return value
    else:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            idx = None
        if idx in ID_TO_LABEL and idx == value:
            return ID_TO_LABEL[idx]
    raise ValueError(f"Invalid label {value!r}. Allowed: 0, 1, 'negative', 'positive'.")


# --------------------------------------------------------------- metrics


def compute_metrics(y_true: Sequence, y_pred: Sequence) -> dict:
    """Macro-F1 (primary), accuracy and confusion matrix (secondary).

    Inputs can be 0/1 or "negative"/"positive". Confusion matrix rows are the true
    label, columns the predicted label, in the order [negative, positive].
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} true vs {len(y_pred)} predicted.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty list.")
    true_ids = [LABEL_TO_ID[to_label_name(v)] for v in y_true]
    pred_ids = [LABEL_TO_ID[to_label_name(v)] for v in y_pred]
    return {
        "macro_f1": float(
            f1_score(true_ids, pred_ids, average="macro", labels=[0, 1], zero_division=0)
        ),
        "accuracy": float(accuracy_score(true_ids, pred_ids)),
        "confusion_matrix": confusion_matrix(true_ids, pred_ids, labels=[0, 1]).tolist(),
        "n": len(true_ids),
    }


# ------------------------------------------------------------------ data


def _split_dir(sample: bool) -> Path:
    return SAMPLE_DIR if sample else PROCESSED_DIR


def load_split(name: str, sample: bool = False) -> pd.DataFrame:
    """Load train, validation or test with columns row_id, review_text, label.

    sample=True reads the small committed fixture in data/sample/ instead of the
    full splits in data/processed/. Use the fixture for tests and development.
    """
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}. Allowed: {SPLITS}.")
    path = _split_dir(sample) / f"{name}.csv"
    if not path.exists():
        hint = "" if sample else " Run `python -m src.prepare_data` first."
        raise FileNotFoundError(f"{path} not found.{hint}")
    df = pd.read_csv(path)
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["label"] = df["label"].astype(int)
    return df[DATA_COLUMNS]


def load_test_subset(n: int, sample: bool = False) -> pd.DataFrame:
    """First n test rows in the fixed shuffled order from test_order.csv.

    Use this when the full test set is too expensive (the LLM). BERT should be scored
    on the same subset for the comparison. Any n works without regenerating files,
    and a smaller n is always a prefix of a larger one.
    """
    test = load_split("test", sample=sample)
    order_path = _split_dir(sample) / TEST_ORDER_FILE
    if not order_path.exists():
        raise FileNotFoundError(f"{order_path} not found. Run `python -m src.prepare_data`.")
    order = pd.read_csv(order_path)["row_id"].tolist()
    if not 0 < n <= len(order):
        raise ValueError(f"n must be between 1 and {len(order)}, got {n}.")
    subset = test.set_index("row_id").loc[order[:n]].reset_index()
    return subset[DATA_COLUMNS]


# ----------------------------------------------------------- predictions


def write_predictions(
    path: Path | str,
    row_ids: Sequence,
    y_true: Sequence,
    preds: Sequence,
    scores: Sequence[float | None] | None = None,
) -> Path:
    """Write the predictions file that BERT and LLM both produce.

    Columns: row_id, label_true, pred, score. Labels are written as names.
    score is empty when the model has no score.
    """
    n = len(row_ids)
    if not (len(y_true) == len(preds) == n):
        raise ValueError("row_ids, y_true and preds must have the same length.")
    if scores is not None and len(scores) != n:
        raise ValueError("scores must have the same length as row_ids.")
    df = pd.DataFrame(
        {
            "row_id": list(row_ids),
            "label_true": [to_label_name(v) for v in y_true],
            "pred": [to_label_name(v) for v in preds],
            "score": list(scores) if scores is not None else [None] * n,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_predictions(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[PREDICTION_COLUMNS]


# ------------------------------------------------------------- predictor


@dataclass(frozen=True)
class Prediction:
    label: str  # exactly "negative" or "positive"
    score: float | None = None  # confidence for `label` if the model provides one

    def __post_init__(self):
        to_label_name(self.label)


@runtime_checkable
class Predictor(Protocol):
    """Interface every model adapter implements (BERT adapter, mock, and so on).

    predict takes a list of review_text strings and returns one Prediction per input,
    in the same order. Scores are not promised to be calibrated probabilities.
    """

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class MockPredictor:
    """Keyword-based stand-in so the API and tests work before the BERT model exists."""

    name = "mock-keyword"

    POSITIVE = ("good", "great", "love", "excellent", "perfect", "best", "amazing")
    NEGATIVE = ("bad", "terrible", "hate", "worst", "broken", "awful", "waste")

    def predict(self, texts: list[str]) -> list[Prediction]:
        out = []
        for text in texts:
            lowered = text.lower()
            pos = sum(word in lowered for word in self.POSITIVE)
            neg = sum(word in lowered for word in self.NEGATIVE)
            label = "positive" if pos >= neg else "negative"
            out.append(Prediction(label=label, score=0.5))
        return out
