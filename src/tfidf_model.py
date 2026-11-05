"""TF-IDF + classifier adapter that follows the shared Predictor contract."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.common import ID_TO_LABEL, Prediction

MODEL_FILE = "model.joblib"


def _read_name(model_dir: Path) -> str:
    """Name the model from metadata.json, falling back to the folder name."""
    try:
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        return f"tfidf-{metadata['classifier']}"
    except (OSError, ValueError, KeyError):
        return f"tfidf-{model_dir.name}"


class TfidfPredictor:
    """Load a saved scikit-learn pipeline (TfidfVectorizer + classifier) and predict labels."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILE

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved TF-IDF model not found at {model_path}. "
                "Run `python -m src.train_tfidf` first."
            )

        self.pipeline = joblib.load(model_path)
        self.name = _read_name(model_dir)

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []

        probabilities = self.pipeline.predict_proba(texts)
        classes = [int(c) for c in self.pipeline.classes_]

        predictions = []
        for row in probabilities:
            best = int(row.argmax())
            # The score is the model's probability for the predicted label.
            predictions.append(Prediction(label=ID_TO_LABEL[classes[best]], score=float(row[best])))
        return predictions
