"""nlptown BERT sentiment model adapter (pretrained or fine-tuned)."""

import json
import logging
import os
from pathlib import Path

from transformers import pipeline

from src.common import Prediction

logger = logging.getLogger(__name__)

PRETRAINED_DIR_NAME = "nlptown_sentiment"
FINETUNED_DIR_NAME = "nlptown_finetuned_v2"
MODEL_DIR_ENV = "REVIEWPULSE_MODEL_DIR"


def resolve_model_dir(models_dir: str | Path, override: str | Path | None = None) -> Path:
    """Choose the model folder to serve.

    Order: an explicit `override`, else the `REVIEWPULSE_MODEL_DIR` environment variable, else the
    fine-tuned model if its folder exists, else the pretrained model (with a warning).
    """
    chosen = override if override is not None else os.environ.get(MODEL_DIR_ENV)
    if chosen:
        return Path(chosen)

    models_dir = Path(models_dir)
    finetuned = models_dir / FINETUNED_DIR_NAME
    if finetuned.exists():
        return finetuned

    logger.warning(
        "Fine-tuned model not found at %s; serving the pretrained model instead.", finetuned
    )
    return models_dir / PRETRAINED_DIR_NAME


class BertPredictor:
    """Adapt the nlptown Hugging Face pipeline to the shared Predictor contract.

    The same adapter serves the pretrained and the fine-tuned model: both are saved in the same
    format and use the same 5-star output, mapped to negative (1-3 stars) and positive (4-5).
    `metadata.json` in the model folder (if present) says whether the model is fine-tuned.
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found at {model_path}. Run `python -m src.train_bert` for the "
                "pretrained model, or unzip the fine-tuned model into models/nlptown_finetuned_v2/."
            )

        self.metadata = self._read_metadata(model_path)
        self.fine_tuned = bool(self.metadata.get("fine_tuned", False))
        self.name = "nlptown-bert-finetuned" if self.fine_tuned else "nlptown-bert-multilingual"

        self.classifier = pipeline(
            "sentiment-analysis",
            model=str(model_path),
            tokenizer=str(model_path),
            truncation=True,
            max_length=512,
        )

    @staticmethod
    def _read_metadata(model_path: Path) -> dict:
        """Read `metadata.json` if it exists; a missing or unreadable file means no metadata."""
        try:
            data = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def map_stars_to_label(star_label: str) -> str:
        """Map 1–3 stars to negative and 4–5 stars to positive."""
        mapping = {
            "1 star": "negative",
            "2 stars": "negative",
            "3 stars": "negative",
            "4 stars": "positive",
            "5 stars": "positive",
        }

        try:
            return mapping[star_label]
        except KeyError:
            raise ValueError(f"Unexpected nlptown label: {star_label!r}") from None

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []

        raw_predictions = self.classifier(
            texts,
            batch_size=32,
            truncation=True,
            max_length=512,
        )

        if len(raw_predictions) != len(texts):
            raise RuntimeError(
                f"Model returned {len(raw_predictions)} predictions "
                f"for {len(texts)} input texts."
            )

        return [
            Prediction(
                label=self.map_stars_to_label(result["label"]),
                score=float(result["score"]),
            )
            for result in raw_predictions
        ]
