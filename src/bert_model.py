"""nlptown BERT sentiment model adapter (pretrained or fine-tuned)."""

import json
import logging
import os
import time
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
        logger.info("Loaded %s from %s (fine_tuned=%s)", self.name, model_path, self.fine_tuned)
        logger.debug("Model metadata: %s", self.metadata)

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
        return self.predict_timed(texts)[0]

    def predict_timed(self, texts: list[str]) -> tuple[list[Prediction], dict]:
        """Predict and also return how long each stage took, in milliseconds.

        Stages: `tokenize_ms` (the tokenizer run on its own, same text and limits),
        `inference_ms` (the whole pipeline call minus `tokenize_ms`, so the model forward pass and
        the pipeline's own post-processing) and `label_mapping_ms` (stars to negative/positive).
        `total_ms` is their sum. The pipeline tokenizes internally and cannot be split, so the
        tokenizer is timed separately; this costs about one extra tokenizer run. Timing never
        changes the predictions, and a timing failure is logged and shown as 0 for that stage.
        The timings are returned, not stored on the object, so concurrent requests cannot mix them.
        """
        if not texts:
            empty = {"tokenize_ms": 0.0, "inference_ms": 0.0, "label_mapping_ms": 0.0}
            return [], {**empty, "total_ms": 0.0}

        logger.debug("BERT predict: %d text(s), lengths %s", len(texts), [len(t) for t in texts])

        tokenize_ms = 0.0
        tokenizer = getattr(self.classifier, "tokenizer", None)
        if tokenizer is not None:
            try:
                started = time.perf_counter()
                tokenizer(texts, truncation=True, max_length=512)
                tokenize_ms = (time.perf_counter() - started) * 1000
            except Exception:
                logger.exception("Could not time tokenization; reporting 0 ms for that stage")

        started = time.perf_counter()
        raw_predictions = self.classifier(
            texts,
            batch_size=32,
            truncation=True,
            max_length=512,
        )
        pipeline_ms = (time.perf_counter() - started) * 1000
        inference_ms = max(pipeline_ms - tokenize_ms, 0.0)

        if len(raw_predictions) != len(texts):
            raise RuntimeError(
                f"Model returned {len(raw_predictions)} predictions "
                f"for {len(texts)} input texts."
            )

        logger.debug("BERT raw output: %s", raw_predictions)
        started = time.perf_counter()
        predictions = [
            Prediction(
                label=self.map_stars_to_label(result["label"]),
                score=float(result["score"]),
            )
            for result in raw_predictions
        ]
        label_mapping_ms = (time.perf_counter() - started) * 1000
        logger.debug("BERT mapped output: %s", predictions)

        stages = {
            "tokenize_ms": tokenize_ms,
            "inference_ms": inference_ms,
            "label_mapping_ms": label_mapping_ms,
            "total_ms": tokenize_ms + inference_ms + label_mapping_ms,
        }
        logger.debug("BERT stage times (ms): %s", {k: round(v, 3) for k, v in stages.items()})
        return predictions, stages
