"""nlptown BERT sentiment model adapter."""

from pathlib import Path

from transformers import pipeline

from src.common import Prediction


class BertPredictor:
    """Adapt the nlptown Hugging Face pipeline to the shared Predictor contract."""

    name = "nlptown-bert-multilingual"

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found at {model_path}. " "Run `python -m src.train_bert` first."
            )

        self.classifier = pipeline(
            "sentiment-analysis",
            model=str(model_path),
            tokenizer=str(model_path),
            truncation=True,
            max_length=512,
        )

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
