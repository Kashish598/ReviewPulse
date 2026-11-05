"""Evaluate the selected BERT model on the held-out test set."""

import json

import pandas as pd

from src.bert_model import BertPredictor
from src.common import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    compute_metrics,
    load_split,
)


def main():
    # Load the complete held-out test split.
    test_df = load_split("test")

    print(f"Loaded {len(test_df)} held-out test reviews.")

    # Load the saved pretrained model.
    model = BertPredictor(MODELS_DIR / "nlptown_sentiment")

    # Generate predictions.
    predictions = model.predict(test_df["review_text"].tolist())
    predicted_labels = [prediction.label for prediction in predictions]

    # Convert numerical ground-truth labels into readable labels.
    label_map = {0: "negative", 1: "positive"}

    results_df = pd.DataFrame(
        {
            "row_id": test_df["row_id"],
            "label_true": test_df["label"].map(label_map),
            "pred": predicted_labels,
            "score": [prediction.score for prediction in predictions],
        }
    )

    # Calculate evaluation metrics.
    metrics = compute_metrics(
        test_df["label"].tolist(),
        predicted_labels,
    )

    # Save predictions and metrics.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = PREDICTIONS_DIR / "bert_test.csv"
    metrics_path = PREDICTIONS_DIR / "bert_test_metrics.json"

    results_df.to_csv(predictions_path, index=False)

    metrics_to_save = {
        key: value.tolist() if hasattr(value, "tolist") else value for key, value in metrics.items()
    }

    metrics_to_save["model"] = model.name
    metrics_to_save["split"] = "test"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_to_save, file, indent=4)

    print("\nHeld-out test results")
    print("---------------------")
    print(json.dumps(metrics_to_save, indent=4))

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
