"""Check that the model the API serves is the selected fine-tuned model.

Scores the 600 held-out test reviews with the model folder the API would load and compares
the result with `data/predictions/bert_finetuned_v2_test.csv` (made on Kaggle). Nothing is
written. Run it from the repository root:

    python scripts/verify_served_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bert_model import BertPredictor, resolve_model_dir  # noqa: E402
from src.common import MODELS_DIR, PREDICTIONS_DIR, load_split  # noqa: E402

EXPECTED_FILE = PREDICTIONS_DIR / "bert_finetuned_v2_test.csv"
MIN_AGREEMENT = 0.99  # fp16 on Kaggle vs fp32 here can flip a few borderline reviews


def main() -> int:
    model_dir = resolve_model_dir(MODELS_DIR)
    predictor = BertPredictor(model_dir)
    print(f"Model folder : {model_dir}")
    print(f"Model name   : {predictor.name} (fine_tuned={predictor.fine_tuned})")
    if not predictor.fine_tuned:
        print("FAIL: this is not the fine-tuned model (no metadata.json with fine_tuned=true).")
        return 1

    test = load_split("test")
    served = pd.DataFrame(
        {
            "row_id": test["row_id"],
            "pred": [p.label for p in predictor.predict(test["review_text"].tolist())],
        }
    )
    expected = pd.read_csv(EXPECTED_FILE)[["row_id", "label_true", "pred"]]
    merged = expected.merge(served, on="row_id", suffixes=("_kaggle", "_here"))
    if len(merged) != len(test):
        print(f"FAIL: {len(merged)} of {len(test)} test rows matched by row_id.")
        return 1

    agreement = (merged["pred_kaggle"] == merged["pred_here"]).mean()
    accuracy = (merged["pred_here"] == merged["label_true"]).mean()
    print(f"Agreement with the Kaggle predictions: {agreement:.4f}")
    print(f"Test accuracy of the served model    : {accuracy:.4f} (selected run: 0.9517)")

    if agreement < MIN_AGREEMENT:
        print("FAIL: the served model does not reproduce the Kaggle predictions.")
        return 1
    print("OK: the API serves the selected fine-tuned model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
