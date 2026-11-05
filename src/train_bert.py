"""Save the selected pretrained BERT model, then evaluate it once on the held-out test set.

No fine-tuning happens here. The nlptown checkpoint was selected on validation macro-F1
(see notebooks/bert_candidate_selection.ipynb and docs/mlflow_bert_candidates.csv).
This script downloads it, saves the artifact with a metadata file, and runs the test
evaluation from src/evaluate_bert.py.

Run:  python -m src.train_bert
"""

import json

import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import MLFLOW_BERT_EXPERIMENT, MODELS_DIR
from src.evaluate_bert import main as evaluate_on_test

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
SAVE_DIR = MODELS_DIR / "nlptown_sentiment"


def main():
    print(f"Downloading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    print(f"Saving artifact to {SAVE_DIR}...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    metadata = {
        "model_name": MODEL_NAME,
        "label_mapping": "1-3 stars -> negative, 4-5 stars -> positive",
        "max_token_length": 512,
        "fine_tuned": False,
    }
    (SAVE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    mlflow.set_experiment(MLFLOW_BERT_EXPERIMENT)
    with mlflow.start_run(run_name="save_best_pretrained"):
        mlflow.log_params(metadata)
        mlflow.log_artifact(str(SAVE_DIR / "metadata.json"))

    print(f"Model stored in {SAVE_DIR}/ (kept out of Git). Evaluating on the held-out test set...")
    evaluate_on_test()


if __name__ == "__main__":
    main()
