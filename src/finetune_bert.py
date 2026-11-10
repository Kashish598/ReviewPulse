"""Fine-tune the nlptown BERT on the 10,000 training reviews and track everything in MLflow.

Why this design
---------------
* Start from the model we already selected: `nlptown/bert-base-multilingual-uncased-sentiment`
  (1 to 5 stars). Its zero-shot test accuracy on the 600 reviews is the number to beat.
* Keep the 5-star output layer. The training labels are only negative/positive, so the loss
  is computed on the *mapped* probabilities: P(negative) = P(1) + P(2) + P(3) and
  P(positive) = P(4) + P(5), the same mapping the API uses. Training therefore starts from the
  zero-shot model instead of from a randomly initialised head, and the saved model is a
  drop-in replacement for `models/nlptown_sentiment` (same labels, same `BertPredictor`).
* Predictions use the same rule as the API: take the most likely star, then map it.
  Metrics from this script are therefore the ones the API will reproduce.
* Three recipes are compared on the validation split only. The best one (macro-F1, accuracy
  breaks ties) is saved and scored on the test split exactly once. No test result is used
  to choose anything.

MLflow (experiment `reviewpulse-bert`, tagged stage=finetune_candidate / finetune_selected)
-------------------------------------------------------------------------------------------
One run per recipe (parameters, per-epoch validation metrics, best epoch, timing) and one
`finetune_selected_<recipe>` run with validation and test metrics plus the saved metadata.

Run (GPU strongly recommended, see README):
    python -m src.finetune_bert --smoke           # tiny end-to-end check, no MLflow, own outputs
    python -m src.finetune_bert                   # all three recipes on the real splits
    python -m src.finetune_bert --configs lr2e5_ep3
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

import numpy as np

from src.common import (
    DATASET_REVISION,
    MLFLOW_BERT_EXPERIMENT,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SEED,
    compute_metrics,
    load_split,
    write_predictions,
)

BASE_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
FINETUNED_DIR = MODELS_DIR / "nlptown_finetuned"
DEFAULT_TRACKING_URI = "sqlite:///reviewpulse_mlflow.db"

# Index 0 is "1 star", index 4 is "5 stars". 1 to 3 stars -> negative (0), 4 to 5 -> positive (1).
STAR_TO_BINARY = [0, 0, 0, 1, 1]
EVAL_MAX_LENGTH = 512  # same truncation length as the API and the zero-shot evaluation


@dataclass(frozen=True)
class FinetuneConfig:
    """One fine-tuning recipe. `freeze_layers` freezes the embeddings and that many layers."""

    name: str
    learning_rate: float
    epochs: int
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    freeze_layers: int = 0
    max_length: int = 256


CONFIGS = {
    config.name: config
    for config in (
        # Small steps, short run: least risk of forgetting what the model already knows.
        FinetuneConfig("lr1e5_ep2", learning_rate=1e-5, epochs=2),
        # The usual BERT recipe.
        FinetuneConfig("lr2e5_ep3", learning_rate=2e-5, epochs=3),
        # Train only the upper half of the network, with a larger step size.
        FinetuneConfig("lr3e5_ep3_freeze6", learning_rate=3e-5, epochs=3, freeze_layers=6),
    )
}


# ------------------------------------------------------------- pure helpers (no torch)


def pick_best(results: dict[str, dict]) -> str:
    """Name of the recipe with the best validation macro-F1 (accuracy breaks ties)."""
    if not results:
        raise ValueError("No results to choose from.")
    return max(results, key=lambda name: (results[name]["macro_f1"], results[name]["accuracy"]))


def decide(probabilities: np.ndarray) -> tuple[list[int], list[float]]:
    """Serving rule: most likely star, mapped to 0/1. Score = probability of that star."""
    probabilities = np.asarray(probabilities)
    stars = probabilities.argmax(axis=1)
    labels = [STAR_TO_BINARY[int(star)] for star in stars]
    scores = [float(score) for score in probabilities.max(axis=1)]
    return labels, scores


def decide_by_sum(probabilities: np.ndarray) -> list[int]:
    """Alternative rule: positive when P(4) + P(5) > 0.5. Logged as a diagnostic only."""
    probabilities = np.asarray(probabilities)
    return [int(value > 0.5) for value in probabilities[:, 3:].sum(axis=1)]


def evaluate_probabilities(probabilities: np.ndarray, true_labels: list[int]) -> dict:
    """Metrics for a (N, 5) probability matrix, using the serving rule."""
    labels, _ = decide(probabilities)
    metrics = compute_metrics(true_labels, labels)
    metrics["accuracy_sum_rule"] = compute_metrics(true_labels, decide_by_sum(probabilities))[
        "accuracy"
    ]
    return metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------ MLflow wrapper


class Tracker:
    """Thin MLflow wrapper. When disabled every call does nothing, so the script runs anywhere."""

    def __init__(self, enabled: bool, tracking_uri: str | None = None):
        self.enabled = enabled
        if enabled:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_BERT_EXPERIMENT)

    @contextlib.contextmanager
    def run(self, run_name: str, tags: dict | None = None):
        if not self.enabled:
            yield self
            return
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags or {})
            yield self

    def params(self, values: dict) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_params({key: str(value) for key, value in values.items()})

    def metrics(self, values: dict, step: int | None = None) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_metrics({k: float(v) for k, v in values.items()}, step=step)

    def dict(self, content: dict, file_name: str) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_dict(content, file_name)

    def artifact(self, path) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_artifact(str(path))


# ------------------------------------------------------------------- torch helpers


def binary_log_probs(logits):
    """(N, 5) star logits -> (N, 2) log-probabilities of [negative, positive]."""
    import torch

    log_p = torch.log_softmax(logits.float(), dim=-1)
    negative = torch.logsumexp(log_p[:, :3], dim=-1)
    positive = torch.logsumexp(log_p[:, 3:], dim=-1)
    return torch.stack([negative, positive], dim=-1)


def training_loss(logits, labels):
    """Negative log-likelihood of the true binary label under the mapped probabilities."""
    from torch.nn import functional

    return functional.nll_loss(binary_log_probs(logits), labels)


def freeze_lower_layers(model, n_layers: int) -> None:
    """Freeze the embeddings and the first `n_layers` encoder layers."""
    if n_layers <= 0:
        return
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.bert.encoder.layer[:n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _autocast(torch, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def predict_probabilities(model, tokenizer, texts, device, use_amp, batch_size=64):
    """Softmax probabilities over the 5 stars, in the original order of `texts`."""
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch
    output = np.zeros((len(texts), 5), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch = tokenizer(
                [texts[i] for i in index],
                truncation=True,
                max_length=EVAL_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with _autocast(torch, use_amp):
                logits = model(**batch).logits
            output[index] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return output


def share_truncated(tokenizer, texts: list[str], max_length: int) -> float:
    """Fraction of reviews longer than `max_length` tokens (they are cut during training)."""
    lengths = tokenizer(texts, truncation=False, verbose=False)["input_ids"]
    return float(np.mean([len(ids) > max_length for ids in lengths]))


def make_scaler(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# -------------------------------------------------------------------------- training


def train_one(config, train, validation, tokenizer, device, use_amp, seed, tracker, zero_shot):
    """Fine-tune one recipe. Returns the best-epoch model and a summary dict."""
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
    freeze_lower_layers(model, config.freeze_layers)
    total_params, trainable_params = count_parameters(model)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in named if not any(k in n for k in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in named if any(k in n for k in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
    )
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    scaler = make_scaler(torch, use_amp)

    texts = train["review_text"].tolist()
    labels = train["label"].tolist()
    validation_texts = validation["review_text"].tolist()
    validation_labels = validation["label"].tolist()

    tracker.params(
        {
            **asdict(config),
            "base_model": BASE_MODEL,
            "seed": seed,
            "train_size": len(train),
            "validation_size": len(validation),
            "dataset_revision": DATASET_REVISION,
            "device": str(device),
            "mixed_precision": use_amp,
            "eval_max_length": EVAL_MAX_LENGTH,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "train_share_truncated": round(share_truncated(tokenizer, texts, config.max_length), 4),
            "loss": "nll on mapped binary probabilities (1-3 stars vs 4-5 stars)",
        }
    )
    tracker.metrics(
        {
            "val_accuracy": zero_shot["accuracy"],
            "val_macro_f1": zero_shot["macro_f1"],
            "val_accuracy_sum_rule": zero_shot["accuracy_sum_rule"],
        },
        step=0,
    )

    best = {"macro_f1": -1.0, "accuracy": -1.0, "epoch": 0, "state": None}
    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(texts))
        epoch_started = time.perf_counter()
        losses = []
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = tokenizer(
                [texts[i] for i in index],
                truncation=True,
                max_length=config.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            target = torch.tensor([labels[i] for i in index], device=device)

            with _autocast(torch, use_amp):
                logits = model(**batch).logits
            loss = training_loss(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            losses.append(float(loss.detach()))
            if global_step % 50 == 0:
                tracker.metrics({"train_loss_step": float(np.mean(losses[-50:]))}, global_step)

        probabilities = predict_probabilities(model, tokenizer, validation_texts, device, use_amp)
        metrics = evaluate_probabilities(probabilities, validation_labels)
        tracker.metrics(
            {
                "train_loss": float(np.mean(losses)),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_accuracy_sum_rule": metrics["accuracy_sum_rule"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            },
            step=epoch,
        )
        print(
            f"  epoch {epoch}/{config.epochs}: train loss {np.mean(losses):.4f}, "
            f"validation accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}"
        )

        if (metrics["macro_f1"], metrics["accuracy"]) > (best["macro_f1"], best["accuracy"]):
            best = {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "metrics": metrics,
            }

    model.load_state_dict(best["state"])
    summary = {
        "macro_f1": best["macro_f1"],
        "accuracy": best["accuracy"],
        "best_epoch": best["epoch"],
        "metrics": best["metrics"],
        "train_seconds": time.perf_counter() - started,
        "trainable_parameters": trainable_params,
    }
    tracker.metrics(
        {
            "best_val_accuracy": summary["accuracy"],
            "best_val_macro_f1": summary["macro_f1"],
            "best_epoch": summary["best_epoch"],
            "train_seconds": summary["train_seconds"],
        }
    )
    tracker.dict(summary["metrics"], "validation_metrics.json")
    return model, summary


def release() -> None:
    """Free memory after the caller has deleted a model (call `del model` first)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------- main


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune nlptown BERT and track it in MLflow.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS),
        choices=list(CONFIGS),
        help="recipes to run (default: all three)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end check: a few reviews, 1 epoch, no MLflow, separate output files",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow database (default: $MLFLOW_TRACKING_URI, else a local sqlite file)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cpu":
        print("WARNING: no GPU found. Fine-tuning BERT on a CPU takes many hours. Use --smoke")
        print("         to check the script, and run the real training on a GPU (Kaggle, Colab).")

    suffix = "_smoke" if args.smoke else ""
    model_dir = FINETUNED_DIR.with_name(FINETUNED_DIR.name + suffix)
    use_mlflow = not (args.no_mlflow or args.smoke)
    configs = {
        n: (replace(CONFIGS[n], epochs=1) if args.smoke else CONFIGS[n]) for n in args.configs
    }
    train = load_split("train")
    validation = load_split("validation")
    if args.smoke:
        print("Smoke mode: a few reviews, 1 epoch, no MLflow. The scores mean nothing.")
        train, validation = train.head(160), validation.head(80)
    print(
        f"Train: {len(train)} reviews. Validation: {len(validation)} reviews. "
        f"Device: {device}, mixed precision: {use_amp}."
    )

    tracker = Tracker(use_mlflow, args.tracking_uri)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Zero-shot reference on validation: it should be close to 0.930 accuracy / 0.9298 macro-F1,
    # the numbers from the Kaggle candidate comparison. A big difference means a setup problem.
    print("Scoring the pretrained model on validation (reference)...")
    reference = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
    zero_shot = evaluate_probabilities(
        predict_probabilities(
            reference, tokenizer, validation["review_text"].tolist(), device, use_amp
        ),
        validation["label"].tolist(),
    )
    del reference
    release()
    print(
        f"  zero-shot validation accuracy {zero_shot['accuracy']:.4f}, "
        f"macro-F1 {zero_shot['macro_f1']:.4f}"
    )

    results: dict[str, dict] = {}
    best_name = None
    base_tags = {"model_family": "bert", "base_model": BASE_MODEL, "head": "nlptown_5_star_mapped"}
    for name, config in configs.items():
        print(f"\nRecipe {name}: {asdict(config)}")
        with tracker.run(
            f"finetune_{name}",
            {**base_tags, "stage": "finetune_candidate", "transformers": transformers.__version__},
        ):
            model, summary = train_one(
                config, train, validation, tokenizer, device, use_amp, args.seed, tracker, zero_shot
            )
        results[name] = summary

        if pick_best(results) == name:  # new best so far: keep these weights on disk
            if model_dir.exists():
                shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            best_name = name
            print(f"  new best on validation, saved to {model_dir}")
        del model
        release()

    print(f"\nSelected on validation: {best_name}")
    chosen = configs[best_name]
    chosen_summary = results[best_name]

    metadata = {
        "model_name": BASE_MODEL,
        "label_mapping": "1-3 stars -> negative, 4-5 stars -> positive",
        "max_token_length": EVAL_MAX_LENGTH,
        "fine_tuned": True,
        "recipe": asdict(chosen),
        "training_loss": "nll on mapped binary probabilities (1-3 stars vs 4-5 stars)",
        "best_epoch": chosen_summary["best_epoch"],
        "train_size": len(train),
        "dataset_revision": DATASET_REVISION,
        "seed": args.seed,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "validation_accuracy": chosen_summary["accuracy"],
        "validation_macro_f1": chosen_summary["macro_f1"],
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Score the held-out test split once, through the saved artifact.
    test = load_split("test")
    if args.smoke:
        test = test.head(80)
    saved = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    probabilities = predict_probabilities(
        saved, tokenizer, test["review_text"].tolist(), device, use_amp
    )
    del saved
    release()
    labels, scores = decide(probabilities)
    test_metrics = compute_metrics(test["label"].tolist(), labels)
    test_metrics["model"] = f"nlptown-finetuned-{best_name}"
    test_metrics["split"] = "test"

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = write_predictions(
        PREDICTIONS_DIR / f"bert_finetuned{suffix}_test.csv",
        test["row_id"].tolist(),
        test["label"].tolist(),
        labels,
        scores,
    )
    metrics_path = PREDICTIONS_DIR / f"bert_finetuned{suffix}_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=4), encoding="utf-8")
    metadata["test_accuracy"] = test_metrics["accuracy"]
    metadata["test_macro_f1"] = test_metrics["macro_f1"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with tracker.run(
        f"finetune_selected_{best_name}",
        {**base_tags, "stage": "finetune_selected", "transformers": transformers.__version__},
    ):
        tracker.params(
            {
                **asdict(chosen),
                "base_model": BASE_MODEL,
                "seed": args.seed,
                "train_size": len(train),
                "validation_size": len(validation),
                "test_size": len(test),
                "dataset_revision": DATASET_REVISION,
                "artifact_path": str(model_dir),
            }
        )
        tracker.metrics(
            {
                "val_accuracy": chosen_summary["accuracy"],
                "val_macro_f1": chosen_summary["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "zero_shot_val_accuracy": zero_shot["accuracy"],
                "zero_shot_val_macro_f1": zero_shot["macro_f1"],
                "best_epoch": chosen_summary["best_epoch"],
            }
        )
        tracker.dict(chosen_summary["metrics"], "validation_metrics.json")
        for path in (metadata_path, metrics_path, predictions_path):
            tracker.artifact(path)

    print("\nValidation results")
    print(
        f"  {'zero-shot (pretrained)':<22} accuracy {zero_shot['accuracy']:.4f}  "
        f"macro-F1 {zero_shot['macro_f1']:.4f}"
    )
    for name, summary in results.items():
        print(
            f"  {name:<22} accuracy {summary['accuracy']:.4f}  macro-F1 {summary['macro_f1']:.4f}"
            f"  (best epoch {summary['best_epoch']}, {summary['train_seconds'] / 60:.1f} min)"
        )

    print(f"\nHeld-out test results for {test_metrics['model']}")
    print(f"  accuracy {test_metrics['accuracy']:.4f}, macro-F1 {test_metrics['macro_f1']:.4f}")
    print(f"  confusion matrix: {test_metrics['confusion_matrix']}")
    zero_shot_file = PREDICTIONS_DIR / "bert_test_metrics.json"
    if zero_shot_file.exists() and not args.smoke:
        before = json.loads(zero_shot_file.read_text(encoding="utf-8"))
        print(
            f"  pretrained model on the same test set: accuracy {before['accuracy']:.4f}, "
            f"macro-F1 {before['macro_f1']:.4f}"
        )
    print(f"Predictions saved to: {predictions_path}")
    print(f"Model stored in {model_dir}/ (kept out of Git).")
    if use_mlflow:
        print(f"MLflow experiment: {MLFLOW_BERT_EXPERIMENT} at {args.tracking_uri}")


if __name__ == "__main__":
    main()
