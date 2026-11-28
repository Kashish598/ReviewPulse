"""k-fold cross-validation of the nlptown fine-tuning recipe, logged to MLflow as nested runs.

Why
---
The first fine-tuning study picked a recipe on one fixed validation set of 1,000 reviews, so
the choice depended on which 1,000 reviews were in it. Cross-validation reuses the 10,000
training reviews: every review is held out exactly once, and the spread of the score across
folds shows how much the result depends on the data.

Design
------
* Only the 10,000 training reviews are used. The 1,000 validation and 600 test reviews are not
  touched, so they stay available for choosing and reporting the final model.
* Stratified folds (same share of negative and positive in each fold), fixed seed. The fold
  assignment is saved (`folds.json`) so anyone can reproduce it.
* The recipe is the proven one (learning rate 1e-5) and only the number of epochs changes
  (1 and 2), because the question is how many epochs. Each epoch count gets its own run with its
  own learning-rate schedule, so the numbers are comparable.
* The score of a fold is the score after the LAST epoch. `train_one` also restores the best
  epoch, but picking the best epoch on the held-out fold would make the estimate optimistic.
* The pretrained model (no training) is scored on the same folds, so every fold has a baseline
  and the gain from fine-tuning is measured per fold.
* Two steps, like `finetune_scaling`:
    train  trains every fold on the GPU and writes one JSON file per fold. No MLflow, no
           token needed, and a network problem cannot kill a training run.
    log    replays the JSON files into MLflow: one parent run per recipe, one nested run per
           fold, aggregate metrics on the parent. Run it from any machine with the MLflow
           address and login set in the environment.
* Fold models are not saved (650 MB each). Cross-validation chooses a setting. The final model
  is trained once afterwards with the chosen setting.

Run (GPU needed for `train`, see the notebook):
    python -m src.finetune_cv train --smoke           # tiny end-to-end check, no MLflow
    python -m src.finetune_cv train                   # 5 folds, 2 recipes
    python -m src.finetune_cv log --tracking-uri sqlite:///reviewpulse_mlflow.db
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.common import DATASET_REVISION, PREDICTIONS_DIR, SEED, load_split
from src.finetune_bert import (
    BASE_MODEL,
    CONFIGS,
    evaluate_probabilities,
    predict_probabilities,
    release,
    train_one,
)

CV_EXPERIMENT = "reviewpulse-bert-cv"
CV_DIR = PREDICTIONS_DIR / "cv"
CV_SMOKE_DIR = PREDICTIONS_DIR / "cv_smoke"
DEFAULT_FOLDS = 5
DEFAULT_TRACKING_URI = "sqlite:///reviewpulse_mlflow.db"

# The winning recipe of the first study, with only the number of epochs changed.
CV_CONFIGS = {
    "lr1e5_ep1": replace(CONFIGS["lr1e5_ep2"], name="lr1e5_ep1", epochs=1),
    "lr1e5_ep2": CONFIGS["lr1e5_ep2"],
}


# ------------------------------------------------------------- pure helpers (no torch)


def make_folds(labels, n_folds: int, seed: int) -> list[dict]:
    """Stratified k-fold split. Every row is in exactly one validation fold."""
    if n_folds < 2:
        raise ValueError("Cross-validation needs at least 2 folds.")
    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (train_index, validation_index) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        folds.append(
            {
                "fold": k,
                "train_index": train_index.tolist(),
                "validation_index": validation_index.tolist(),
            }
        )
    return folds


def summarize_values(values) -> dict:
    """Mean and sample standard deviation (n-1) of per-fold scores."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("No values to summarise.")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def final_epoch_scores(series: dict, epochs: int) -> dict:
    """Validation scores after the last epoch, from the recorded (step, value) pairs."""
    scores = {}
    for name in ("val_accuracy", "val_macro_f1"):
        at_last = [value for step, value in series.get(name, []) if step == epochs]
        if not at_last:
            raise ValueError(f"No {name} recorded at epoch {epochs}.")
        scores[name.removeprefix("val_")] = float(at_last[-1])
    return scores


def epoch_table(series: dict, epochs: int) -> list[dict]:
    """One row per epoch: validation accuracy, macro-F1 and training loss."""
    rows = []
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch}
        for name in ("val_accuracy", "val_macro_f1", "train_loss"):
            found = [value for step, value in series.get(name, []) if step == epoch]
            row[name] = float(found[-1]) if found else None
        rows.append(row)
    return rows


def aggregate(fold_results: list[dict]) -> dict:
    """Mean and spread over folds, plus the gain over the untrained model on the same folds."""
    if not fold_results:
        raise ValueError("No fold results to aggregate.")
    final = [r["final"] for r in fold_results]
    zero = [r["zero_shot"] for r in fold_results]
    return {
        "n_folds": len(fold_results),
        "val_accuracy": summarize_values(f["accuracy"] for f in final),
        "val_macro_f1": summarize_values(f["macro_f1"] for f in final),
        "zero_shot_val_accuracy": summarize_values(z["accuracy"] for z in zero),
        "zero_shot_val_macro_f1": summarize_values(z["macro_f1"] for z in zero),
        "gain_macro_f1": summarize_values(
            f["macro_f1"] - z["macro_f1"] for f, z in zip(final, zero, strict=True)
        ),
        "gain_accuracy": summarize_values(
            f["accuracy"] - z["accuracy"] for f, z in zip(final, zero, strict=True)
        ),
        "mean_train_seconds": float(np.mean([r.get("train_seconds", 0.0) for r in fold_results])),
    }


class Recorder:
    """Stands in for the MLflow tracker during training: it only remembers what was logged.

    `train_one` calls params(), metrics() and dict(); this class keeps them in memory so they
    can be saved to a JSON file and replayed into MLflow later by the `log` step.
    """

    def __init__(self):
        self.parameters: dict[str, str] = {}
        self.series: dict[str, list[list[float]]] = {}
        self.final_metrics: dict[str, float] = {}
        self.documents: dict[str, dict] = {}

    def params(self, values: dict) -> None:
        self.parameters.update({key: str(value) for key, value in values.items()})

    def metrics(self, values: dict, step: int | None = None) -> None:
        for name, value in values.items():
            if step is None:
                self.final_metrics[name] = float(value)
            else:
                self.series.setdefault(name, []).append([int(step), float(value)])

    def dict(self, content: dict, file_name: str) -> None:
        self.documents[file_name] = content

    def artifact(self, path) -> None:  # fold models are not kept
        return None

    def to_json(self) -> dict:
        return {
            "parameters": self.parameters,
            "series": self.series,
            "final_metrics": self.final_metrics,
            "documents": self.documents,
        }


def _jsonable(value):
    """json.dumps helper for numpy numbers and arrays."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot save {type(value).__name__} to JSON.")


def result_path(out_dir: Path, config_name: str, fold: int) -> Path:
    return Path(out_dir) / config_name / f"fold{fold}.json"


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=1, default=_jsonable), encoding="utf-8")


def load_fold_results(out_dir: Path, config_name: str) -> list[dict]:
    """All finished folds of one recipe, ordered by fold number."""
    folder = Path(out_dir) / config_name
    results = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("fold*.json")]
    return sorted(results, key=lambda r: r["fold"])


# ------------------------------------------------------------------------ train step


def run_training(args) -> None:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cpu" and not args.smoke:
        print("WARNING: no GPU found. Cross-validation on a CPU takes many hours. Use --smoke")
        print("         to check the script, and run the real training on a GPU (Kaggle).")

    out_dir = Path(args.out_dir) if args.out_dir else (CV_SMOKE_DIR if args.smoke else CV_DIR)
    train = load_split("train")
    n_folds = args.folds
    config_names = args.configs
    if args.smoke:
        print("Smoke mode: 80 reviews, 2 folds, 1 epoch. The scores mean nothing.")
        train = train.groupby("label", group_keys=False).head(40).reset_index(drop=True)
        n_folds = 2
        config_names = config_names[:1]
    configs = {
        name: (replace(CV_CONFIGS[name], epochs=1) if args.smoke else CV_CONFIGS[name])
        for name in config_names
    }

    labels = train["label"].to_numpy()
    texts = train["review_text"].tolist()
    folds = make_folds(labels, n_folds, args.seed)
    write_json(
        out_dir / "folds.json",
        {
            "n_folds": n_folds,
            "seed": args.seed,
            "train_pool_size": len(train),
            "dataset_revision": DATASET_REVISION,
            "validation_row_ids": {
                str(f["fold"]): train["row_id"].iloc[f["validation_index"]].tolist() for f in folds
            },
        },
    )
    print(f"{len(train)} reviews, {n_folds} folds, device {device}, mixed precision {use_amp}.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # The pretrained model is scored once on all training rows. Each fold's baseline is the
    # slice of those predictions that belongs to the fold's held-out reviews.
    print("Scoring the pretrained model on the training reviews (baseline)...")
    reference = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
    baseline = predict_probabilities(reference, tokenizer, texts, device, use_amp)
    del reference
    release()

    for name, config in configs.items():
        print(f"\nRecipe {name}: {asdict(config)}")
        for fold in folds:
            k = fold["fold"]
            path = result_path(out_dir, name, k)
            if path.exists() and not args.force:
                print(f"  fold {k}: already done, skipped ({path})")
                continue
            train_part = train.iloc[fold["train_index"]]
            held_out = train.iloc[fold["validation_index"]]
            zero_shot = evaluate_probabilities(
                baseline[fold["validation_index"]], held_out["label"].tolist()
            )
            print(f"  fold {k}: train {len(train_part)}, held out {len(held_out)}")
            recorder = Recorder()
            started = time.perf_counter()
            model, summary = train_one(
                config=config,
                train=train_part,
                validation=held_out,
                tokenizer=tokenizer,
                device=device,
                use_amp=use_amp,
                seed=args.seed,
                tracker=recorder,
                zero_shot=zero_shot,
            )
            del model
            release()
            series = recorder.series
            write_json(
                path,
                {
                    "config_name": name,
                    "config": asdict(config),
                    "fold": k,
                    "n_folds": n_folds,
                    "cv_seed": args.seed,
                    "train_size": len(train_part),
                    "validation_size": len(held_out),
                    "zero_shot": zero_shot,
                    "final": final_epoch_scores(series, config.epochs),
                    "epochs": epoch_table(series, config.epochs),
                    "best_epoch_by_validation": summary["best_epoch"],
                    "train_seconds": time.perf_counter() - started,
                    "base_model": BASE_MODEL,
                    "dataset_revision": DATASET_REVISION,
                    "transformers": transformers.__version__,
                    "device": str(device),
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "recorded": recorder.to_json(),
                },
            )
    print(f"\nDone. Results are in {out_dir}. Next: python -m src.finetune_cv log")


# -------------------------------------------------------------------------- log step


def already_logged(mlflow, experiment: str, run_name: str) -> bool:
    found = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    return len(found) > 0


def _flat(prefix: str, summary: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in summary.items() if key != "n"}


def log_results(
    out_dir: Path,
    tracking_uri: str,
    experiment: str = CV_EXPERIMENT,
    config_names: list[str] | None = None,
    allow_partial: bool = False,
    allow_duplicates: bool = False,
) -> list[dict]:
    """Replay saved fold results into MLflow. Returns one summary row per logged recipe."""
    import mlflow

    out_dir = Path(out_dir)
    folds_path = out_dir / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError(f"{folds_path} not found. Run the `train` step first.")
    folds_doc = json.loads(folds_path.read_text(encoding="utf-8"))
    names = config_names or sorted(p.name for p in out_dir.iterdir() if p.is_dir())

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    who = os.environ.get("MLFLOW_TRACKING_USERNAME", "unknown")

    rows = []
    for name in names:
        results = load_fold_results(out_dir, name)
        if not results:
            print(f"{name}: no fold results, skipped.")
            continue
        if len(results) < folds_doc["n_folds"] and not allow_partial:
            print(f"{name}: only {len(results)} of {folds_doc['n_folds']} folds, skipped.")
            continue
        run_name = f"cv_{name}"
        if not allow_duplicates and already_logged(mlflow, experiment, run_name):
            print(f"{run_name}: already in MLflow, skipped (use --allow-duplicates to log again).")
            continue

        summary = aggregate(results)
        first = results[0]
        tags = {
            "stage": "cross_validation",
            "model_family": "bert",
            "base_model": first["base_model"],
            "recipe": name,
            "logged_by": who,
        }
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags)
            mlflow.log_params(
                {
                    **{k: str(v) for k, v in first["config"].items()},
                    "n_folds": summary["n_folds"],
                    "cv_seed": first["cv_seed"],
                    "train_pool_size": folds_doc["train_pool_size"],
                    "dataset_revision": first["dataset_revision"],
                    "split": "stratified k-fold on the training reviews only",
                    "score_used": "last epoch of each fold",
                }
            )
            for result in results:
                with mlflow.start_run(run_name=f"{run_name}_fold{result['fold']}", nested=True):
                    mlflow.set_tags({**tags, "fold": str(result["fold"])})
                    recorded = result["recorded"]
                    mlflow.log_params(recorded["parameters"])
                    mlflow.log_params({"fold": result["fold"], "cv_seed": result["cv_seed"]})
                    for metric, points in recorded["series"].items():
                        for step, value in points:
                            mlflow.log_metric(metric, value, step=int(step))
                    mlflow.log_metrics(
                        {
                            "final_val_accuracy": result["final"]["accuracy"],
                            "final_val_macro_f1": result["final"]["macro_f1"],
                            "zero_shot_val_accuracy": result["zero_shot"]["accuracy"],
                            "zero_shot_val_macro_f1": result["zero_shot"]["macro_f1"],
                            "train_seconds": result["train_seconds"],
                        }
                    )
                    mlflow.log_dict(
                        {"epochs": result["epochs"], "zero_shot": result["zero_shot"]},
                        "fold_result.json",
                    )
            mlflow.log_metrics(
                {
                    **_flat("cv_val_accuracy", summary["val_accuracy"]),
                    **_flat("cv_val_macro_f1", summary["val_macro_f1"]),
                    **_flat("cv_zero_shot_val_macro_f1", summary["zero_shot_val_macro_f1"]),
                    **_flat("cv_gain_macro_f1", summary["gain_macro_f1"]),
                    "cv_mean_train_seconds": summary["mean_train_seconds"],
                }
            )
            mlflow.log_dict(summary, "cv_summary.json")
            mlflow.log_dict(folds_doc, "folds.json")
        rows.append(
            {
                "recipe": name,
                "folds": summary["n_folds"],
                "macro_f1_mean": summary["val_macro_f1"]["mean"],
                "macro_f1_std": summary["val_macro_f1"]["std"],
                "accuracy_mean": summary["val_accuracy"]["mean"],
                "accuracy_std": summary["val_accuracy"]["std"],
                "gain_macro_f1_mean": summary["gain_macro_f1"]["mean"],
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("Nothing was logged.")
        return
    print(f"{'recipe':<12}{'folds':>6}{'macro-F1':>18}{'accuracy':>18}{'gain vs pretrained':>22}")
    for row in rows:
        print(
            f"{row['recipe']:<12}{row['folds']:>6}"
            f"{row['macro_f1_mean']:>10.4f} +/- {row['macro_f1_std']:.4f}"
            f"{row['accuracy_mean']:>10.4f} +/- {row['accuracy_std']:.4f}"
            f"{row['gain_macro_f1_mean']:>+16.4f}"
        )


# -------------------------------------------------------------------------- command line


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train every fold (GPU), write one JSON file per fold")
    train.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    train.add_argument("--seed", type=int, default=SEED)
    train.add_argument("--configs", nargs="+", choices=list(CV_CONFIGS), default=list(CV_CONFIGS))
    train.add_argument("--out-dir", default=None)
    train.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    train.add_argument(
        "--force", action="store_true", help="retrain folds that already have results"
    )
    train.add_argument("--no-amp", action="store_true", help="turn off mixed precision")

    log = sub.add_parser("log", help="replay the saved fold results into MLflow")
    log.add_argument("--out-dir", default=str(CV_DIR))
    log.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    log.add_argument("--experiment", default=CV_EXPERIMENT)
    log.add_argument("--configs", nargs="+", default=None)
    log.add_argument("--allow-partial", action="store_true")
    log.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        run_training(args)
    else:
        rows = log_results(
            Path(args.out_dir),
            args.tracking_uri,
            experiment=args.experiment,
            config_names=args.configs,
            allow_partial=args.allow_partial,
            allow_duplicates=args.allow_duplicates,
        )
        print_table(rows)


if __name__ == "__main__":
    main()
