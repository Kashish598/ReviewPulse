"""Seeds and training-set size for the fine-tuned nlptown BERT, tracked in MLflow.

Why
---
`src/finetune_bert.py` trained on 10,000 reviews with one seed. Its three recipes were within
about one point of each other on 1,000 validation reviews, which is inside the noise of that
set, and one seed cannot say whether 0.944 is typical or lucky. This script asks two questions:

  1. How much does the result move between seeds?      (3 seeds per setting)
  2. How much does more training data help?              (10,000, 20,000 and 40,000 reviews)

Design
------
* One recipe (default `lr1e5_ep2`, the winner of the first study) and the same training code
  (`finetune_bert.train_one`): same loss, same 5-star head, same serving rule.
* Validation (1,000) and test (600) are exactly the old splits. The extra training reviews come
  from the rest of the official train pool, with duplicates removed and anything that appears in
  validation or test excluded. The 10,000 original reviews are always the first rows, so the
  training sets are nested: 10,000 inside 20,000 inside 40,000.
* Every (size, seed) pair is one run. Runs are split between workers (one per GPU), so two
  T4s train two runs at the same time.
* Workers never talk to MLflow. They record what they would have logged in a JSON file next to
  the results, and the `summarize` step replays everything into MLflow from one process. That
  avoids two processes writing to one sqlite file, and a logging problem cannot kill a
  20-minute training run.
* Selection uses validation only (macro-F1, accuracy breaks ties). The test set is scored for
  every run so the spread between seeds can be reported, but it is never used to choose
  anything. Quote the MEAN test accuracy over seeds, not the best seed: the best seed was
  picked on validation and is a slightly optimistic pick.

Commands (from the repository root)
-----------------------------------
    python -m src.finetune_scaling prepare                       # write train_extended.csv
    python -m src.finetune_scaling train --worker 0 --workers 2  # GPU 0 (set CUDA_VISIBLE_DEVICES)
    python -m src.finetune_scaling train --worker 1 --workers 2  # GPU 1, at the same time
    python -m src.finetune_scaling summarize                     # select, re-score, log to MLflow
    python -m src.finetune_scaling train --smoke ; ... summarize --smoke   # tiny end-to-end check
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    DATA_COLUMNS,
    DATASET_REVISION,
    MLFLOW_BERT_EXPERIMENT,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    SEED,
    compute_metrics,
    load_split,
    read_predictions,
    write_predictions,
)
from src.finetune_bert import (
    BASE_MODEL,
    CONFIGS,
    DEFAULT_TRACKING_URI,
    EVAL_MAX_LENGTH,
    Tracker,
    decide,
    evaluate_probabilities,
    pick_best,
    predict_probabilities,
    release,
    train_one,
)

DEFAULT_RECIPE = "lr1e5_ep2"
DEFAULT_SIZES = (10_000, 20_000, 40_000)
DEFAULT_SEEDS = (42, 43, 44)
SMOKE_SIZES = (80, 160)
SMOKE_SEEDS = (42, 43)
SMOKE_EVAL_ROWS = 80
EXTENDED_TRAIN_FILE = "train_extended.csv"


@dataclass(frozen=True)
class Layout:
    """Where one study keeps its files. The smoke test gets its own folders."""

    results_dir: Path
    runs_dir: Path
    final_model_dir: Path
    suffix: str


def layout(smoke: bool) -> Layout:
    suffix = "_smoke" if smoke else ""
    return Layout(
        results_dir=PREDICTIONS_DIR / f"scaling{suffix}",
        runs_dir=MODELS_DIR / f"scaling_runs{suffix}",
        final_model_dir=MODELS_DIR / f"nlptown_finetuned_v2{suffix}",
        suffix=suffix,
    )


# ------------------------------------------------------------- pure helpers (no torch)


def run_name(size: int, seed: int) -> str:
    return f"scale_n{size}_seed{seed}"


def make_jobs(sizes, seeds) -> list[tuple[int, int]]:
    """Every (training size, seed) pair."""
    sizes, seeds = [int(s) for s in sizes], [int(s) for s in seeds]
    if not sizes or not seeds:
        raise ValueError("Need at least one size and one seed.")
    if len(set(sizes)) != len(sizes) or len(set(seeds)) != len(seeds):
        raise ValueError("Sizes and seeds must not repeat.")
    if min(sizes) < 1:
        raise ValueError("Sizes must be positive.")
    return [(size, seed) for size in sizes for seed in seeds]


def assign_jobs(jobs, n_workers: int) -> list[list[tuple[int, int]]]:
    """Split jobs between workers so the total number of training reviews is balanced.

    Greedy: biggest job first, always to the least loaded worker. Inside a worker the jobs run
    from small to large, so the first results (a sanity check) arrive early.
    """
    if n_workers < 1:
        raise ValueError("Need at least one worker.")
    loads = [0] * n_workers
    assigned: list[list[tuple[int, int]]] = [[] for _ in range(n_workers)]
    for job in sorted(jobs, key=lambda j: (-j[0], j[1])):
        worker = loads.index(min(loads))
        assigned[worker].append(job)
        loads[worker] += job[0]
    return [sorted(worker_jobs) for worker_jobs in assigned]


def extend_train(
    base: pd.DataFrame,
    pool: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    n_total: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """The original training reviews first, then extra reviews from the rest of the train pool.

    `pool` is the cleaned official train split (see `prepare_data.clean_reviews`). Extra rows
    never include anything from `base`, `validation` or `test`, and are a random sample, so
    taking the first k rows of the result always gives a nested training set.
    """
    if n_total < len(base):
        raise ValueError(f"n_total ({n_total}) is smaller than the base split ({len(base)}).")

    pool_text = pool.set_index("row_id")["review_text"]
    base_text = base.set_index("row_id")["review_text"]
    if (
        not base_text.index.isin(pool_text.index).all()
        or not (pool_text.reindex(base_text.index) == base_text).all()
    ):
        raise ValueError(
            "The base training split does not come from this pool. "
            "Check that the dataset revision and the seed are the ones used for the splits."
        )

    banned = set(base["review_text"]) | set(validation["review_text"]) | set(test["review_text"])
    rest = pool[~pool["review_text"].isin(banned) & ~pool["row_id"].isin(base["row_id"])]
    need = n_total - len(base)
    if len(rest) < need:
        raise ValueError(f"Only {len(rest)} extra reviews are available, need {need}.")

    extra = rest.sample(n=need, random_state=seed)
    result = pd.concat([base[DATA_COLUMNS], extra[DATA_COLUMNS]], ignore_index=True)
    if not result["review_text"].is_unique or not result["row_id"].is_unique:
        raise ValueError("The extended training set contains duplicates.")
    return result


def exact_mcnemar_p(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test from the two discordant counts."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def wilson_interval(correct: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for an accuracy."""
    if n <= 0:
        raise ValueError("n must be positive.")
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def compare_predictions(path_a, path_b) -> dict | None:
    """Paired comparison of two prediction files on the same reviews (A is the reference).

    `fixed` = reviews A gets wrong and B gets right, `broken` = the other way round.
    Returns None when either file is missing.
    """
    if not Path(path_a).exists() or not Path(path_b).exists():
        return None
    a = read_predictions(path_a).set_index("row_id")
    b = read_predictions(path_b).set_index("row_id")
    if set(a.index) != set(b.index):
        raise ValueError("The two prediction files do not cover the same reviews.")
    b = b.loc[a.index]
    if not (a["label_true"] == b["label_true"]).all():
        raise ValueError("The two prediction files disagree on the true labels.")
    a_right = (a["pred"] == a["label_true"]).to_numpy()
    b_right = (b["pred"] == b["label_true"]).to_numpy()
    fixed = int((~a_right & b_right).sum())
    broken = int((a_right & ~b_right).sum())
    return {
        "n": int(len(a)),
        "accuracy_a": float(a_right.mean()),
        "accuracy_b": float(b_right.mean()),
        "fixed": fixed,
        "broken": broken,
        "mcnemar_p": exact_mcnemar_p(fixed, broken),
    }


def _json_default(value):
    """numpy numbers become plain numbers; anything else becomes text."""
    return value.item() if hasattr(value, "item") else str(value)


class RecordingTracker:
    """Same interface as `Tracker`, but only remembers the calls (plain dicts, JSON-safe)."""

    enabled = True

    def __init__(self):
        self.calls: list[dict] = []

    @contextlib.contextmanager
    def run(self, run_name, tags=None):
        yield self

    def params(self, values: dict) -> None:
        self.calls.append({"op": "params", "values": {k: str(v) for k, v in values.items()}})

    def metrics(self, values: dict, step: int | None = None) -> None:
        self.calls.append(
            {"op": "metrics", "values": {k: float(v) for k, v in values.items()}, "step": step}
        )

    def dict(self, content: dict, file_name: str) -> None:
        self.calls.append(
            {
                "op": "dict",
                "content": json.loads(json.dumps(content, default=_json_default)),
                "file": file_name,
            }
        )

    def artifact(self, path) -> None:
        pass  # files are not copied; the summarize step logs the files it produces


def replay(calls: list[dict], tracker) -> None:
    """Send recorded calls to a real `Tracker` (inside an open run)."""
    for call in calls:
        if call["op"] == "params":
            tracker.params(call["values"])
        elif call["op"] == "metrics":
            tracker.metrics(call["values"], step=call["step"])
        elif call["op"] == "dict":
            tracker.dict(call["content"], call["file"])
        else:
            raise ValueError(f"Unknown recorded call {call['op']!r}.")


def load_summaries(results_dir: Path) -> list[dict]:
    """All finished runs, sorted by size then seed."""
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(results_dir).glob("*_summary.json"))
    ]
    return sorted(summaries, key=lambda s: (s["train_size"], s["seed"]))


def select_run(summaries: list[dict]) -> str:
    """Name of the run with the best validation macro-F1 (accuracy breaks ties)."""
    return pick_best({s["name"]: s["val"] for s in summaries})


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    mean = float(np.mean(values))
    return mean, (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def aggregate_by_size(summaries: list[dict]) -> list[dict]:
    """Mean and standard deviation over seeds, one row per training size."""
    rows = []
    for size in sorted({s["train_size"] for s in summaries}):
        group = [s for s in summaries if s["train_size"] == size]
        row: dict = {"train_size": size, "runs": len(group), "seeds": [s["seed"] for s in group]}
        for split in ("val", "test"):
            for metric in ("accuracy", "macro_f1"):
                mean, std = _mean_std([s[split][metric] for s in group])
                row[f"{split}_{metric}_mean"] = mean
                row[f"{split}_{metric}_std"] = std
        accuracies = [s["test"]["accuracy"] for s in group]
        row["test_accuracy_min"], row["test_accuracy_max"] = min(accuracies), max(accuracies)
        rows.append(row)
    return rows


def ensemble_by_size(
    summaries: list[dict], results_dir: Path, val_labels: list[int], test_labels: list[int]
) -> list[dict]:
    """Average the star probabilities of all seeds of one size, then apply the serving rule.

    Diagnostic only: serving the ensemble would mean loading several models.
    """
    rows = []
    for size in sorted({s["train_size"] for s in summaries}):
        group = [s for s in summaries if s["train_size"] == size]
        if len(group) < 2:
            continue
        stored = [np.load(Path(results_dir) / f"{s['name']}_probs.npz") for s in group]
        val = evaluate_probabilities(np.mean([p["val"] for p in stored], axis=0), val_labels)
        test = evaluate_probabilities(np.mean([p["test"] for p in stored], axis=0), test_labels)
        rows.append(
            {
                "train_size": size,
                "runs": len(group),
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
            }
        )
    return rows


def build_report(
    summaries: list[dict], results_dir: Path, val_labels: list[int], test_labels: list[int]
) -> dict:
    """Everything the summarize step reports, computed without torch."""
    if not summaries:
        raise ValueError("No finished runs found. Run the train step first.")
    selected = select_run(summaries)
    chosen = next(s for s in summaries if s["name"] == selected)
    n_test = chosen["test"]["n"]
    low, high = wilson_interval(round(chosen["test"]["accuracy"] * n_test), n_test)
    return {
        "selected_run": selected,
        "selected_val": chosen["val"],
        "selected_test": chosen["test"],
        "selected_test_wilson_95": [low, high],
        "by_size": aggregate_by_size(summaries),
        "seed_ensembles": ensemble_by_size(summaries, results_dir, val_labels, test_labels),
    }


def runs_table(summaries: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": s["name"],
                "train_size": s["train_size"],
                "seed": s["seed"],
                "best_epoch": s["best_epoch"],
                "val_accuracy": s["val"]["accuracy"],
                "val_macro_f1": s["val"]["macro_f1"],
                "test_accuracy": s["test"]["accuracy"],
                "test_macro_f1": s["test"]["macro_f1"],
                "train_minutes": round(s["train_seconds"] / 60, 1),
            }
            for s in summaries
        ]
    )


# ------------------------------------------------------------------------------ prepare


def cmd_prepare(args) -> None:
    """Build data/processed/train_extended.csv: the 10,000 original reviews plus extras."""
    from src.prepare_data import clean_reviews, load_raw

    base = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    train_raw, _, revision = load_raw(DATASET_REVISION)
    pool, cleaning = clean_reviews(train_raw, "tr")
    del train_raw
    extended = extend_train(base, pool, validation, test, args.max_size, seed=args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / EXTENDED_TRAIN_FILE
    extended.to_csv(path, index=False)
    counts = {str(k): int(v) for k, v in extended["label"].value_counts().sort_index().items()}
    report = {
        "dataset_revision": revision,
        "seed": args.seed,
        "rows": len(extended),
        "base_rows": len(base),
        "extra_rows": len(extended) - len(base),
        "label_counts": counts,
        "pool_cleaning": cleaning,
    }
    (PROCESSED_DIR / "train_extended_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")


def load_training_pool(max_size: int) -> pd.DataFrame:
    """The extended file when it exists (it starts with the original rows), else the base split."""
    path = PROCESSED_DIR / EXTENDED_TRAIN_FILE
    frame = pd.read_csv(path) if path.exists() else load_split("train")
    frame["label"] = frame["label"].astype(int)
    if len(frame) < max_size:
        raise ValueError(
            f"Need {max_size} training reviews but only {len(frame)} are available. "
            "Run `python -m src.finetune_scaling prepare` first."
        )
    return frame[DATA_COLUMNS]


def load_eval_splits(smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation, test = load_split("validation"), load_split("test")
    if smoke:
        validation, test = validation.head(SMOKE_EVAL_ROWS), test.head(SMOKE_EVAL_ROWS)
    return validation, test


# -------------------------------------------------------------------------------- train


def cmd_train(args) -> None:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not 0 <= args.worker < args.workers:
        raise SystemExit(f"--worker must be between 0 and {args.workers - 1}.")
    lay = layout(args.smoke)
    sizes = args.sizes or (SMOKE_SIZES if args.smoke else DEFAULT_SIZES)
    seeds = args.seeds or (SMOKE_SEEDS if args.smoke else DEFAULT_SEEDS)
    config = CONFIGS[args.recipe]
    if args.smoke:
        config = replace(config, epochs=1)
    mine = assign_jobs(make_jobs(sizes, seeds), args.workers)[args.worker]
    if not mine:
        print(f"Worker {args.worker} has no jobs.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cpu":
        print("WARNING: no GPU found. Use --smoke on a CPU and run the real study on a GPU.")
    pool = load_training_pool(max(size for size, _ in mine))
    validation, test = load_eval_splits(args.smoke)
    val_texts, val_labels = validation["review_text"].tolist(), validation["label"].tolist()
    test_texts, test_labels = test["review_text"].tolist(), test["label"].tolist()
    print(
        f"Worker {args.worker}/{args.workers}: recipe {config.name}, jobs {mine}. "
        f"Validation {len(validation)}, test {len(test)}. "
        f"Device {device}, mixed precision {use_amp}."
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    reference = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
    zero_shot = evaluate_probabilities(
        predict_probabilities(reference, tokenizer, val_texts, device, use_amp), val_labels
    )
    del reference
    release()
    print(
        f"  zero-shot validation accuracy {zero_shot['accuracy']:.4f}, "
        f"macro-F1 {zero_shot['macro_f1']:.4f}"
    )

    lay.results_dir.mkdir(parents=True, exist_ok=True)
    for size, seed in mine:
        name = run_name(size, seed)
        summary_path = lay.results_dir / f"{name}_summary.json"
        if summary_path.exists() and not args.force:
            print(f"\n{name}: already done, skipping (use --force to redo).")
            continue

        print(f"\n{name}: {size} training reviews, seed {seed}")
        train = pool.head(size).reset_index(drop=True)
        recorder = RecordingTracker()
        model, summary = train_one(
            config, train, validation, tokenizer, device, use_amp, seed, recorder, zero_shot
        )
        val_probs = predict_probabilities(model, tokenizer, val_texts, device, use_amp)
        test_probs = predict_probabilities(model, tokenizer, test_texts, device, use_amp)
        val_metrics = evaluate_probabilities(val_probs, val_labels)
        test_metrics = evaluate_probabilities(test_probs, test_labels)
        recorder.metrics(
            {"test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}
        )

        run_dir = lay.runs_dir / name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)
        del model
        release()

        np.savez_compressed(
            lay.results_dir / f"{name}_probs.npz",
            val=val_probs.astype(np.float32),
            test=test_probs.astype(np.float32),
        )
        labels, scores = decide(test_probs)
        write_predictions(
            lay.results_dir / f"{name}_test.csv",
            test["row_id"].tolist(),
            test_labels,
            labels,
            scores,
        )
        record = {
            "name": name,
            "train_size": len(train),
            "seed": seed,
            "recipe": asdict(config),
            "best_epoch": summary["best_epoch"],
            "train_seconds": summary["train_seconds"],
            "val": val_metrics,
            "test": test_metrics,
            "model_dir": str(run_dir),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "log": recorder.calls,
        }
        # Written last: a run without its summary file counts as not done and is redone.
        text = json.dumps(record, indent=2, default=_json_default)
        summary_path.write_text(text, encoding="utf-8")
        print(
            f"  {name}: validation accuracy {val_metrics['accuracy']:.4f} "
            f"(training-time best {summary['accuracy']:.4f}), "
            f"test accuracy {test_metrics['accuracy']:.4f}, "
            f"{summary['train_seconds'] / 60:.1f} min"
        )
    print(f"\nWorker {args.worker} finished.")


# --------------------------------------------------------------------------- summarize


def _print_report(report: dict, summaries: list[dict], comparisons: dict) -> None:
    print("\nAll runs")
    print(runs_table(summaries).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nBy training size (mean +/- std over seeds)")
    for row in report["by_size"]:

        def fmt(key, row=row):
            std = row[f"{key}_std"]
            return f"{row[f'{key}_mean']:.4f}" + (f" +/- {std:.4f}" if std is not None else "")

        print(
            f"  {row['train_size']:>6} reviews, {row['runs']} seeds: "
            f"validation accuracy {fmt('val_accuracy')}, test accuracy {fmt('test_accuracy')} "
            f"(range {row['test_accuracy_min']:.4f} to {row['test_accuracy_max']:.4f})"
        )
    if report["seed_ensembles"]:
        print("\nSeed ensembles (average of the seeds' probabilities, diagnostic only)")
        for row in report["seed_ensembles"]:
            print(
                f"  {row['train_size']:>6} reviews: validation accuracy {row['val_accuracy']:.4f}, "
                f"test accuracy {row['test_accuracy']:.4f}"
            )

    low, high = report["selected_test_wilson_95"]
    print(f"\nSelected on validation: {report['selected_run']}")
    print(
        f"  validation accuracy {report['selected_val']['accuracy']:.4f}, "
        f"test accuracy {report['selected_test']['accuracy']:.4f} "
        f"(95% interval {low:.4f} to {high:.4f}), "
        f"confusion matrix {report['selected_test']['confusion_matrix']}"
    )
    for label, result in comparisons.items():
        if result is None:
            continue
        print(
            f"  vs {label}: {result['accuracy_a']:.4f} -> {result['accuracy_b']:.4f}, "
            f"fixed {result['fixed']}, broken {result['broken']}, "
            f"exact McNemar p = {result['mcnemar_p']:.4f}"
        )
    print("\nQuote the mean over seeds as the headline number. The selected run is the best")
    print("seed on validation, so its test score is a slightly optimistic pick.")


def cmd_summarize(args) -> None:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    lay = layout(args.smoke)
    summaries = load_summaries(lay.results_dir)
    validation, test = load_eval_splits(args.smoke)
    val_labels, test_labels = validation["label"].tolist(), test["label"].tolist()
    for s in summaries:
        if s["val"]["n"] != len(validation) or s["test"]["n"] != len(test):
            raise SystemExit(f"{s['name']} was scored on different splits than the current ones.")
    report = build_report(summaries, lay.results_dir, val_labels, test_labels)
    selected = next(s for s in summaries if s["name"] == report["selected_run"])

    # Keep the winner as the new fine-tuned model; delete the other checkpoints.
    chosen_dir = Path(selected["model_dir"])
    if not chosen_dir.exists():
        raise SystemExit(f"{chosen_dir} is missing. Re-run the train step with --force for it.")
    if lay.final_model_dir.exists():
        shutil.rmtree(lay.final_model_dir)
    lay.final_model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(chosen_dir), str(lay.final_model_dir))
    if not args.keep_runs:
        for s in summaries:
            other = Path(s["model_dir"])
            if other != chosen_dir and other.exists():
                shutil.rmtree(other)

    # Score the test split once more through the saved artifact: this is what will be served.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    tokenizer = AutoTokenizer.from_pretrained(lay.final_model_dir)
    saved = AutoModelForSequenceClassification.from_pretrained(lay.final_model_dir).to(device)
    probabilities = predict_probabilities(
        saved, tokenizer, test["review_text"].tolist(), device, use_amp
    )
    del saved
    release()
    labels, scores = decide(probabilities)
    in_run_labels, _ = decide(np.load(lay.results_dir / f"{selected['name']}_probs.npz")["test"])
    agreement = float(np.mean(np.array(labels) == np.array(in_run_labels)))
    test_metrics = compute_metrics(test_labels, labels)
    test_metrics["model"] = f"nlptown-finetuned-v2-{selected['name']}"
    test_metrics["split"] = "test"

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = write_predictions(
        PREDICTIONS_DIR / f"bert_finetuned_v2{lay.suffix}_test.csv",
        test["row_id"].tolist(),
        test_labels,
        labels,
        scores,
    )
    metrics_path = PREDICTIONS_DIR / f"bert_finetuned_v2{lay.suffix}_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=4), encoding="utf-8")

    comparisons: dict = {}
    if not args.smoke:
        comparisons = {
            "pretrained model": compare_predictions(
                PREDICTIONS_DIR / "bert_test.csv", predictions_path
            ),
            "first fine-tuned model (10,000 reviews, 1 seed)": compare_predictions(
                PREDICTIONS_DIR / "bert_finetuned_test.csv", predictions_path
            ),
        }

    metadata = {
        "model_name": BASE_MODEL,
        "label_mapping": "1-3 stars -> negative, 4-5 stars -> positive",
        "max_token_length": EVAL_MAX_LENGTH,
        "fine_tuned": True,
        "recipe": selected["recipe"],
        "training_loss": "nll on mapped binary probabilities (1-3 stars vs 4-5 stars)",
        "selected_run": selected["name"],
        "train_size": selected["train_size"],
        "seed": selected["seed"],
        "best_epoch": selected["best_epoch"],
        "dataset_revision": DATASET_REVISION,
        "selection": "best validation macro-F1 over all (size, seed) runs",
        "validation_accuracy": selected["val"]["accuracy"],
        "validation_macro_f1": selected["val"]["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "artifact_agreement_with_training_run": agreement,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (lay.final_model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    table_path = lay.results_dir / "runs_table.csv"
    runs_table(summaries).to_csv(table_path, index=False)
    by_size_path = lay.results_dir / "by_size.csv"
    pd.DataFrame(report["by_size"]).to_csv(by_size_path, index=False)
    report["comparisons"] = comparisons
    report_path = lay.results_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    _print_report(report, summaries, comparisons)
    print(f"\nSaved artifact agrees with the training run on {agreement:.1%} of test predictions.")
    print(f"Model: {lay.final_model_dir}/   Predictions: {predictions_path}")

    if args.smoke or args.no_mlflow:
        return
    tracker = Tracker(True, args.tracking_uri)
    base_tags = {"model_family": "bert", "base_model": BASE_MODEL, "head": "nlptown_5_star_mapped"}
    for s in summaries:
        with tracker.run(
            s["name"],
            {**base_tags, "stage": "scaling_candidate", "transformers": s["transformers"]},
        ):
            replay(s["log"], tracker)
    flat: dict[str, float] = {
        "val_accuracy": selected["val"]["accuracy"],
        "val_macro_f1": selected["val"]["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "artifact_agreement": agreement,
    }
    for row in report["by_size"]:
        for key, value in row.items():
            if key.startswith(("val_", "test_")) and value is not None:
                flat[f"n{row['train_size']}_{key}"] = value
    for row in report["seed_ensembles"]:
        flat[f"n{row['train_size']}_ensemble_test_accuracy"] = row["test_accuracy"]
        flat[f"n{row['train_size']}_ensemble_val_accuracy"] = row["val_accuracy"]
    for key, result in (("zero_shot", comparisons.get("pretrained model")),):
        if result:
            flat[f"{key}_fixed"], flat[f"{key}_broken"] = result["fixed"], result["broken"]
            flat[f"{key}_mcnemar_p"] = result["mcnemar_p"]
    with tracker.run(
        f"scaling_selected_{selected['name']}",
        {**base_tags, "stage": "scaling_selected", "transformers": transformers.__version__},
    ):
        tracker.params(
            {
                **selected["recipe"],
                "base_model": BASE_MODEL,
                "selected_run": selected["name"],
                "seed": selected["seed"],
                "train_size": selected["train_size"],
                "sizes": [row["train_size"] for row in report["by_size"]],
                "seeds_per_size": [row["runs"] for row in report["by_size"]],
                "dataset_revision": DATASET_REVISION,
                "artifact_path": str(lay.final_model_dir),
            }
        )
        tracker.metrics(flat)
        for path in (
            lay.final_model_dir / "metadata.json",
            metrics_path,
            predictions_path,
            table_path,
            by_size_path,
            report_path,
        ):
            tracker.artifact(path)
    print(f"MLflow experiment: {MLFLOW_BERT_EXPERIMENT} at {args.tracking_uri}")


# ------------------------------------------------------------------------------- CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Seeds and training size for fine-tuned BERT.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="build data/processed/train_extended.csv")
    prepare.add_argument("--max-size", type=int, default=max(DEFAULT_SIZES))
    prepare.add_argument("--seed", type=int, default=SEED)

    train = commands.add_parser("train", help="train this worker's share of the runs")
    train.add_argument("--worker", type=int, default=0, help="index of this worker (0-based)")
    train.add_argument("--workers", type=int, default=1, help="number of workers (one per GPU)")
    train.add_argument("--recipe", default=DEFAULT_RECIPE, choices=list(CONFIGS))
    train.add_argument("--sizes", type=int, nargs="+", help=f"default {list(DEFAULT_SIZES)}")
    train.add_argument("--seeds", type=int, nargs="+", help=f"default {list(DEFAULT_SEEDS)}")
    train.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    train.add_argument("--force", action="store_true", help="redo runs that already finished")
    train.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision")

    summarize = commands.add_parser("summarize", help="select, re-score and log to MLflow")
    summarize.add_argument("--smoke", action="store_true")
    summarize.add_argument("--no-mlflow", action="store_true")
    summarize.add_argument("--keep-runs", action="store_true", help="keep every run's checkpoint")
    summarize.add_argument("--no-amp", action="store_true")
    summarize.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow database (default: $MLFLOW_TRACKING_URI, else a local sqlite file)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    {"prepare": cmd_prepare, "train": cmd_train, "summarize": cmd_summarize}[args.command](args)
    print(f"({args.command} took {(time.perf_counter() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
