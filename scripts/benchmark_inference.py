"""Inference benchmark: how heavy and how fast is each model when it serves requests?

    python scripts/benchmark_inference.py --models tfidf,pretrained,finetuned --n 200

For each model it reports disk size, load time, memory after loading, per-request latency
(p50, p95, p99, one review per call, like the API) and batch throughput. Each model is measured in
its own process so the memory numbers are not mixed up. Writes docs/inference_benchmark.json and
docs/inference_benchmark.md and records the machine they were measured on.

Memory needs `pip install psutil` (benchmark only, it is not a serving dependency).
Run it where you deploy, or say which machine the numbers are from: they depend on the CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("reviewpulse.benchmark")

MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
# Keep in step with src/train_tfidf.py and src/bert_model.py (tests check this).
MODEL_DIRS = {
    "tfidf": "tfidf_v1",
    "pretrained": "nlptown_sentiment",
    "finetuned": "nlptown_finetuned_v2",
}
MODEL_NAMES = (*MODEL_DIRS, "mock")

FALLBACK_TEXTS = [
    "I bought this for my son and he loves it. It works well and arrived on time.",
    "Stopped working after two weeks. Customer service never answered my emails.",
    "Great value for the price. The battery lasts all day and the screen is bright.",
    "Terrible quality. The zipper broke the first time I used it and the fabric feels cheap.",
    "Exactly what I needed. Easy to set up, sturdy, and it looks nice on my desk.",
    "I expected more. The sound is flat and the case scratches easily.",
]


# ------------------------------------------------------------------ measuring


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rss_mb() -> float | None:
    """Resident memory of this process in MB, or None when psutil is not installed."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def dir_size_mb(path: Path | None) -> float | None:
    if path is None or not Path(path).exists():
        return None
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 * 1024)


def measure_latency(predictor, texts: list[str], warmup: int = 5) -> list[float]:
    """Milliseconds per call, one review per call, after `warmup` calls that are not counted."""
    for text in texts[:warmup] or texts[:1]:
        predictor.predict([text])
    timings = []
    for text in texts:
        started = time.perf_counter()
        predictor.predict([text])
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def measure_throughput(predictor, texts: list[str]) -> float:
    """Reviews per second when all the reviews are sent in one call."""
    started = time.perf_counter()
    predictor.predict(texts)
    return len(texts) / (time.perf_counter() - started)


def benchmark_model(
    name: str, loader, texts: list[str], model_dir: Path | None, warmup: int
) -> dict:
    """Measure one model. `loader` builds the predictor and is timed as the load."""
    memory_before = rss_mb()
    started = time.perf_counter()
    predictor = loader()
    load_seconds = time.perf_counter() - started
    memory_after_load = rss_mb()

    timings = measure_latency(predictor, texts, warmup)
    throughput = measure_throughput(predictor, texts)
    memory_after_predict = rss_mb()

    def delta(after):
        return None if after is None or memory_before is None else after - memory_before

    return {
        "model": name,
        "served_as": getattr(predictor, "name", type(predictor).__name__),
        "disk_mb": dir_size_mb(model_dir),
        "load_seconds": load_seconds,
        "memory_after_load_mb": delta(memory_after_load),
        "memory_after_predict_mb": delta(memory_after_predict),
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "p50": percentile(timings, 50),
            "p95": percentile(timings, 95),
            "p99": percentile(timings, 99),
        },
        "throughput_reviews_per_second": throughput,
        "requests_timed": len(timings),
    }


# ------------------------------------------------------------------ models and data


def make_loader(name: str, models_dir: Path):
    """Return (loader, model_dir). Heavy imports happen inside the loader, so they are timed."""
    if name == "mock":
        from src.common import MockPredictor

        return MockPredictor, None

    model_dir = Path(models_dir) / MODEL_DIRS[name]
    if name == "tfidf":

        def load_tfidf():
            from src.tfidf_model import TfidfPredictor

            return TfidfPredictor(model_dir)

        return load_tfidf, model_dir

    def load_bert():
        from src.bert_model import BertPredictor

        return BertPredictor(model_dir)

    return load_bert, model_dir


def load_texts(csv_path: Path | None, n: int, seed: int) -> list[str]:
    import pandas as pd

    try:
        if csv_path:
            frame = pd.read_csv(csv_path)
        else:
            from src.common import load_split

            frame = load_split("test")
    except FileNotFoundError:
        logger.warning("No test data found, using %d built-in reviews", len(FALLBACK_TEXTS))
        return (FALLBACK_TEXTS * (n // len(FALLBACK_TEXTS) + 1))[:n]
    texts = [t for t in frame["review_text"].astype(str) if t.strip()]
    return pd.Series(texts).sample(n=min(n, len(texts)), random_state=seed).tolist()


def run_worker(args) -> int:
    """Child process: measure one model and write the result as JSON."""
    texts = json.loads(Path(args.texts_file).read_text(encoding="utf-8"))
    loader, model_dir = make_loader(args.worker, Path(args.models_dir))
    if model_dir is not None and not model_dir.exists():
        result = {"model": args.worker, "skipped": f"no model folder at {model_dir}"}
    else:
        result = benchmark_model(args.worker, loader, texts, model_dir, args.warmup)
    Path(args.result_file).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ reporting


def environment() -> dict:
    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "scikit_learn": version("scikit-learn"),
        "psutil_installed": version("psutil") is not None,
    }


def _fmt(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict) -> str:
    env, settings = report["environment"], report["settings"]
    lines = [
        "# Inference benchmark",
        "",
        f"Measured {env['measured_at']} on {env['platform']} with {env['cpu_count']} CPU(s), "
        f"Python {env['python']}, torch {env['torch'] or 'n/a'}.",
        f"{settings['n']} reviews, one review per call for latency, one call for throughput. "
        "Memory is the growth in process memory (RSS) while loading and serving the model.",
        "",
        "| Model | Disk (MB) | Load (s) | Memory after load (MB) | p50 (ms) | p95 (ms) "
        "| p99 (ms) | Throughput (reviews/s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        if "skipped" in row:
            lines.append(f"| {row['model']} | skipped: {row['skipped']} | | | | | | |")
            continue
        latency = row["latency_ms"]
        lines.append(
            f"| {row['model']} | {_fmt(row['disk_mb'])} | {_fmt(row['load_seconds'], 2)} "
            f"| {_fmt(row['memory_after_load_mb'])} | {_fmt(latency['p50'], 2)} "
            f"| {_fmt(latency['p95'], 2)} | {_fmt(latency['p99'], 2)} "
            f"| {_fmt(row['throughput_reviews_per_second'])} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ command line


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark model inference.")
    parser.add_argument("--models", default="tfidf,pretrained,finetuned")
    parser.add_argument("--n", type=int, default=200, help="number of test reviews")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, help="CSV with review_text (default: the test split)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--worker", choices=MODEL_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--texts-file", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return run_worker(args)

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        parser.error(f"unknown model(s) {unknown}, choose from {list(MODEL_NAMES)}")

    texts = load_texts(args.csv, args.n, args.seed)
    logger.info("Benchmarking %s on %d reviews", names, len(texts))

    results = []
    with tempfile.TemporaryDirectory() as work:
        texts_file = Path(work) / "texts.json"
        texts_file.write_text(json.dumps(texts), encoding="utf-8")
        for name in names:
            result_file = Path(work) / f"{name}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker", name,
                "--texts-file", str(texts_file),
                "--result-file", str(result_file),
                "--warmup", str(args.warmup),
                "--models-dir", str(args.models_dir),
            ]  # fmt: skip
            logger.info("Measuring %s in its own process", name)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0 or not result_file.exists():
                logger.error("Benchmark of %s failed (exit code %s)", name, completed.returncode)
                results.append({"model": name, "skipped": "the benchmark process failed"})
                continue
            results.append(json.loads(result_file.read_text(encoding="utf-8")))

    report = {
        "environment": environment(),
        "settings": {"n": len(texts), "warmup": args.warmup, "seed": args.seed},
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "inference_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (args.out_dir / "inference_benchmark.md").write_text(markdown, encoding="utf-8")
    for line in markdown.splitlines():
        logger.info(line)
    logger.info("Wrote %s", args.out_dir / "inference_benchmark.json")
    return 0 if any("skipped" not in row for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
