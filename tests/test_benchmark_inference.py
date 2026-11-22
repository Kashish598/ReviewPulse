import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "benchmark_inference", ROOT / "scripts" / "benchmark_inference.py"
)
bench = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_inference"] = bench
_spec.loader.exec_module(bench)


class CountingPredictor:
    name = "counting"

    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(len(texts))
        return []


def test_percentile_interpolates():
    values = [10, 20, 30, 40, 100]

    assert bench.percentile(values, 50) == 30
    assert bench.percentile(values, 100) == 100
    assert 40 < bench.percentile(values, 95) < 100


def test_latency_makes_one_call_per_review_and_does_not_count_the_warmup():
    predictor = CountingPredictor()

    timings = bench.measure_latency(predictor, ["a", "b", "c", "d"], warmup=2)

    assert len(timings) == 4
    assert predictor.calls == [1] * 6  # 2 warmup + 4 timed, always one review per call
    assert all(t >= 0 for t in timings)


def test_throughput_sends_everything_in_one_call():
    predictor = CountingPredictor()

    rate = bench.measure_throughput(predictor, ["a"] * 10)

    assert predictor.calls == [10]
    assert rate > 0


def test_dir_size_adds_up_files_in_subfolders(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 1024 * 1024)
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 1024 * 1024)

    assert bench.dir_size_mb(tmp_path) == pytest.approx(2.0)
    assert bench.dir_size_mb(tmp_path / "missing") is None
    assert bench.dir_size_mb(None) is None


def test_benchmark_model_reports_every_figure(monkeypatch):
    monkeypatch.setattr(bench, "rss_mb", lambda: None)  # as if psutil were not installed

    result = bench.benchmark_model("counting", CountingPredictor, ["a"] * 8, None, warmup=2)

    assert result["served_as"] == "counting"
    assert result["requests_timed"] == 8
    assert set(result["latency_ms"]) == {"mean", "p50", "p95", "p99"}
    assert result["latency_ms"]["p50"] <= result["latency_ms"]["p95"] <= result["latency_ms"]["p99"]
    assert result["load_seconds"] >= 0
    assert result["memory_after_load_mb"] is None  # n/a rather than an invented number
    assert result["throughput_reviews_per_second"] > 0


def test_memory_growth_is_measured_when_psutil_is_available(monkeypatch):
    readings = iter([100.0, 160.0, 175.0])
    monkeypatch.setattr(bench, "rss_mb", lambda: next(readings))

    result = bench.benchmark_model("counting", CountingPredictor, ["a"] * 3, None, warmup=1)

    assert result["memory_after_load_mb"] == 60.0
    assert result["memory_after_predict_mb"] == 75.0


def test_markdown_table_has_a_row_per_model_and_shows_skips():
    report = {
        "environment": {
            "measured_at": "2026-11-25T10:00:00+00:00",
            "platform": "TestOS",
            "cpu_count": 4,
            "python": "3.11.9",
            "torch": None,
        },
        "settings": {"n": 200},
        "results": [
            {
                "model": "tfidf",
                "disk_mb": 12.3,
                "load_seconds": 0.25,
                "memory_after_load_mb": None,
                "latency_ms": {"mean": 1, "p50": 1.5, "p95": 2.5, "p99": 3.5},
                "throughput_reviews_per_second": 900.0,
            },
            {"model": "finetuned", "skipped": "no model folder at models/x"},
        ],
    }

    table = bench.render_markdown(report)

    assert "| tfidf | 12.3 | 0.25 | n/a | 1.50 | 2.50 | 3.50 | 900.0 |" in table
    assert "| finetuned | skipped: no model folder at models/x |" in table
    assert "TestOS" in table


def test_model_folder_names_match_the_rest_of_the_project():
    from src.train_tfidf import TFIDF_MODEL_DIR

    assert bench.MODEL_DIRS["tfidf"] == TFIDF_MODEL_DIR.name
    bert_model = pytest.importorskip("src.bert_model")
    assert bench.MODEL_DIRS["pretrained"] == bert_model.PRETRAINED_DIR_NAME
    assert bench.MODEL_DIRS["finetuned"] == bert_model.FINETUNED_DIR_NAME


def write_csv(path, n=30):
    pd.DataFrame({"review_text": [f"review number {i} is fine" for i in range(n)]}).to_csv(
        path, index=False
    )
    return path


def test_command_line_measures_a_model_in_its_own_process_and_writes_the_reports(tmp_path):
    out = tmp_path / "docs"

    code = bench.main(
        [
            "--models", "mock",
            "--n", "20",
            "--warmup", "2",
            "--csv", str(write_csv(tmp_path / "reviews.csv")),
            "--out-dir", str(out),
        ]
    )  # fmt: skip

    report = json.loads((out / "inference_benchmark.json").read_text())
    assert code == 0
    assert report["settings"]["n"] == 20
    assert report["results"][0]["model"] == "mock"
    assert report["results"][0]["requests_timed"] == 20
    assert report["environment"]["python"]
    assert "| mock |" in (out / "inference_benchmark.md").read_text()


def test_a_model_that_is_not_there_is_skipped_not_fatal(tmp_path):
    out = tmp_path / "docs"

    code = bench.main(
        [
            "--models", "finetuned,mock",
            "--n", "10",
            "--csv", str(write_csv(tmp_path / "reviews.csv")),
            "--models-dir", str(tmp_path / "no_models"),
            "--out-dir", str(out),
        ]
    )  # fmt: skip

    results = {
        r["model"]: r for r in json.loads((out / "inference_benchmark.json").read_text())["results"]
    }
    assert code == 0
    assert "no model folder" in results["finetuned"]["skipped"]
    assert "skipped" not in results["mock"]


def test_unknown_model_name_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        bench.main(["--models", "gpt", "--out-dir", str(tmp_path)])
