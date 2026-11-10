import contextlib
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import src.finetune_scaling as ss
from src.common import write_predictions
from src.finetune_scaling import (
    RecordingTracker,
    aggregate_by_size,
    assign_jobs,
    build_report,
    compare_predictions,
    ensemble_by_size,
    exact_mcnemar_p,
    extend_train,
    make_jobs,
    replay,
    run_name,
    select_run,
    wilson_interval,
)

# ------------------------------------------------------------------ jobs


def test_make_jobs_is_the_product_of_sizes_and_seeds():
    jobs = make_jobs([10, 20], [1, 2, 3])

    assert len(jobs) == 6
    assert (20, 3) in jobs


@pytest.mark.parametrize("sizes, seeds", [([], [1]), ([10], []), ([10, 10], [1]), ([10], [1, 1])])
def test_make_jobs_rejects_empty_or_repeated_values(sizes, seeds):
    with pytest.raises(ValueError):
        make_jobs(sizes, seeds)


def test_assign_jobs_covers_every_job_once_and_balances_the_load():
    jobs = make_jobs([10_000, 20_000, 40_000], [42, 43, 44])

    parts = assign_jobs(jobs, 2)

    assert sorted(job for part in parts for job in part) == sorted(jobs)
    loads = [sum(size for size, _ in part) for part in parts]
    assert abs(loads[0] - loads[1]) <= 10_000
    for part in parts:
        assert part == sorted(part)  # small jobs first


def test_assign_jobs_with_one_worker_keeps_everything():
    jobs = make_jobs([5, 10], [1, 2])

    assert assign_jobs(jobs, 1) == [sorted(jobs)]


def test_assign_jobs_rejects_zero_workers():
    with pytest.raises(ValueError):
        assign_jobs([(1, 1)], 0)


# ---------------------------------------------------------- extended training set


def frame(prefix, start, n):
    return pd.DataFrame(
        {
            "row_id": [f"{prefix}{i}" for i in range(start, start + n)],
            "review_text": [f"review {prefix} {i}" for i in range(start, start + n)],
            "label": [i % 2 for i in range(start, start + n)],
        }
    )


@pytest.fixture
def splits():
    pool = frame("tr", 0, 200)
    base, validation = pool.iloc[:20], pool.iloc[20:30]
    test = frame("te", 0, 10)
    return pool, base.reset_index(drop=True), validation.reset_index(drop=True), test


def test_extend_train_keeps_base_first_and_adds_clean_extras(splits):
    pool, base, validation, test = splits

    extended = extend_train(base, pool, validation, test, 60, seed=1)

    assert len(extended) == 60
    assert extended.iloc[:20].equals(base)
    extra = extended.iloc[20:]
    assert not set(extra["row_id"]) & (set(base["row_id"]) | set(validation["row_id"]))
    assert not set(extra["review_text"]) & set(validation["review_text"])
    assert extended["review_text"].is_unique


def test_extend_train_excludes_text_that_appears_in_test(splits):
    pool, base, validation, test = splits
    test = test.copy()
    test.loc[0, "review_text"] = pool.loc[100, "review_text"]  # a pool review also in test

    extended = extend_train(base, pool, validation, test, 189, seed=1)

    assert pool.loc[100, "review_text"] not in set(extended["review_text"])


def test_extend_train_is_reproducible_and_seed_dependent(splits):
    pool, base, validation, test = splits

    a = extend_train(base, pool, validation, test, 60, seed=1)
    b = extend_train(base, pool, validation, test, 60, seed=1)
    c = extend_train(base, pool, validation, test, 60, seed=2)

    assert a.equals(b)
    assert not a.equals(c)


def test_extend_train_to_the_base_size_returns_the_base(splits):
    pool, base, validation, test = splits

    assert extend_train(base, pool, validation, test, 20).equals(base)


def test_extend_train_errors(splits):
    pool, base, validation, test = splits

    with pytest.raises(ValueError, match="smaller"):
        extend_train(base, pool, validation, test, 10)
    with pytest.raises(ValueError, match="available"):
        extend_train(base, pool, validation, test, 500)
    other_pool = pool.copy()
    other_pool.loc[0, "review_text"] = "changed"
    with pytest.raises(ValueError, match="does not come from this pool"):
        extend_train(base, other_pool, validation, test, 60)


# ------------------------------------------------------------------ statistics


def test_exact_mcnemar_matches_the_known_result():
    assert exact_mcnemar_p(16, 7) == pytest.approx(0.0932, abs=1e-3)
    assert exact_mcnemar_p(7, 16) == exact_mcnemar_p(16, 7)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 5) == 1.0
    assert exact_mcnemar_p(0, 12) < 0.001


def test_wilson_interval_matches_the_known_result():
    low, high = wilson_interval(566, 600)

    assert low == pytest.approx(0.9219, abs=5e-4)
    assert high == pytest.approx(0.9592, abs=5e-4)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def write_pred(path, rows):
    ids = [r[0] for r in rows]
    write_predictions(path, ids, [r[1] for r in rows], [r[2] for r in rows])


def test_compare_predictions_counts_fixed_and_broken(tmp_path):
    a = [("r1", 0, 1), ("r2", 1, 1), ("r3", 0, 0), ("r4", 1, 0)]  # a: wrong, right, right, wrong
    b = [("r1", 0, 0), ("r2", 1, 0), ("r3", 0, 0), ("r4", 1, 1)]  # b: right, wrong, right, right
    write_pred(tmp_path / "a.csv", a)
    write_pred(tmp_path / "b.csv", b)

    result = compare_predictions(tmp_path / "a.csv", tmp_path / "b.csv")

    assert result["n"] == 4
    assert result["fixed"] == 2
    assert result["broken"] == 1
    assert result["accuracy_a"] == pytest.approx(0.5)
    assert result["accuracy_b"] == pytest.approx(0.75)
    assert result["mcnemar_p"] == pytest.approx(1.0)


def test_compare_predictions_handles_missing_and_mismatched_files(tmp_path):
    write_pred(tmp_path / "a.csv", [("r1", 0, 0)])
    write_pred(tmp_path / "b.csv", [("r2", 0, 0)])

    assert compare_predictions(tmp_path / "a.csv", tmp_path / "nope.csv") is None
    with pytest.raises(ValueError, match="same reviews"):
        compare_predictions(tmp_path / "a.csv", tmp_path / "b.csv")


# ---------------------------------------------------------- aggregation and selection


def make_summary(size, seed, val, test, tmp_path=None, probs=None):
    name = run_name(size, seed)
    if tmp_path is not None and probs is not None:
        np.savez_compressed(tmp_path / f"{name}_probs.npz", val=probs[0], test=probs[1])
    return {
        "name": name,
        "train_size": size,
        "seed": seed,
        "best_epoch": 1,
        "train_seconds": 60.0,
        "val": {"accuracy": val, "macro_f1": val, "n": 4, "confusion_matrix": [[1, 0], [0, 1]]},
        "test": {"accuracy": test, "macro_f1": test, "n": 4, "confusion_matrix": [[1, 0], [0, 1]]},
    }


def test_aggregate_by_size_reports_mean_std_and_range():
    summaries = [
        make_summary(10, 1, 0.90, 0.88),
        make_summary(10, 2, 0.92, 0.90),
        make_summary(20, 1, 0.94, 0.93),
    ]

    rows = aggregate_by_size(summaries)

    assert [r["train_size"] for r in rows] == [10, 20]
    assert rows[0]["runs"] == 2
    assert rows[0]["val_accuracy_mean"] == pytest.approx(0.91)
    assert rows[0]["test_accuracy_std"] == pytest.approx(np.std([0.88, 0.90], ddof=1))
    assert (rows[0]["test_accuracy_min"], rows[0]["test_accuracy_max"]) == (0.88, 0.90)
    assert rows[1]["val_accuracy_std"] is None  # one seed: no spread


def test_select_run_uses_validation_only():
    summaries = [
        make_summary(10, 1, 0.94, 0.80),  # best validation, worst test: must still win
        make_summary(10, 2, 0.93, 0.99),
    ]

    assert select_run(summaries) == "scale_n10_seed1"


def stars(prediction_per_row):
    """(N, 5) probabilities: 0 means a confident 1 star (negative), 1 a confident 5 stars."""
    probs = np.zeros((len(prediction_per_row), 5), dtype=np.float32)
    for i, value in enumerate(prediction_per_row):
        probs[i, 4 if value else 0] = 1.0
    return probs


def test_ensemble_averages_probabilities_before_deciding(tmp_path):
    labels = [1, 0, 1]
    run_a = make_summary(10, 1, 0.5, 0.5, tmp_path, (stars([1, 0, 0]), stars([1, 0, 0])))
    run_b = make_summary(10, 2, 0.5, 0.5, tmp_path, (stars([1, 0, 1]), stars([1, 1, 1])))
    run_c = make_summary(10, 3, 0.5, 0.5, tmp_path, (stars([1, 1, 1]), stars([0, 0, 1])))

    rows = ensemble_by_size([run_a, run_b, run_c], tmp_path, labels, labels)

    assert len(rows) == 1
    # votes per row (positive of three): validation 3, 1, 2 -> 1, 0, 1 -> all correct
    assert rows[0]["val_accuracy"] == pytest.approx(1.0)
    # test: 2, 1, 2 -> 1, 0, 1 -> all correct
    assert rows[0]["test_accuracy"] == pytest.approx(1.0)


def test_ensemble_needs_at_least_two_seeds(tmp_path):
    only = make_summary(10, 1, 0.5, 0.5, tmp_path, (stars([1]), stars([1])))

    assert ensemble_by_size([only], tmp_path, [1], [1]) == []


def test_build_report_selects_and_summarises(tmp_path):
    summaries = [
        make_summary(10, 1, 0.90, 0.88),
        make_summary(20, 1, 0.95, 0.94),
    ]

    report = build_report(summaries, tmp_path, [1], [1])

    assert report["selected_run"] == "scale_n20_seed1"
    assert report["selected_test"]["accuracy"] == 0.94
    low, high = report["selected_test_wilson_95"]
    assert low < 0.94 < high
    assert len(report["by_size"]) == 2
    with pytest.raises(ValueError):
        build_report([], tmp_path, [1], [1])


# -------------------------------------------------------------------- recording


def test_recording_tracker_round_trips_into_a_real_tracker_interface():
    recorder = RecordingTracker()
    with recorder.run("ignored", {"stage": "x"}) as run:
        run.params({"learning_rate": 1e-5, "seed": 42})
        run.metrics({"val_accuracy": np.float32(0.5)}, step=0)
        run.metrics({"val_accuracy": 0.9}, 1)
        run.dict({"n": np.int64(3)}, "validation_metrics.json")
        run.artifact("ignored.txt")

    saved = json.loads(json.dumps(recorder.calls))  # must survive JSON
    seen = []

    class Sink:
        def params(self, values):
            seen.append(("params", values))

        def metrics(self, values, step=None):
            seen.append(("metrics", values, step))

        def dict(self, content, file_name):
            seen.append(("dict", content, file_name))

    replay(saved, Sink())

    assert seen == [
        ("params", {"learning_rate": "1e-05", "seed": "42"}),
        ("metrics", {"val_accuracy": 0.5}, 0),
        ("metrics", {"val_accuracy": 0.9}, 1),
        ("dict", {"n": 3}, "validation_metrics.json"),
    ]
    with pytest.raises(ValueError):
        replay([{"op": "nope"}], Sink())


# ------------------------------------------------------------ end to end with stand-ins


class FakeModel:
    def to(self, device):
        return self

    def save_pretrained(self, path):
        (path / "model.safetensors").write_text("weights")


class FakeTokenizer:
    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}")


class LoggingTracker:
    """Stand-in for `Tracker` that remembers runs."""

    runs: list = []

    def __init__(self, enabled, tracking_uri=None):
        self.current = None

    @contextlib.contextmanager
    def run(self, run_name, tags=None):
        self.current = {"name": run_name, "tags": tags, "params": {}, "metrics": {}, "files": []}
        LoggingTracker.runs.append(self.current)
        yield self

    def params(self, values):
        self.current["params"].update({k: str(v) for k, v in values.items()})

    def metrics(self, values, step=None):
        self.current["metrics"].update(values)

    def dict(self, content, file_name):
        pass

    def artifact(self, path):
        self.current["files"].append(str(path))


@pytest.fixture
def stand_ins(monkeypatch, tmp_path):
    fake_torch = types.SimpleNamespace(
        __version__="0",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: types.SimpleNamespace(type=name),
    )
    fake_transformers = types.SimpleNamespace(
        __version__="0",
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: FakeTokenizer()),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=lambda name: FakeModel()
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def make(n, offset=0):
        labels = [i % 2 for i in range(n)]
        texts = [("good " if label else "bad ") + str(i + offset) for i, label in enumerate(labels)]
        return pd.DataFrame(
            {"row_id": [f"x{i + offset}" for i in range(n)], "review_text": texts, "label": labels}
        )

    data = {"train": make(400), "validation": make(100, 1000), "test": make(60, 2000)}
    monkeypatch.setattr(ss, "load_split", lambda name, sample=False: data[name])
    monkeypatch.setattr(ss, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(ss, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(ss, "MODELS_DIR", tmp_path / "models")

    def fake_predict(model, tokenizer, texts, device, use_amp, batch_size=64):
        probabilities = np.zeros((len(texts), 5))
        for i, text in enumerate(texts):
            probabilities[i, 4 if text.startswith("good") else 0] = 1.0
        return probabilities

    trained = []

    def fake_train(config, train, validation, tokenizer, device, use_amp, seed, tracker, zero_shot):
        trained.append((len(train), seed))
        tracker.params({"seed": seed, "train_size": len(train)})
        tracker.metrics({"val_accuracy": 0.9, "val_macro_f1": 0.9}, 1)
        summary = {
            "macro_f1": 0.9,
            "accuracy": 0.9,
            "best_epoch": 1,
            "metrics": {},
            "train_seconds": 6.0,
        }
        return FakeModel(), summary

    monkeypatch.setattr(ss, "predict_probabilities", fake_predict)
    monkeypatch.setattr(ss, "train_one", fake_train)
    monkeypatch.setattr(ss, "Tracker", LoggingTracker)
    LoggingTracker.runs = []
    return types.SimpleNamespace(trained=trained, tmp_path=tmp_path, data=data)


def test_smoke_train_then_summarize(stand_ins):
    tmp_path = stand_ins.tmp_path

    ss.main(["train", "--smoke"])
    ss.main(["summarize", "--smoke"])

    assert sorted(stand_ins.trained) == [(80, 42), (80, 43), (160, 42), (160, 43)]
    results = tmp_path / "predictions" / "scaling_smoke"
    assert len(list(results.glob("*_summary.json"))) == 4
    final = tmp_path / "models" / "nlptown_finetuned_v2_smoke"
    assert (final / "model.safetensors").exists()
    assert (final / "tokenizer.json").exists()
    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["fine_tuned"] is True
    assert metadata["artifact_agreement_with_training_run"] == 1.0
    assert metadata["test_accuracy"] == 1.0
    # the other checkpoints are removed
    assert list((tmp_path / "models" / "scaling_runs_smoke").iterdir()) == []
    predictions = pd.read_csv(tmp_path / "predictions" / "bert_finetuned_v2_smoke_test.csv")
    assert list(predictions.columns) == ["row_id", "label_true", "pred", "score"]
    assert len(predictions) == len(stand_ins.data["test"])  # fewer rows than the smoke cap
    report = json.loads((results / "report.json").read_text())
    assert len(report["by_size"]) == 2
    assert LoggingTracker.runs == []  # smoke never logs to MLflow


def test_train_skips_finished_runs_and_splits_work_between_workers(stand_ins):
    args = ["train", "--sizes", "80", "160", "--seeds", "1", "2", "--workers", "2"]

    ss.main([*args, "--worker", "0"])
    first = list(stand_ins.trained)
    ss.main([*args, "--worker", "0"])  # nothing new to do
    ss.main([*args, "--worker", "1"])

    assert stand_ins.trained[: len(first)] == first
    assert len(stand_ins.trained) == 4
    assert len(set(stand_ins.trained)) == 4  # every (size, seed) once, across both workers
    ss.main([*args, "--worker", "1", "--force"])
    assert len(stand_ins.trained) == 4 + len(assign_jobs(make_jobs([80, 160], [1, 2]), 2)[1])


def test_train_rejects_a_bad_worker_index(stand_ins):
    with pytest.raises(SystemExit):
        ss.main(["train", "--worker", "2", "--workers", "2"])


def test_train_needs_the_extended_file_for_big_sizes(stand_ins):
    with pytest.raises(ValueError, match="prepare"):
        ss.main(["train", "--sizes", "1000", "--seeds", "1"])


def test_summarize_logs_every_run_and_the_selection_to_mlflow(stand_ins):
    tmp_path = stand_ins.tmp_path
    ss.main(["train", "--sizes", "80", "160", "--seeds", "1", "2"])

    ss.main(["summarize", "--tracking-uri", "sqlite:///unused.db"])

    names = [run["name"] for run in LoggingTracker.runs]
    assert names[:4] == [
        "scale_n80_seed1",
        "scale_n80_seed2",
        "scale_n160_seed1",
        "scale_n160_seed2",
    ]
    candidate = LoggingTracker.runs[0]
    assert candidate["tags"]["stage"] == "scaling_candidate"
    assert candidate["params"]["train_size"] == "80"  # replayed from the worker's record
    assert candidate["metrics"]["val_accuracy"] == 0.9
    assert candidate["metrics"]["test_accuracy"] == 1.0
    selected = LoggingTracker.runs[-1]
    assert selected["name"].startswith("scaling_selected_")
    assert selected["tags"]["stage"] == "scaling_selected"
    assert selected["metrics"]["test_accuracy"] == 1.0
    assert "n80_test_accuracy_mean" in selected["metrics"]
    assert "n160_ensemble_test_accuracy" in selected["metrics"]
    assert any(f.endswith("report.json") for f in selected["files"])
    assert (tmp_path / "predictions" / "bert_finetuned_v2_test.csv").exists()


def test_summarize_without_runs_fails_clearly(stand_ins):
    with pytest.raises(ValueError, match="No finished runs"):
        ss.main(["summarize", "--no-mlflow"])


def test_prepare_extends_the_real_splits_without_leaking(monkeypatch, tmp_path):
    from src.prepare_data import build_splits

    def raw(prefix, n):
        return pd.DataFrame(
            {
                "content": [f"{prefix} review number {i}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
                "title": ["t"] * n,
            }
        )

    train_raw, test_raw = raw("train", 400), raw("test", 60)
    splits, _ = build_splits(train_raw, test_raw, n_train=50, n_validation=20, n_test=10)
    monkeypatch.setattr(ss, "load_split", lambda name, sample=False: splits[name])
    monkeypatch.setattr(ss, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr("src.prepare_data.load_raw", lambda revision: (train_raw, test_raw, "rev"))

    ss.main(["prepare", "--max-size", "200"])

    extended = pd.read_csv(tmp_path / ss.EXTENDED_TRAIN_FILE)
    assert len(extended) == 200
    pd.testing.assert_frame_equal(
        extended.iloc[:50].reset_index(drop=True),
        splits["train"].reset_index(drop=True),
        check_dtype=False,
    )
    assert not set(extended["row_id"]) & set(splits["validation"]["row_id"])
    assert not set(extended["review_text"]) & set(splits["test"]["review_text"])
    assert extended["review_text"].is_unique
    report = json.loads((tmp_path / "train_extended_report.json").read_text())
    assert (report["rows"], report["base_rows"], report["extra_rows"]) == (200, 50, 150)
    # and the training step reads it back, first rows first
    assert ss.load_training_pool(120).iloc[:50].equals(splits["train"])
