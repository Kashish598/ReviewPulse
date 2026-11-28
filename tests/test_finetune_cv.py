"""Tests for src/finetune_cv.py. No GPU and no model download: only the parts around training."""

import json

import numpy as np
import pytest

from src import finetune_cv as cv


def make_result(fold: int, macro_f1: float, zero: float = 0.90, epochs: int = 2) -> dict:
    """A fold result shaped like the JSON file the train step writes."""
    series = {
        "val_accuracy": [[0, zero]] + [[e, macro_f1] for e in range(1, epochs + 1)],
        "val_macro_f1": [[0, zero]] + [[e, macro_f1] for e in range(1, epochs + 1)],
        "train_loss": [[e, 0.3 / e] for e in range(1, epochs + 1)],
        "train_loss_step": [[50, 0.4], [100, 0.3]],
    }
    return {
        "config_name": "lr1e5_ep2",
        "config": {"name": "lr1e5_ep2", "learning_rate": 1e-5, "epochs": epochs},
        "fold": fold,
        "n_folds": 3,
        "cv_seed": 42,
        "train_size": 80,
        "validation_size": 40,
        "zero_shot": {"accuracy": zero, "macro_f1": zero},
        "final": {"accuracy": macro_f1, "macro_f1": macro_f1},
        "epochs": cv.epoch_table(series, epochs),
        "best_epoch_by_validation": epochs,
        "train_seconds": 60.0 + fold,
        "base_model": "base/model",
        "dataset_revision": "abc123",
        "recorded": {
            "parameters": {"learning_rate": "1e-05", "seed": "42"},
            "series": series,
            "final_metrics": {"best_val_macro_f1": macro_f1},
            "documents": {},
        },
    }


def write_results(out_dir, name="lr1e5_ep2", scores=(0.94, 0.95, 0.96)):
    cv.write_json(
        out_dir / "folds.json",
        {"n_folds": len(scores), "seed": 42, "train_pool_size": 120, "validation_row_ids": {}},
    )
    for k, score in enumerate(scores):
        result = make_result(k, score)
        result["n_folds"] = len(scores)
        cv.write_json(cv.result_path(out_dir, name, k), result)


# ----------------------------------------------------------------------------- folds


def test_folds_cover_every_row_once_and_do_not_overlap():
    labels = np.array([0] * 60 + [1] * 40)
    folds = cv.make_folds(labels, 5, seed=42)
    assert len(folds) == 5
    held_out = sorted(i for f in folds for i in f["validation_index"])
    assert held_out == list(range(100))
    for f in folds:
        assert set(f["train_index"]).isdisjoint(f["validation_index"])
        assert len(f["train_index"]) + len(f["validation_index"]) == 100


def test_folds_keep_the_label_balance():
    labels = np.array([0] * 60 + [1] * 40)
    for f in cv.make_folds(labels, 5, seed=1):
        share = labels[f["validation_index"]].mean()
        assert share == pytest.approx(0.4, abs=0.001)


def test_folds_are_repeatable_and_depend_on_the_seed():
    labels = np.array([0, 1] * 50)
    assert cv.make_folds(labels, 4, seed=7) == cv.make_folds(labels, 4, seed=7)
    assert cv.make_folds(labels, 4, seed=7) != cv.make_folds(labels, 4, seed=8)


def test_folds_need_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        cv.make_folds(np.array([0, 1] * 10), 1, seed=0)


# --------------------------------------------------------------------- numbers and logs


def test_summarize_values_uses_the_sample_standard_deviation():
    summary = cv.summarize_values([0.9, 0.95, 1.0])
    assert summary["mean"] == pytest.approx(0.95)
    assert summary["std"] == pytest.approx(0.05)
    assert summary["min"] == 0.9 and summary["max"] == 1.0 and summary["n"] == 3
    assert cv.summarize_values([0.5])["std"] == 0.0
    with pytest.raises(ValueError):
        cv.summarize_values([])


def test_recorder_keeps_steps_and_unstepped_values_apart():
    rec = cv.Recorder()
    rec.params({"seed": 42})
    rec.metrics({"val_accuracy": 0.9}, step=0)
    rec.metrics({"val_accuracy": 0.95}, step=1)
    rec.metrics({"best_epoch": 1})
    rec.dict({"a": 1}, "x.json")
    assert rec.parameters == {"seed": "42"}
    assert rec.series["val_accuracy"] == [[0, 0.9], [1, 0.95]]
    assert rec.final_metrics == {"best_epoch": 1.0}
    assert rec.documents["x.json"] == {"a": 1}
    json.dumps(rec.to_json())  # must be saveable


def test_final_epoch_score_ignores_the_untrained_step_and_earlier_epochs():
    series = {
        "val_accuracy": [[0, 0.80], [1, 0.90], [2, 0.93]],
        "val_macro_f1": [[0, 0.79], [1, 0.89], [2, 0.92]],
    }
    assert cv.final_epoch_scores(series, 2) == {"accuracy": 0.93, "macro_f1": 0.92}
    assert cv.final_epoch_scores(series, 1)["accuracy"] == 0.90
    with pytest.raises(ValueError, match="epoch 3"):
        cv.final_epoch_scores(series, 3)


def test_epoch_table_has_one_row_per_epoch():
    series = {
        "val_accuracy": [[0, 0.8], [1, 0.9], [2, 0.93]],
        "val_macro_f1": [[1, 0.89], [2, 0.92]],
        "train_loss": [[1, 0.4], [2, 0.2]],
    }
    rows = cv.epoch_table(series, 2)
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[1] == {"epoch": 2, "val_accuracy": 0.93, "val_macro_f1": 0.92, "train_loss": 0.2}


def test_aggregate_reports_the_gain_over_the_untrained_model():
    summary = cv.aggregate([make_result(0, 0.94, zero=0.90), make_result(1, 0.96, zero=0.92)])
    assert summary["n_folds"] == 2
    assert summary["val_macro_f1"]["mean"] == pytest.approx(0.95)
    assert summary["zero_shot_val_macro_f1"]["mean"] == pytest.approx(0.91)
    assert summary["gain_macro_f1"]["mean"] == pytest.approx(0.04)
    with pytest.raises(ValueError):
        cv.aggregate([])


def test_the_two_recipes_differ_only_in_epochs():
    one, two = cv.CV_CONFIGS["lr1e5_ep1"], cv.CV_CONFIGS["lr1e5_ep2"]
    assert (one.epochs, two.epochs) == (1, 2)
    assert one.learning_rate == two.learning_rate
    assert one.batch_size == two.batch_size and one.warmup_ratio == two.warmup_ratio


def test_saved_json_survives_numpy_numbers(tmp_path):
    path = tmp_path / "x.json"
    cv.write_json(path, {"a": np.float32(0.5), "b": np.int64(3), "c": np.array([1, 2])})
    assert json.loads(path.read_text()) == {"a": 0.5, "b": 3, "c": [1, 2]}


def test_fold_results_load_in_fold_order(tmp_path):
    write_results(tmp_path, scores=(0.9, 0.91, 0.92))
    loaded = cv.load_fold_results(tmp_path, "lr1e5_ep2")
    assert [r["fold"] for r in loaded] == [0, 1, 2]


# ---------------------------------------------------------------------------- MLflow


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts next to the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "tester")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def runs_of(uri, experiment):
    import mlflow

    mlflow.set_tracking_uri(uri)
    return mlflow.search_runs(experiment_names=[experiment])


def test_log_creates_one_parent_run_with_one_nested_run_per_fold(tmp_path, local_mlflow):
    write_results(tmp_path / "cv", scores=(0.94, 0.95, 0.96))
    rows = cv.log_results(tmp_path / "cv", local_mlflow)

    assert len(rows) == 1 and rows[0]["recipe"] == "lr1e5_ep2" and rows[0]["folds"] == 3
    runs = runs_of(local_mlflow, cv.CV_EXPERIMENT)
    parents = runs[runs["tags.mlflow.runName"] == "cv_lr1e5_ep2"]
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert len(parents) == 1 and len(children) == 3
    parent = parents.iloc[0]
    assert children["tags.mlflow.parentRunId"].eq(parent["run_id"]).all()
    assert parent["metrics.cv_val_macro_f1_mean"] == pytest.approx(0.95)
    assert parent["metrics.cv_val_macro_f1_std"] == pytest.approx(0.01)
    assert parent["metrics.cv_gain_macro_f1_mean"] == pytest.approx(0.05)
    assert parent["tags.logged_by"] == "tester"
    assert parent["params.n_folds"] == "3"
    assert set(children["tags.fold"]) == {"0", "1", "2"}


def test_fold_runs_keep_the_per_epoch_curve(tmp_path, local_mlflow):
    import mlflow

    write_results(tmp_path / "cv", scores=(0.94, 0.95, 0.96))
    cv.log_results(tmp_path / "cv", local_mlflow)
    runs = runs_of(local_mlflow, cv.CV_EXPERIMENT)
    fold0 = runs[runs["tags.mlflow.runName"] == "cv_lr1e5_ep2_fold0"].iloc[0]
    history = mlflow.MlflowClient().get_metric_history(fold0["run_id"], "val_macro_f1")
    assert [(m.step, round(m.value, 2)) for m in history] == [(0, 0.9), (1, 0.94), (2, 0.94)]
    assert fold0["metrics.final_val_macro_f1"] == pytest.approx(0.94)


def test_logging_twice_does_not_duplicate_runs(tmp_path, local_mlflow):
    write_results(tmp_path / "cv")
    cv.log_results(tmp_path / "cv", local_mlflow)
    assert cv.log_results(tmp_path / "cv", local_mlflow) == []
    runs = runs_of(local_mlflow, cv.CV_EXPERIMENT)
    assert (runs["tags.mlflow.runName"] == "cv_lr1e5_ep2").sum() == 1
    again = cv.log_results(tmp_path / "cv", local_mlflow, allow_duplicates=True)
    assert len(again) == 1


def test_unfinished_recipes_are_skipped_unless_allowed(tmp_path, local_mlflow):
    out = tmp_path / "cv"
    write_results(out, scores=(0.94, 0.95, 0.96))
    cv.result_path(out, "lr1e5_ep2", 2).unlink()  # fold 2 never finished
    assert cv.log_results(out, local_mlflow) == []
    rows = cv.log_results(out, local_mlflow, allow_partial=True)
    assert rows[0]["folds"] == 2


def test_log_needs_the_train_step_first(tmp_path, local_mlflow):
    with pytest.raises(FileNotFoundError, match="train"):
        cv.log_results(tmp_path / "nothing", local_mlflow)
