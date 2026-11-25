"""Retraining on the original data plus labelled requests (M4). Small toy data, no network."""

import json

import pandas as pd
import pytest

from src import feedback_store, retrain
from src.feedback_store import FeedbackStore
from src.retrain import (
    BASELINE_NAME,
    CANDIDATES,
    NewData,
    RetrainError,
    original_texts,
    prepare_new_data,
    rows_from_csv,
    run_retraining,
)
from src.tfidf_model import TfidfPredictor

POSITIVE = ["great product love it", "love this great quality", "excellent product great value"]
NEGATIVE = ["bad product hate it", "hate this bad quality", "terrible product bad value"]


def frame(positive, negative):
    texts = list(positive) + list(negative)
    labels = [1] * len(positive) + [0] * len(negative)
    return pd.DataFrame({"row_id": range(len(texts)), "review_text": texts, "label": labels})


def numbered(words, count, start=0):
    return [f"{words} number {start + i} item" for i in range(count)]


@pytest.fixture
def original():
    """Toy original splits: train, validation and test, all about great/bad."""
    train = frame(numbered("great love product", 15), numbered("bad hate product", 15))
    validation = frame(numbered("great love quality", 6, 100), numbered("bad hate quality", 6, 100))
    test = frame(numbered("great love value", 4, 200), numbered("bad hate value", 4, 200))
    return train, validation, test


@pytest.fixture
def toy_splits(original, monkeypatch):
    train, validation, test = original
    monkeypatch.setattr(
        "src.retrain.load_split",
        lambda name, sample=False: {"train": train, "validation": validation, "test": test}[name],
    )
    return train, validation, test


def request_rows(positive_words, negative_words, per_class, split, start=0):
    rows = []
    for i in range(per_class):
        rows.append(
            {
                "text": f"{positive_words} request {start + i}",
                "llm_label": "positive",
                "split": split,
            }
        )
        rows.append(
            {
                "text": f"{negative_words} request {start + i}",
                "llm_label": "negative",
                "split": split,
            }
        )
    return rows


# ------------------------------------------------------------------ preparing the new data


def test_labels_are_mapped_and_the_split_column_is_used():
    rows = [
        {"text": "good one", "llm_label": "positive", "split": "train"},
        {"text": "bad one", "llm_label": "negative", "split": "validation"},
    ]

    new = prepare_new_data(rows, set())

    assert new.train.to_dict("records") == [{"review_text": "good one", "label": 1}]
    assert new.validation.to_dict("records") == [{"review_text": "bad one", "label": 0}]
    assert new.report["used_train"] == 1 and new.report["used_validation"] == 1


def test_a_missing_split_is_assigned_from_the_text_like_the_store_does():
    text = "a review without a split"
    expected = feedback_store.assign_split(feedback_store.text_hash(text))

    new = prepare_new_data([{"text": text, "llm_label": "positive", "split": None}], set())

    assert len(new.validation if expected == "validation" else new.train) == 1


@pytest.mark.parametrize("bad", ["neutral", "Positive", "", None, "2", 7, "positive review"])
def test_a_label_that_is_not_exactly_negative_or_positive_is_dropped(bad):
    new = prepare_new_data([{"text": "some text", "llm_label": bad, "split": "train"}], set())

    assert len(new.train) == 0
    assert new.report["dropped_bad_label"] == 1


def test_empty_duplicate_and_known_texts_are_dropped_and_counted():
    rows = [
        {"text": "  ", "llm_label": "positive", "split": "train"},
        {"text": None, "llm_label": "positive", "split": "train"},
        {"text": "fresh review", "llm_label": "positive", "split": "train"},
        {"text": " fresh review ", "llm_label": "positive", "split": "train"},
        {"text": "already in the original data", "llm_label": "negative", "split": "train"},
    ]

    new = prepare_new_data(rows, {"already in the original data"})

    assert new.train["review_text"].tolist() == ["fresh review"]
    assert new.report["dropped_empty"] == 2
    assert new.report["dropped_duplicate"] == 1
    assert new.report["dropped_in_original_data"] == 1


def test_a_test_review_can_never_reach_training(original):
    train, validation, test = original
    leaked = test["review_text"].iloc[0]
    rows = [{"text": leaked, "llm_label": "positive", "split": "train"}]

    new = prepare_new_data(rows, original_texts(train, validation, test))

    assert len(new.train) == 0 and len(new.validation) == 0
    assert new.report["dropped_in_original_data"] == 1


# ------------------------------------------------------------------ where the rows come from


def test_rows_come_from_a_csv(tmp_path):
    path = tmp_path / "requests.csv"
    pd.DataFrame(
        {
            "text": ["a", "b"],
            "llm_label": ["positive", "negative"],
            "split": ["train", "validation"],
        }
    ).to_csv(path, index=False)

    rows = rows_from_csv(path)

    assert [r["text"] for r in rows] == ["a", "b"]
    assert [r["split"] for r in rows] == ["train", "validation"]


def test_a_csv_may_use_label_instead_of_llm_label_and_have_no_split(tmp_path):
    path = tmp_path / "requests.csv"
    pd.DataFrame({"text": ["a"], "label": ["positive"]}).to_csv(path, index=False)

    rows = rows_from_csv(path)

    assert rows[0]["llm_label"] == "positive"
    assert rows[0]["split"] is None


def test_a_csv_without_the_needed_columns_is_refused(tmp_path):
    path = tmp_path / "requests.csv"
    pd.DataFrame({"review": ["a"]}).to_csv(path, index=False)

    with pytest.raises(RetrainError, match="needs the columns"):
        rows_from_csv(path)


def test_a_missing_csv_is_refused(tmp_path):
    with pytest.raises(RetrainError, match="not found"):
        rows_from_csv(tmp_path / "nope.csv")


def test_without_a_database_url_the_store_is_not_read(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RetrainError, match="DATABASE_URL"):
        retrain.rows_from_store()


def test_labelled_requests_are_read_from_the_store(tmp_path, monkeypatch):
    store = FeedbackStore(f"sqlite:///{tmp_path / 'feedback.db'}")
    first = store.add_request("loved it", "positive", 0.9, "m", "v1")
    store.add_request("still waiting", "positive", 0.9, "m", "v1")  # never labelled
    store.mark_labelled(first, "positive", "llm", "v1")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///unused.db")
    feedback_store.set_store(store)
    try:
        rows = retrain.rows_from_store()
    finally:
        feedback_store.set_store(None)
        store.dispose()

    assert [r["text"] for r in rows] == ["loved it"]
    assert rows[0]["llm_label"] == "positive"


# ------------------------------------------------------------------ candidates


def test_there_are_at_least_three_distinct_candidates():
    names = [candidate.name for candidate in CANDIDATES]

    assert len(names) >= 3
    assert len(set(names)) == len(names)


def test_every_candidate_builds_a_pipeline_that_gives_probabilities():
    texts, labels = POSITIVE * 3 + NEGATIVE * 3, [1] * 9 + [0] * 9
    for candidate in CANDIDATES:
        pipeline = retrain.build_pipeline(candidate)
        pipeline.set_params(tfidf__min_df=1)
        pipeline.fit(texts, labels)
        assert pipeline.predict_proba(["great product"]).shape == (1, 2)


def test_unknown_candidates_are_refused():
    with pytest.raises(RetrainError, match="Unknown candidate"):
        retrain.pick_candidates(["not_a_model"])


def test_candidates_can_be_picked_by_name():
    picked = retrain.pick_candidates(["nb_bigram", "nb_unigram"])

    assert [c.name for c in picked] == ["nb_bigram", "nb_unigram"]


# ------------------------------------------------------------------ the whole run


def make_new(original):
    """New requests that teach two words the original data never saw."""
    new_rows = request_rows("superb wonderful", "dreadful awful", 10, "train")
    new_rows += request_rows("superb wonderful", "dreadful awful", 6, "validation", start=500)
    return prepare_new_data(new_rows, original_texts(*original))


def test_a_run_trains_every_candidate_saves_it_and_writes_the_summary(original, tmp_path):
    train, validation, _ = original
    new = make_new(original)

    summary = run_retraining(
        train,
        validation,
        new,
        retrain.pick_candidates(["nb_bigram", "nb_unigram", "svc_unigram_c0.5"]),
        tmp_path,
        track=False,
        source="test",
    )

    assert [c["name"] for c in summary["candidates"]] == [
        "nb_bigram",
        "nb_unigram",
        "svc_unigram_c0.5",
    ]
    assert summary["baseline"]["name"] == BASELINE_NAME
    assert summary["n_train_original"] == len(train)
    assert summary["n_new_train"] == 20
    assert summary["new_data"]["used_validation"] == 12
    assert "test split has not been used" in summary["note"]
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["best_on_validation"] == summary["best_on_validation"]
    best = max(summary["candidates"], key=lambda c: (c["val_macro_f1"], c["val_accuracy"]))
    assert summary["best_on_validation"] == best["name"]


def test_a_saved_candidate_loads_like_any_other_tfidf_model(original, tmp_path):
    train, validation, _ = original
    summary = run_retraining(
        train,
        validation,
        make_new(original),
        retrain.pick_candidates(["nb_bigram"]),
        tmp_path,
        track=False,
    )

    predictor = TfidfPredictor(summary["candidates"][0]["path"])

    assert predictor.name == "tfidf-nb_bigram"
    assert predictor.predict(["great product"])[0].label == "positive"
    metadata = json.loads((tmp_path / "nb_bigram" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_new_train"] == 20 and metadata["stage"] == "retrain_candidate"


def test_the_new_requests_teach_the_model_words_the_original_data_never_had(original, tmp_path):
    train, validation, _ = original

    summary = run_retraining(
        train,
        validation,
        make_new(original),
        retrain.pick_candidates(["nb_bigram"]),
        tmp_path,
        track=False,
    )

    baseline, candidate = summary["baseline"], summary["candidates"][0]
    assert candidate["new_val_accuracy"] > baseline["new_val_accuracy"]
    assert candidate["new_val_accuracy"] >= 0.9


def test_the_extra_validation_is_not_reported_when_there_are_too_few_new_rows(original, tmp_path):
    train, validation, _ = original
    few = prepare_new_data(
        request_rows("superb", "dreadful", 2, "validation")
        + request_rows("superb", "dreadful", 6, "train"),
        original_texts(*original),
    )

    summary = run_retraining(
        train, validation, few, retrain.pick_candidates(["nb_bigram"]), tmp_path, track=False
    )

    assert summary["candidates"][0]["new_val_accuracy"] is None


def test_the_new_validation_rows_are_never_trained_on(original, tmp_path, monkeypatch):
    train, validation, _ = original
    new = make_new(original)
    trained_on = []
    real = retrain.fit_and_score

    def spying(candidate, training, validation_frame, new_validation):
        trained_on.append(set(training["review_text"]))
        return real(candidate, training, validation_frame, new_validation)

    monkeypatch.setattr("src.retrain.fit_and_score", spying)

    run_retraining(
        train, validation, new, retrain.pick_candidates(["nb_bigram"]), tmp_path, track=False
    )

    held_out = set(new.validation["review_text"])
    assert len(trained_on) == 2  # the baseline and the candidate
    assert held_out and all(not (texts & held_out) for texts in trained_on)
    assert all(not (texts & set(validation["review_text"])) for texts in trained_on)
    assert set(new.train["review_text"]) <= trained_on[1]  # the candidate did get the new rows
    assert not (set(new.train["review_text"]) & trained_on[0])  # the baseline did not


def test_every_candidate_and_the_baseline_are_logged_to_mlflow(original, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    train, validation, _ = original
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.create_experiment(
        retrain.MLFLOW_RETRAIN_EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    try:
        summary = run_retraining(
            train,
            validation,
            make_new(original),
            retrain.pick_candidates(["nb_bigram", "nb_unigram", "svc_unigram_c0.5"]),
            tmp_path / "models",
            track=True,
        )

        experiment = mlflow.get_experiment_by_name(retrain.MLFLOW_RETRAIN_EXPERIMENT)
        runs = mlflow.search_runs([experiment.experiment_id])
        assert len(runs) == 4  # three candidates and the baseline
        assert set(runs["tags.stage"]) == {"candidate", "baseline"}
        candidate = runs[runs["params.candidate"] == "nb_bigram"].iloc[0]
        assert float(candidate["metrics.val_macro_f1"]) > 0
        assert candidate["params.n_new_train"] == "20"
        run_id = summary["candidates"][0]["run_id"]
        assert run_id == candidate["run_id"]
        names = [item.path for item in mlflow.MlflowClient().list_artifacts(run_id, "model")]
        assert any(name.endswith("model.joblib") for name in names)
    finally:
        mlflow.set_tracking_uri(previous_uri)


# ------------------------------------------------------------------ the command line


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_the_command_line_retrains_from_a_csv(toy_splits, tmp_path):
    rows = request_rows("superb wonderful", "dreadful awful", 12, "train") + request_rows(
        "superb wonderful", "dreadful awful", 6, "validation", start=500
    )
    csv = write_csv(tmp_path / "requests.csv", rows)

    code = retrain.main(
        [
            "--new-data",
            str(csv),
            "--no-mlflow",
            "--min-new",
            "20",
            "--candidates",
            "nb_bigram,nb_unigram,svc_unigram_c0.5",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["new_data"]["used_train"] == 24
    assert summary["new_data"]["received"] == 36
    assert (tmp_path / "out" / "nb_bigram" / "model.joblib").exists()


def test_the_command_line_stops_when_there_is_too_little_new_data(toy_splits, tmp_path, caplog):
    csv = write_csv(tmp_path / "requests.csv", request_rows("superb", "dreadful", 3, "train"))

    code = retrain.main(["--new-data", str(csv), "--no-mlflow", "--out-dir", str(tmp_path / "out")])

    assert code == 2
    assert "need at least 50" in caplog.text
    assert not (tmp_path / "out" / "summary.json").exists()


def test_the_command_line_reports_a_missing_split_file(monkeypatch, tmp_path, caplog):
    def missing(name, sample=False):
        raise FileNotFoundError(f"{name}.csv not found. Run `python -m src.prepare_data` first.")

    monkeypatch.setattr("src.retrain.load_split", missing)
    csv = write_csv(tmp_path / "requests.csv", request_rows("superb", "dreadful", 30, "train"))

    code = retrain.main(["--new-data", str(csv), "--no-mlflow"])

    assert code == 2
    assert "prepare_data" in caplog.text


def test_the_test_split_is_read_only_to_exclude_its_texts(toy_splits, tmp_path, monkeypatch):
    train, validation, test = toy_splits
    leaked = test["review_text"].iloc[0]
    rows = request_rows("superb wonderful", "dreadful awful", 15, "train")
    rows.append({"text": leaked, "llm_label": "positive", "split": "train"})
    csv = write_csv(tmp_path / "requests.csv", rows)

    code = retrain.main(
        [
            "--new-data",
            str(csv),
            "--no-mlflow",
            "--min-new",
            "20",
            "--candidates",
            "nb_bigram",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert summary["new_data"]["dropped_in_original_data"] == 1
    assert summary["new_data"]["used_train"] == 30


def test_newdata_is_a_plain_container():
    empty = pd.DataFrame(columns=["review_text", "label"])

    assert NewData(empty, empty, {}).report == {}
