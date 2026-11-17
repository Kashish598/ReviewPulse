import json

import pytest

from src.bert_model import (
    FINETUNED_DIR_NAME,
    MODEL_DIR_ENV,
    PRETRAINED_DIR_NAME,
    BertPredictor,
    resolve_model_dir,
)
from src.common import Prediction


@pytest.fixture
def predictor(tmp_path, monkeypatch):
    # Create a temporary model directory.
    model_dir = tmp_path / "nlptown_sentiment"
    model_dir.mkdir()

    # Simulate Hugging Face pipeline output.
    def fake_pipeline(*args, **kwargs):
        def classify(texts, **call_kwargs):
            return [
                {"label": "5 stars", "score": 0.95},
                {"label": "1 star", "score": 0.90},
                {"label": "3 stars", "score": 0.80},
            ][: len(texts)]

        return classify

    monkeypatch.setattr("src.bert_model.pipeline", fake_pipeline)

    return BertPredictor(model_dir)


def test_model_initialization(predictor):
    assert predictor.name == "nlptown-bert-multilingual"
    assert predictor.fine_tuned is False


@pytest.mark.parametrize(
    ("star_label", "expected"),
    [
        ("1 star", "negative"),
        ("2 stars", "negative"),
        ("3 stars", "negative"),
        ("4 stars", "positive"),
        ("5 stars", "positive"),
    ],
)
def test_star_mapping(star_label, expected):
    assert BertPredictor.map_stars_to_label(star_label) == expected


def test_invalid_star_label():
    with pytest.raises(ValueError):
        BertPredictor.map_stars_to_label("6 stars")


def test_predict_returns_prediction_objects(predictor):
    results = predictor.predict(["Great product", "Terrible product"])

    assert len(results) == 2
    assert all(isinstance(result, Prediction) for result in results)

    assert results[0].label == "positive"
    assert results[0].score == 0.95

    assert results[1].label == "negative"
    assert results[1].score == 0.90


def test_predict_empty_input(predictor):
    assert predictor.predict([]) == []


def test_missing_model_directory(tmp_path):
    missing_path = tmp_path / "missing_model"

    with pytest.raises(FileNotFoundError):
        BertPredictor(missing_path)


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr("src.bert_model.pipeline", lambda *args, **kwargs: None)


def test_metadata_marks_a_fine_tuned_model(tmp_path, fake_pipeline):
    (tmp_path / "metadata.json").write_text(json.dumps({"fine_tuned": True, "seed": 43}))

    predictor = BertPredictor(tmp_path)

    assert predictor.fine_tuned is True
    assert predictor.name == "nlptown-bert-finetuned"
    assert predictor.metadata["seed"] == 43


def test_metadata_for_the_pretrained_model(tmp_path, fake_pipeline):
    (tmp_path / "metadata.json").write_text(json.dumps({"fine_tuned": False}))

    predictor = BertPredictor(tmp_path)

    assert predictor.fine_tuned is False
    assert predictor.name == "nlptown-bert-multilingual"


@pytest.mark.parametrize("content", ["not json", "[1, 2]", ""])
def test_unreadable_metadata_is_ignored(tmp_path, fake_pipeline, content):
    (tmp_path / "metadata.json").write_text(content)

    predictor = BertPredictor(tmp_path)

    assert predictor.metadata == {}
    assert predictor.fine_tuned is False


def test_resolve_prefers_the_fine_tuned_model(tmp_path, monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    (tmp_path / FINETUNED_DIR_NAME).mkdir()
    (tmp_path / PRETRAINED_DIR_NAME).mkdir()

    assert resolve_model_dir(tmp_path) == tmp_path / FINETUNED_DIR_NAME


def test_resolve_falls_back_to_the_pretrained_model(tmp_path, monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    (tmp_path / PRETRAINED_DIR_NAME).mkdir()

    assert resolve_model_dir(tmp_path) == tmp_path / PRETRAINED_DIR_NAME


def test_resolve_uses_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path / "custom"))
    (tmp_path / FINETUNED_DIR_NAME).mkdir()

    assert resolve_model_dir(tmp_path) == tmp_path / "custom"


def test_resolve_explicit_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path / "from_env"))

    assert resolve_model_dir(tmp_path, override=tmp_path / "explicit") == tmp_path / "explicit"
