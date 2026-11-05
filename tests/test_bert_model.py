import pytest

from src.bert_model import BertPredictor
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
