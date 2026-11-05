import joblib
import pytest

from src.common import Prediction
from src.tfidf_model import MODEL_FILE, TfidfPredictor
from src.train_tfidf import build_candidates

POSITIVE = [
    "great product love it",
    "love this great quality",
    "excellent product great value",
    "great love perfect",
    "perfect great product",
    "love great excellent",
]
NEGATIVE = [
    "bad product hate it",
    "hate this bad quality",
    "terrible product bad value",
    "bad hate awful",
    "awful bad product",
    "hate bad terrible",
]
TEXTS = POSITIVE + NEGATIVE
LABELS = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
TOY_PARAMS = {"min_df": 1}


@pytest.fixture
def model_dir(tmp_path):
    pipeline = build_candidates(TOY_PARAMS)["multinomial_nb"]
    pipeline.fit(TEXTS, LABELS)
    joblib.dump(pipeline, tmp_path / MODEL_FILE)
    return tmp_path


def test_predict_returns_prediction_objects(model_dir):
    results = TfidfPredictor(model_dir).predict(["great product", "awful product"])

    assert len(results) == 2
    assert all(isinstance(result, Prediction) for result in results)
    assert [result.label for result in results] == ["positive", "negative"]


def test_scores_are_probabilities(model_dir):
    results = TfidfPredictor(model_dir).predict(["great", "bad", "so so"])

    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_predict_empty_input(model_dir):
    assert TfidfPredictor(model_dir).predict([]) == []


def test_missing_model_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        TfidfPredictor(tmp_path / "missing_model")


@pytest.mark.parametrize("name", ["linear_svc", "multinomial_nb", "random_forest"])
def test_every_candidate_gives_probabilities(name):
    pipeline = build_candidates(TOY_PARAMS)[name]
    pipeline.fit(TEXTS, LABELS)

    probabilities = pipeline.predict_proba(["great product", "bad product"])

    assert probabilities.shape == (2, 2)
    assert list(pipeline.classes_) == [0, 1]
