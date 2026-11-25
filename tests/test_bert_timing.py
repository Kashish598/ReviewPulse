"""Stage timing in the BERT adapter, with a fake pipeline (no model files needed)."""

import time

import pytest

import src.bert_model as bert_model
from src.bert_model import BertPredictor


class FakeClassifier:
    def __init__(self, with_tokenizer=True, tokenizer_fails=False):
        if with_tokenizer:
            self.tokenizer = self._tokenize
        self.tokenizer_fails = tokenizer_fails

    def _tokenize(self, texts, **kwargs):
        if self.tokenizer_fails:
            raise RuntimeError("tokenizer broke")
        time.sleep(0.002)

    def __call__(self, texts, **kwargs):
        time.sleep(0.005)
        return [{"label": "5 stars", "score": 0.9} for _ in texts]


@pytest.fixture
def make_predictor(monkeypatch, tmp_path):
    def build(**fake_kwargs):
        monkeypatch.setattr(bert_model, "pipeline", lambda *a, **k: FakeClassifier(**fake_kwargs))
        return BertPredictor(tmp_path)

    return build


def test_predict_timed_returns_the_predictions_and_all_stage_times(make_predictor):
    predictions, stages = make_predictor().predict_timed(["good", "great"])

    assert [p.label for p in predictions] == ["positive", "positive"]
    assert set(stages) == {"tokenize_ms", "inference_ms", "label_mapping_ms", "total_ms"}
    assert stages["tokenize_ms"] > 0
    assert stages["inference_ms"] >= 0
    assert stages["label_mapping_ms"] >= 0
    parts = stages["tokenize_ms"] + stages["inference_ms"] + stages["label_mapping_ms"]
    assert stages["total_ms"] == pytest.approx(parts)


def test_predict_still_returns_only_the_predictions(make_predictor):
    predictions = make_predictor().predict(["good"])

    assert len(predictions) == 1
    assert predictions[0].label == "positive"
    assert predictions[0].score == 0.9


def test_a_pipeline_without_a_tokenizer_attribute_still_works(make_predictor):
    predictions, stages = make_predictor(with_tokenizer=False).predict_timed(["good"])

    assert predictions[0].label == "positive"
    assert stages["tokenize_ms"] == 0.0


def test_a_failing_tokenizer_timing_never_breaks_the_prediction(make_predictor):
    predictions, stages = make_predictor(tokenizer_fails=True).predict_timed(["good"])

    assert predictions[0].label == "positive"
    assert stages["tokenize_ms"] == 0.0


def test_empty_input_gives_no_predictions_and_zero_times(make_predictor):
    predictions, stages = make_predictor().predict_timed([])

    assert predictions == []
    assert stages["total_ms"] == 0.0
