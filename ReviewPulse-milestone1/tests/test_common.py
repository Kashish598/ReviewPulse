import pytest

from src.common import (
    MockPredictor,
    Prediction,
    Predictor,
    compute_metrics,
    read_predictions,
    to_label_name,
    write_predictions,
)


def test_to_label_name_accepts_ids_and_exact_names():
    assert to_label_name(0) == "negative"
    assert to_label_name(1) == "positive"
    assert to_label_name("positive") == "positive"


@pytest.mark.parametrize("bad", ["Positive", "pos", "neutral", "", 2, None, 0.5, " positive", True])
def test_to_label_name_rejects_everything_else(bad):
    with pytest.raises(ValueError):
        to_label_name(bad)


def test_compute_metrics_perfect():
    m = compute_metrics([0, 1, 0, 1], ["negative", "positive", "negative", "positive"])
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_metrics_known_values():
    # true:  0 0 0 1 ; pred: 0 0 1 1  -> accuracy 0.75
    m = compute_metrics([0, 0, 0, 1], [0, 0, 1, 1])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[2, 1], [0, 1]]
    assert m["n"] == 4


def test_compute_metrics_validates_input():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, "maybe"])


def test_predictions_file_roundtrip(tmp_path):
    path = write_predictions(
        tmp_path / "p.csv", ["te1", "te2"], [0, 1], ["negative", "positive"], [0.9, None]
    )
    df = read_predictions(path)
    assert list(df.columns) == ["row_id", "label_true", "pred", "score"]
    assert df["pred"].tolist() == ["negative", "positive"]
    assert df["score"].isna().tolist() == [False, True]


def test_write_predictions_rejects_invalid_pred(tmp_path):
    with pytest.raises(ValueError):
        write_predictions(tmp_path / "p.csv", ["te1"], [0], ["unclear"])


def test_mock_predictor_follows_interface():
    predictor = MockPredictor()
    assert isinstance(predictor, Predictor)
    out = predictor.predict(["I love it, great", "terrible, worst, broken"])
    assert [p.label for p in out] == ["positive", "negative"]


def test_prediction_rejects_bad_label():
    with pytest.raises(ValueError):
        Prediction(label="neutral")
