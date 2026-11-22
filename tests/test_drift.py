import importlib.util
import json
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import drift

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_drift_reference", ROOT / "scripts" / "build_drift_reference.py"
)
build_script = importlib.util.module_from_spec(_spec)
sys.modules["build_drift_reference"] = build_script
_spec.loader.exec_module(build_script)

VOCAB_SIZE = 3000
WORDS = [f"word{i}" for i in range(VOCAB_SIZE)]
ZIPF = 1 / np.arange(1, VOCAB_SIZE + 1)
ZIPF = ZIPF / ZIPF.sum()


def english_reviews(n: int, seed: int) -> list[str]:
    """Synthetic reviews: Zipf-distributed words, lengths spread like real reviews."""
    rng = np.random.default_rng(seed)
    lengths = np.clip(rng.lognormal(mean=3.6, sigma=0.6, size=n).astype(int), 5, 400)
    return [" ".join(rng.choice(WORDS, size=int(k), p=ZIPF)) + "." for k in lengths]


def labels(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, 2, size=n).tolist()


def rows_for(texts, preds, vocab):
    return [drift.feature_row(t, p, vocab) for t, p in zip(texts, preds, strict=True)]


@pytest.fixture(scope="module")
def reference_data():
    return drift.build_reference(
        english_reviews(3000, seed=1), labels(3000, seed=2), vocab_size=500, source="synthetic"
    )


@pytest.fixture
def reference(reference_data, tmp_path):
    path = drift.save_reference(reference_data, tmp_path / "reference.json")
    return drift.load_reference(path)


def make_detector(reference, **kwargs):
    return drift.DriftDetector(reference, **kwargs)


def feed(detector, texts, preds=None):
    preds = preds or ["positive" if i % 2 else "negative" for i in range(len(texts))]
    report = None
    for text, pred in zip(texts, preds, strict=True):
        report = detector.observe(text, pred)
    return report


# ---------------------------------------------------------------- features


def test_text_features_for_plain_english():
    vocab = {"great", "product"}

    features = drift.text_features("Great product", vocab)

    assert features == {"chars": 13.0, "words": 2.0, "non_ascii_ratio": 0.0, "oov_rate": 0.0}


def test_text_features_detect_other_scripts_and_unknown_words():
    vocab = {"great"}

    urdu = drift.text_features("یہ پروڈکٹ بہت اچھی ہے", vocab)
    mixed = drift.text_features("great stuff", vocab)

    assert urdu["non_ascii_ratio"] > 0.7
    assert urdu["oov_rate"] == 1.0
    assert mixed["oov_rate"] == 0.5


def test_text_features_handle_empty_and_symbol_only_text():
    assert drift.text_features("", {"a"})["oov_rate"] == 1.0
    assert drift.text_features("", {"a"})["non_ascii_ratio"] == 0.0
    assert drift.text_features("👍👍👍", {"a"})["non_ascii_ratio"] == 1.0


def test_positive_feature_follows_the_prediction():
    assert drift.feature_row("x", "positive", set())["positive"] == 1.0
    assert drift.feature_row("x", "negative", set())["positive"] == 0.0


# ---------------------------------------------------------------- statistics


def test_proportions_put_edge_values_in_the_lower_bin():
    shares = drift.proportions([0.0, 0.02, 0.1, 0.5], [0.0, 0.02, 0.2])

    assert list(shares) == [0.25, 0.25, 0.25, 0.25]


def test_psi_is_zero_for_identical_shares_and_grows_with_the_shift():
    same = drift.psi([0.5, 0.5], [0.5, 0.5])
    small = drift.psi([0.5, 0.5], [0.6, 0.4])
    large = drift.psi([0.5, 0.5], [0.9, 0.1])

    assert same == pytest.approx(0.0)
    assert 0 < small < 0.25 < large


def test_psi_survives_empty_bins():
    value = drift.psi([0.2, 0.8, 0.0], [0.0, 0.5, 0.5])

    assert np.isfinite(value) and value > 0.25


# ---------------------------------------------------------------- reference


def test_reference_contents(reference_data):
    assert reference_data["n_reviews"] == 3000
    assert len(reference_data["vocab"]) == 500
    assert set(reference_data["features"]) == set(drift.FEATURES)
    for spec in reference_data["features"].values():
        assert spec["edges"] == sorted(spec["edges"])
        assert len(spec["proportions"]) == len(spec["edges"]) + 1
        assert sum(spec["proportions"]) == pytest.approx(1.0)


def test_reference_roundtrips_through_a_file(reference_data, reference):
    assert reference.vocab == frozenset(reference_data["vocab"])
    assert reference.features["chars"].edges == tuple(reference_data["features"]["chars"]["edges"])
    assert reference.meta["source"] == "synthetic"


def test_reference_needs_enough_reviews_and_matching_labels():
    with pytest.raises(ValueError):
        drift.build_reference(["a"] * 10, [1] * 10)
    with pytest.raises(ValueError):
        drift.build_reference(["a"] * 60, [1] * 59)


def test_load_reference_rejects_a_wrong_version(reference_data, tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({**reference_data, "version": 99}))

    with pytest.raises(ValueError):
        drift.load_reference(path)


# ---------------------------------------------------------------- detection


def test_nothing_is_decided_before_the_minimum_window(reference):
    detector = make_detector(reference, min_window=100)

    report = feed(detector, english_reviews(99, seed=10))

    assert report.status == "insufficient_data"
    assert report.window == 99
    assert report.features == {}


@pytest.mark.parametrize("window", [100, 200])
def test_clean_data_does_not_raise_false_alarms(reference, window):
    for seed in range(100, 130):
        detector = make_detector(reference, window_size=window, min_window=window)
        report = feed(detector, english_reviews(window, seed=seed), None)
        assert report.status == "ok", (seed, report.to_dict())


def test_a_rolling_window_of_clean_traffic_stays_ok(reference):
    detector = make_detector(reference)
    statuses = set()

    for text, pred in zip(
        english_reviews(600, seed=20), ["positive", "negative"] * 300, strict=False
    ):
        statuses.add(detector.observe(text, pred).status)

    assert statuses <= {"insufficient_data", "ok"}


@pytest.mark.parametrize(
    ("make_texts", "feature"),
    [
        (lambda n: ["ok"] * n, "chars"),
        (lambda n: ["یہ پروڈکٹ بہت اچھی ہے اور میں اسے پسند کرتا ہوں"] * n, "non_ascii_ratio"),
        (lambda n: ["👍👍👍 ❤️ 🔥"] * n, "non_ascii_ratio"),
        (lambda n: [" ".join(["lorem ipsum dolor sit amet"] * 150)] * n, "oov_rate"),
    ],
    ids=["very short", "non-English", "emoji only", "unknown vocabulary"],
)
def test_shifted_inputs_are_flagged(reference, make_texts, feature):
    detector = make_detector(reference)

    report = feed(detector, make_texts(100))

    assert report.status == "drift"
    assert feature in report.drifted_features


def test_a_change_in_the_predictions_alone_is_flagged(reference):
    detector = make_detector(reference)

    report = feed(detector, english_reviews(100, seed=30), ["positive"] * 100)

    assert report.status == "drift"
    assert report.drifted_features == ["positive"]


def test_the_window_forgets_old_requests_and_recovers(reference):
    detector = make_detector(reference, window_size=200, min_window=100)
    assert feed(detector, ["ok"] * 100).status == "drift"

    report = feed(detector, english_reviews(200, seed=40))

    assert report.status == "ok"


def test_drift_callback_is_called_and_a_failing_callback_is_contained(reference, caplog):
    seen = []

    def failing_callback(report):
        seen.append(report.status)
        raise RuntimeError("mail server down")

    detector = make_detector(reference, on_drift=failing_callback)

    with caplog.at_level(logging.ERROR, logger="reviewpulse.drift"):
        report = feed(detector, ["ok"] * 101)

    assert report.status == "drift"
    assert seen and set(seen) == {"drift"}
    assert any("callback failed" in r.getMessage() for r in caplog.records)


def test_drift_and_recovery_are_logged_once_each(reference, caplog):
    detector = make_detector(reference, window_size=100, min_window=100)

    with caplog.at_level(logging.INFO, logger="reviewpulse.drift"):
        feed(detector, ["ok"] * 100)
        feed(detector, ["ok"] * 10)
        feed(detector, english_reviews(100, seed=50))

    messages = [r.getMessage() for r in caplog.records]
    assert sum("Data drift detected" in m for m in messages) == 1
    assert sum("Data drift cleared" in m for m in messages) == 1


def test_report_is_json_serialisable_and_names_the_drifted_features(reference):
    detector = make_detector(reference)

    report = feed(detector, ["ok"] * 100).to_dict()

    assert json.loads(json.dumps(report)) == report
    assert report["status"] == "drift"
    assert "chars" in report["drifted_features"]
    assert set(report["features"]["chars"]) == {"psi", "drifted", "reference_mean", "current_mean"}


def test_min_window_cannot_exceed_window_size(reference):
    with pytest.raises(ValueError):
        make_detector(reference, window_size=50, min_window=60)


def test_concurrent_requests_do_not_break_the_detector(reference):
    detector = make_detector(reference)
    errors = []

    def worker(seed):
        try:
            feed(detector, english_reviews(40, seed=seed))
        except Exception as error:  # pragma: no cover
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert detector.report().window == 200


# ---------------------------------------------------------------- environment and module hooks


@pytest.fixture(autouse=True)
def fresh_module_state():
    drift.reset_detector()
    yield
    drift.reset_detector()


def test_from_env_without_a_reference_file_is_off_and_says_why(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="reviewpulse.drift"):
        detector = drift.DriftDetector.from_env({"DRIFT_REFERENCE": str(tmp_path / "none.json")})

    assert detector is None
    assert any("Drift detection is off" in r.getMessage() for r in caplog.records)


def test_from_env_reads_settings_and_ignores_bad_numbers(reference_data, tmp_path, caplog):
    path = drift.save_reference(reference_data, tmp_path / "ref.json")
    env = {
        "DRIFT_REFERENCE": str(path),
        "DRIFT_WINDOW_SIZE": "80",
        "DRIFT_MIN_WINDOW": "many",
        "DRIFT_PSI_THRESHOLD": "0.4",
    }

    with caplog.at_level(logging.WARNING, logger="reviewpulse.drift"):
        detector = drift.DriftDetector.from_env(env)

    assert detector.window_size == 80
    assert detector.min_window == 80  # default 100 is above the window, so it is capped
    assert detector.psi_threshold == 0.4
    assert any("DRIFT_MIN_WINDOW" in r.getMessage() for r in caplog.records)


def test_from_env_survives_a_corrupt_reference(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")

    assert drift.DriftDetector.from_env({"DRIFT_REFERENCE": str(bad)}) is None


def test_module_hooks_when_drift_detection_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFT_REFERENCE", str(tmp_path / "none.json"))

    assert drift.observe("some text", "positive") is None
    report = drift.current_report()

    assert report.status == "disabled"


def test_module_hooks_use_the_detector_and_the_registered_callback(reference):
    calls = []
    drift.set_detector(make_detector(reference))
    drift.set_on_drift(lambda report: calls.append(report.status))

    for _ in range(100):
        drift.observe("ok", "positive")

    assert drift.current_report().status == "drift"
    assert calls and set(calls) == {"drift"}


def test_observe_never_raises(reference, monkeypatch, caplog):
    detector = make_detector(reference)
    monkeypatch.setattr(detector, "observe", lambda *a: 1 / 0)
    drift.set_detector(detector)

    with caplog.at_level(logging.ERROR, logger="reviewpulse.drift"):
        assert drift.observe("text", "positive") is None

    assert any("Drift observation failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- the build script


def write_csv(path, n, seed):
    pd.DataFrame({"review_text": english_reviews(n, seed), "label": labels(n, seed)}).to_csv(
        path, index=False
    )
    return path


def test_build_script_writes_a_reference_and_checks_clean_data(tmp_path, caplog):
    train = write_csv(tmp_path / "train.csv", 1500, seed=60)
    check = write_csv(tmp_path / "validation.csv", 800, seed=61)
    out = tmp_path / "ref" / "reference.json"

    with caplog.at_level(logging.INFO, logger="reviewpulse.drift_reference"):
        code = build_script.main(
            [
                "--csv",
                str(train),
                "--check-csv",
                str(check),
                "--out",
                str(out),
                "--vocab-size",
                "500",
            ]
        )

    assert code == 0
    assert drift.load_reference(out).meta["n_reviews"] == 1500
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "False alarm check: 0 of 4 clean windows" in messages


def test_build_script_skips_the_check_when_there_is_no_validation_data(tmp_path, caplog):
    train = write_csv(tmp_path / "train.csv", 300, seed=62)
    out = tmp_path / "reference.json"

    with caplog.at_level(logging.WARNING, logger="reviewpulse.drift_reference"):
        code = build_script.main(
            ["--csv", str(train), "--check-csv", str(tmp_path / "missing.csv"), "--out", str(out)]
        )

    assert code == 0
    assert out.is_file()
    assert any("No validation split found" in r.getMessage() for r in caplog.records)
