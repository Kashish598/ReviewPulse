"""Drift detection and the email alert wired into the real FastAPI app (/predict, /drift)."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import alerts, drift
from src.app import app, get_predictor
from src.common import Prediction

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "simulate_drift", ROOT / "scripts" / "simulate_drift.py"
)
sim = importlib.util.module_from_spec(_spec)
sys.modules["simulate_drift"] = sim
_spec.loader.exec_module(sim)


class AlternatingPredictor:
    """Answers negative, positive, negative... so the share of positive answers matches the
    reference built below, and only the text itself can look different."""

    name = "alternating-stub"

    def __init__(self):
        self.calls = 0

    def predict(self, texts):
        out = []
        for _ in texts:
            self.calls += 1
            out.append(Prediction(label="positive" if self.calls % 2 else "negative", score=0.9))
        return out


class Recorder:
    def __init__(self):
        self.sent = []

    def __call__(self, smtp_settings, subject, body):
        self.sent.append((subject, body))


def make_reference():
    rng = random.Random(0)
    texts = [rng.choice(sim.NORMAL_REVIEWS) for _ in range(800)]
    data = drift.build_reference(texts, [rng.randint(0, 1) for _ in texts], vocab_size=100)
    return drift.Reference(
        vocab=frozenset(data["vocab"]),
        features={
            name: drift.FeatureReference(
                edges=tuple(spec["edges"]),
                proportions=tuple(spec["proportions"]),
                mean=spec["mean"],
                std=spec["std"],
            )
            for name, spec in data["features"].items()
        },
    )


@pytest.fixture
def recorder():
    recorder = Recorder()
    alerts.set_manager(
        alerts.AlertManager(
            alerts.SmtpSettings("smtp.example.com", 587, "u", "p", "u@x.com", ("to@x.com",)),
            sender=recorder,
            threaded=False,
        )
    )
    return recorder


@pytest.fixture
def client(recorder):
    drift.set_detector(drift.DriftDetector(make_reference()))
    # One shared instance: FastAPI calls a class override once per request, which would give a
    # fresh counter every time and make every answer "positive".
    predictor = AlternatingPredictor()
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def send(client, texts):
    for text in texts:
        assert client.post("/predict", json={"text": text}).status_code == 200


def test_drift_endpoint_starts_with_a_filling_window(client):
    body = client.get("/drift").json()

    assert body["status"] == "insufficient_data"
    assert body["window"] == 0


def test_every_prediction_is_added_to_the_window(client):
    send(client, sim.build_traffic("normal", 5, seed=1, csv_path=Path("none.csv")))

    assert client.get("/drift").json()["window"] == 5


def test_normal_traffic_is_ok_and_sends_no_email(client, recorder):
    send(client, sim.build_traffic("normal", 200, seed=1, csv_path=Path("none.csv")))

    body = client.get("/drift").json()

    assert body["status"] == "ok"
    assert body["drifted_features"] == []
    assert recorder.sent == []


def test_shifted_traffic_is_flagged_and_one_email_names_what_changed(client, recorder):
    send(client, sim.build_traffic("shifted", 150, seed=2))

    body = client.get("/drift").json()

    assert body["status"] == "drift"
    assert "non_ascii_ratio" in body["drifted_features"]
    assert len(recorder.sent) == 1  # the cooldown allows one email while the drift lasts
    subject, text = recorder.sent[0]
    assert subject.startswith("[ReviewPulse] Data drift detected")
    assert "non_ascii_ratio" in text


def test_without_a_reference_file_drift_is_disabled_and_predict_still_works(monkeypatch):
    monkeypatch.setenv("DRIFT_REFERENCE", str(ROOT / "no_such_reference.json"))
    app.dependency_overrides[get_predictor] = AlternatingPredictor
    try:
        with TestClient(app) as test_client:
            assert test_client.post("/predict", json={"text": "fine"}).status_code == 200
            body = test_client.get("/drift").json()
    finally:
        app.dependency_overrides.clear()

    assert body["status"] == "disabled"


def test_a_failing_detector_never_fails_predict():
    class BrokenDetector:
        on_drift = None

        def observe(self, text, pred_label):
            raise RuntimeError("detector exploded")

    drift.set_detector(BrokenDetector())
    app.dependency_overrides[get_predictor] = AlternatingPredictor
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/predict", json={"text": "fine"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


def test_the_committed_reference_file_loads_and_drift_is_on():
    # No override of the detector: the app loads data/reference/reference.json like in production.
    app.dependency_overrides[get_predictor] = AlternatingPredictor
    try:
        with TestClient(app) as test_client:
            body = test_client.get("/drift").json()
    finally:
        app.dependency_overrides.clear()

    assert body["status"] != "disabled"
