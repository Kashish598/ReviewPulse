"""K3 (/health and /ready) and K2 (/stats and /metrics) through the real FastAPI app."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import src.app as app_module
from src.app import app, get_predictor
from src.common import MockPredictor, Prediction
from src.monitoring import stats


@pytest.fixture(autouse=True)
def _clean_state():
    stats.reset()
    app_module._loaded_predictor = None
    yield
    app.dependency_overrides.clear()
    stats.reset()
    app_module._loaded_predictor = None


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = MockPredictor
    with TestClient(app) as test_client:
        yield test_client


class TimedStub:
    """Looks like the BERT adapter: reports its own stage times."""

    name = "timed-stub"

    def predict_timed(self, texts):
        stages = {"tokenize_ms": 1.0, "inference_ms": 2.0, "label_mapping_ms": 0.5, "total_ms": 3.5}
        return [Prediction(label="positive", score=0.9) for _ in texts], stages


# ------------------------------------------------------------- K3: health and ready


def test_health_answers_without_loading_the_model():
    # No predictor override and MODEL_PRELOAD=off (see conftest): nothing may load the model.
    with TestClient(app) as test_client:
        body = test_client.get("/health").json()

    assert body == {"status": "ok"}
    assert app_module._loaded_predictor is None


def test_ready_says_loading_while_the_model_is_not_loaded():
    with TestClient(app) as test_client:
        response = test_client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_ready_is_200_once_the_model_is_loaded(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "mock-keyword"}


def test_ready_reports_a_failed_load_and_health_stays_up():
    def failing_loader():
        raise RuntimeError("no weights here")

    app.dependency_overrides[get_predictor] = failing_loader
    with TestClient(app) as test_client:
        ready = test_client.get("/ready")
        health = test_client.get("/health")

    assert ready.status_code == 503
    assert ready.json()["status"] == "error"
    assert "no weights here" in ready.json()["detail"]
    assert health.status_code == 200


def test_the_model_loads_in_the_background_while_health_already_answers(monkeypatch):
    release = threading.Event()

    def slow_loader():
        release.wait(timeout=10)
        return MockPredictor()

    monkeypatch.setenv("MODEL_PRELOAD", "on")
    monkeypatch.setattr(app_module, "get_predictor", slow_loader)

    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/ready").status_code == 503
        release.set()
        for _ in range(100):
            if test_client.get("/ready").status_code == 200:
                break
            time.sleep(0.05)
        final = test_client.get("/ready")

    assert final.status_code == 200
    assert final.json()["model"] == "mock-keyword"


# ------------------------------------------------------------ K2: stats and metrics


def test_stats_start_empty(client):
    body = client.get("/stats").json()

    assert body["request_count"] == 0
    assert body["avg_api_latency_ms"] is None
    assert body["avg_model_latency_ms"] is None
    assert body["per_model"] == {}


def test_request_count_rises_with_each_predict_call(client):
    for expected in (1, 2, 3):
        assert client.post("/predict", json={"text": "I love this"}).status_code == 200
        assert client.get("/stats").json()["request_count"] == expected


def test_api_latency_is_above_model_latency_and_the_model_block_is_filled(client):
    for _ in range(5):
        client.post("/predict", json={"text": "great product"})

    body = client.get("/stats").json()

    assert body["avg_api_latency_ms"] > body["avg_model_latency_ms"] > 0
    block = body["per_model"]["mock-keyword"]
    assert block["count"] == 5
    assert block["avg_model_ms"] > 0
    for key in ("avg_tokenize_ms", "avg_inference_ms", "avg_label_mapping_ms"):
        assert key in block  # None for the mock, which has no stages


def test_stage_times_from_the_model_are_averaged_per_model():
    app.dependency_overrides[get_predictor] = TimedStub
    with TestClient(app) as test_client:
        test_client.post("/predict", json={"text": "fine"})
        test_client.post("/predict", json={"text": "fine again"})
        block = test_client.get("/stats").json()["per_model"]["timed-stub"]

    assert block == {
        "count": 2,
        "avg_model_ms": 3.5,
        "avg_tokenize_ms": 1.0,
        "avg_inference_ms": 2.0,
        "avg_label_mapping_ms": 0.5,
    }


def test_only_successful_predict_calls_are_counted(client):
    client.get("/health")
    client.get("/ready")
    client.get("/stats")
    client.post("/predict", json={"text": ""})  # 422
    assert client.get("/stats").json()["request_count"] == 0

    client.post("/predict", json={"text": "good"})
    assert client.get("/stats").json()["request_count"] == 1


def test_metrics_returns_the_same_json_as_stats(client):
    client.post("/predict", json={"text": "good"})

    assert client.get("/metrics").json() == client.get("/stats").json()
