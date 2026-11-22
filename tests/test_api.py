import logging

import pytest
from fastapi.testclient import TestClient

from src.app import MAX_TEXT_CHARS, app, get_predictor
from src.common import MockPredictor


@pytest.fixture
def client():
    # Use MockPredictor instead of loading the real BERT model.
    app.dependency_overrides[get_predictor] = MockPredictor

    with TestClient(app) as test_client:
        yield test_client

    # Clean up after each test.
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model"] == "mock-keyword"


def test_predict_valid(client):
    r = client.post("/predict", json={"text": "I love this, it is great"})

    assert r.status_code == 200

    body = r.json()
    assert body["sentiment"] == "positive"
    assert body["score"] == 0.5


def test_predict_rejects_empty_and_blank(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_too_long(client):
    assert (
        client.post(
            "/predict",
            json={"text": "a" * (MAX_TEXT_CHARS + 1)},
        ).status_code
        == 422
    )

    assert (
        client.post(
            "/predict",
            json={"text": "a" * MAX_TEXT_CHARS},
        ).status_code
        == 200
    )


def test_predict_rejects_invalid_payload(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": 123}).status_code == 422


def test_health_reports_whether_the_model_is_fine_tuned():
    class FineTunedStub:
        name = "nlptown-bert-finetuned"
        fine_tuned = True

    app.dependency_overrides[get_predictor] = FineTunedStub

    try:
        with TestClient(app) as test_client:
            body = test_client.get("/health").json()
    finally:
        app.dependency_overrides.clear()

    assert body == {"status": "ok", "model": "nlptown-bert-finetuned", "fine_tuned": True}


def test_unhandled_error_returns_500_logs_the_traceback_and_the_server_keeps_working(caplog):
    class BrokenPredictor:
        name = "broken"

        def predict(self, texts):
            raise RuntimeError("model exploded")

    app.dependency_overrides[get_predictor] = BrokenPredictor
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            with caplog.at_level(logging.ERROR, logger="src.app"):
                broken = test_client.post("/predict", json={"text": "fine review"})
            app.dependency_overrides[get_predictor] = MockPredictor
            healthy = test_client.post("/predict", json={"text": "fine review"})
    finally:
        app.dependency_overrides.clear()

    assert broken.status_code == 500
    assert broken.json() == {"detail": "Internal server error"}
    logged = [r for r in caplog.records if "Unhandled error" in r.getMessage()]
    assert logged and logged[0].exc_info and "model exploded" in str(logged[0].exc_info[1])
    assert healthy.status_code == 200


def test_predict_logs_what_it_received_and_returned_at_debug(client, caplog):
    with caplog.at_level(logging.DEBUG, logger="src.app"):
        client.post("/predict", json={"text": "I love this, it is great"})

    messages = [r.getMessage() for r in caplog.records if r.name == "src.app"]
    assert any("received 24 chars" in m and "I love this" in m for m in messages)
    assert any("model=mock-keyword label=positive" in m for m in messages)
