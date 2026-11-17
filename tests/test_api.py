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
