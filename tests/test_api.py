from fastapi.testclient import TestClient

from src.app import MAX_TEXT_CHARS, app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model"]  # the served model is always named


def test_predict_valid():
    r = client.post("/predict", json={"text": "I love this, it is great"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] in ("negative", "positive")
    assert "score" in body


def test_predict_rejects_empty_and_blank():
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_too_long():
    assert client.post("/predict", json={"text": "a" * (MAX_TEXT_CHARS + 1)}).status_code == 422
    assert client.post("/predict", json={"text": "a" * MAX_TEXT_CHARS}).status_code == 200


def test_predict_rejects_invalid_payload():
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": 123}).status_code == 422
