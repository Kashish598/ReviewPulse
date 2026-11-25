"""/predict saves each request and wakes the labeller, and neither can break the answer."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text as sql

from src import feedback_store
from src.app import PRIVACY_NOTE, app, get_predictor
from src.common import MockPredictor
from src.feedback_store import PENDING, FeedbackStore


@pytest.fixture
def store(tmp_path):
    created = FeedbackStore(f"sqlite:///{tmp_path / 'feedback.db'}")
    feedback_store.set_store(created)
    yield created
    feedback_store.set_store(None)
    created.dispose()


@pytest.fixture
def client(store):
    app.dependency_overrides[get_predictor] = MockPredictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class FakeLabeler:
    def __init__(self):
        self.woken = 0
        self.stopped = False

    def wake(self):
        self.woken += 1

    def stop(self):
        self.stopped = True


def saved_rows(store):
    with store.engine.connect() as connection:
        found = connection.execute(
            sql("SELECT text, pred_label, model_name, label_status FROM requests")
        )
        return [tuple(row) for row in found]


def test_a_prediction_is_saved_with_the_model_name_and_starts_pending(client, store):
    answer = client.post("/predict", json={"text": "I love this, it is great"})

    assert answer.status_code == 200
    assert saved_rows(store) == [("I love this, it is great", "positive", "mock-keyword", PENDING)]


def test_the_same_review_sent_twice_is_saved_once(client, store):
    client.post("/predict", json={"text": "Same review"})
    client.post("/predict", json={"text": "Same review"})

    assert len(saved_rows(store)) == 1


def test_a_rejected_request_is_not_saved(client, store):
    assert client.post("/predict", json={"text": "   "}).status_code == 422

    assert saved_rows(store) == []


def test_the_labeller_is_woken_only_for_a_new_request(client):
    fake = FakeLabeler()
    app.state.labeler = fake

    client.post("/predict", json={"text": "A brand new review"})
    client.post("/predict", json={"text": "A brand new review"})  # duplicate, nothing to label

    assert fake.woken == 1


def test_a_database_failure_does_not_fail_the_prediction(client, store, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("database is down")

    monkeypatch.setattr(store, "add_request", broken)

    answer = client.post("/predict", json={"text": "I love this, it is great"})

    assert answer.status_code == 200
    assert answer.json()["sentiment"] == "positive"


def test_a_labeller_failure_does_not_fail_the_prediction(client):
    class BrokenLabeler(FakeLabeler):
        def wake(self):
            raise RuntimeError("thread is gone")

    app.state.labeler = BrokenLabeler()

    answer = client.post("/predict", json={"text": "Another review"})

    assert answer.status_code == 200


def test_with_saving_off_predictions_still_work_and_nothing_is_stored(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_STORE", "off")
    feedback_store.set_store(None)
    app.dependency_overrides[get_predictor] = MockPredictor
    try:
        with TestClient(app) as test_client:
            answer = test_client.post("/predict", json={"text": "I love this, it is great"})
    finally:
        app.dependency_overrides.clear()
        feedback_store.set_store(None)

    assert answer.status_code == 200
    assert not (tmp_path / "feedback.db").exists()


def test_the_labeller_is_stopped_when_the_app_shuts_down(store):
    fake = FakeLabeler()
    app.dependency_overrides[get_predictor] = MockPredictor
    try:
        with TestClient(app):
            app.state.labeler = fake
    finally:
        app.dependency_overrides.clear()

    assert fake.stopped is True


def test_the_privacy_note_is_in_the_api_docs(client):
    spec = client.get("/openapi.json").json()

    assert PRIVACY_NOTE in spec["info"]["description"]
    assert PRIVACY_NOTE in spec["paths"]["/predict"]["post"]["description"]
