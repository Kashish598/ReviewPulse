import logging
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src import feedback_store
from src.feedback_store import (
    DISCARDED,
    LABELLED,
    MAX_TEXT_CHARS,
    PENDING,
    RAW_CHARS,
    FeedbackStore,
    metadata,
    requests_table,
)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    """Every store test runs on SQLite, and on Postgres when TEST_DATABASE_URL is set."""
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'feedback.db'}"
    else:
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("set TEST_DATABASE_URL to also run against Postgres")
        database = url.split("?")[0].rsplit("/", 1)[-1].lower()
        if "test" not in database:
            # These tests drop and recreate the tables, so never run them on a real database.
            pytest.fail(f"TEST_DATABASE_URL needs 'test' in the database name, got {database!r}")
    created = FeedbackStore(url, clock=Clock())
    metadata.drop_all(created.engine)
    metadata.create_all(created.engine)
    yield created
    created.dispose()


@pytest.fixture(autouse=True)
def fresh_module_state():
    feedback_store.set_store(None)
    yield
    feedback_store.set_store(None)


def add(store, text="a fine review", label="positive", score=0.9):
    return store.add_request(text, label, score, "test-model", "v1")


def row(store, request_id):
    with store.engine.connect() as connection:
        found = connection.execute(select(requests_table).where(requests_table.c.id == request_id))
        return dict(found.one()._mapping)


# ------------------------------------------------------------------ saving


def test_a_saved_request_keeps_the_prediction_and_starts_pending(store):
    request_id = add(store, "Loved it", "positive", 0.97)

    saved = row(store, request_id)
    assert saved["text"] == "Loved it"
    assert (saved["pred_label"], saved["pred_score"]) == ("positive", 0.97)
    assert (saved["model_name"], saved["model_version"]) == ("test-model", "v1")
    assert saved["label_status"] == PENDING
    assert saved["label_attempts"] == 0
    assert saved["split"] in {"train", "validation"}
    assert saved["llm_label"] is None


def test_the_same_text_is_saved_once(store):
    first = add(store, "Same review")

    assert add(store, "Same review") is None
    assert add(store, "  Same review  ") is None  # surrounding spaces do not make it new
    assert first is not None
    assert sum(store.counts_by_status().values()) == 1


def test_very_long_text_is_cut_to_the_api_limit(store):
    request_id = add(store, "x" * (MAX_TEXT_CHARS + 500))

    assert len(row(store, request_id)["text"]) == MAX_TEXT_CHARS


def test_the_split_is_fixed_by_the_text_and_close_to_80_20():
    digests = [feedback_store.text_hash(f"review {i}") for i in range(4000)]
    splits = [feedback_store.assign_split(d) for d in digests]

    assert splits == [feedback_store.assign_split(d) for d in digests]
    share = splits.count("validation") / len(splits)
    assert 0.17 < share < 0.23


def test_record_never_raises_and_logs_the_error(store, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("database is down")

    monkeypatch.setattr(store, "add_request", broken)

    with caplog.at_level(logging.ERROR, logger="reviewpulse.store"):
        assert store.record("text", "positive", 0.9, "m") is None

    assert any("Could not save the request" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ labelling


def test_pending_requests_come_oldest_first_and_respect_the_limit(store):
    ids = [add(store, f"review {i}") for i in range(5)]

    claimed = store.claim_pending(3)

    assert [r["id"] for r in claimed] == ids[:3]
    assert set(claimed[0]) == {"id", "text", "pred_label", "pred_score", "model_name"}


def test_marking_a_label_stores_who_said_what_and_removes_it_from_pending(store):
    request_id = add(store)

    assert store.mark_labelled(request_id, "negative", "llm-x", "v1") is True

    saved = row(store, request_id)
    assert saved["label_status"] == LABELLED
    assert (saved["llm_label"], saved["llm_model"], saved["prompt_version"]) == (
        "negative",
        "llm-x",
        "v1",
    )
    assert saved["label_attempts"] == 1
    assert saved["labelled_at"] is not None
    assert store.claim_pending(10) == []


def test_a_request_can_only_be_finished_once(store):
    request_id = add(store)
    store.mark_labelled(request_id, "negative", "llm-x", "v1")

    assert store.mark_labelled(request_id, "positive", "llm-y", "v2") is False
    assert store.mark_discarded(request_id, "late", "llm-y", "v2") is False
    assert row(store, request_id)["llm_label"] == "negative"


def test_discarding_keeps_the_rejected_answer_but_not_all_of_it(store):
    request_id = add(store)

    store.mark_discarded(request_id, "I think " * 100, "llm-x", "v1")

    saved = row(store, request_id)
    assert saved["label_status"] == DISCARDED
    assert saved["llm_label"] is None
    assert len(saved["llm_raw"]) == RAW_CHARS


def test_a_failed_attempt_is_counted_and_keeps_the_request_pending(store):
    request_id = add(store)

    assert store.record_attempt(request_id, "neutral " * 100) == 1
    assert store.record_attempt(request_id, "still no label") == 2

    saved = row(store, request_id)
    assert saved["label_status"] == PENDING
    assert saved["label_attempts"] == 2
    assert saved["llm_raw"] == "still no label"
    assert saved["last_attempt_at"] is not None


def test_a_long_rejected_answer_is_cut_when_an_attempt_is_recorded(store):
    request_id = add(store)

    store.record_attempt(request_id, "x" * 1000)

    assert len(row(store, request_id)["llm_raw"]) == RAW_CHARS


def test_recording_an_attempt_on_a_finished_request_does_nothing(store):
    request_id = add(store)
    store.mark_labelled(request_id, "positive", "m", "v1")

    assert store.record_attempt(request_id, "late") == 0
    assert row(store, request_id)["label_attempts"] == 1


def test_discarding_after_a_recorded_attempt_does_not_count_it_twice(store):
    request_id = add(store)
    store.record_attempt(request_id, "neutral")

    store.mark_discarded(request_id, "neutral", "m", "v1", count_attempt=False)

    assert row(store, request_id)["label_attempts"] == 1


def test_requests_that_already_failed_are_claimed_after_fresh_ones(store):
    first, second = add(store, "one"), add(store, "two")
    store.record_attempt(first, "neutral")

    assert [r["id"] for r in store.claim_pending(10)] == [second, first]


def test_counts_include_every_status_even_when_empty(store):
    first, second = add(store, "one"), add(store, "two")
    add(store, "three")
    store.mark_labelled(first, "positive", "m", "v1")
    store.mark_discarded(second, "??", "m", "v1")

    assert store.counts_by_status() == {PENDING: 1, LABELLED: 1, DISCARDED: 1}


def test_labelled_rows_can_be_filtered_by_split(store):
    for i in range(60):
        request_id = add(store, f"review {i}", "positive")
        store.mark_labelled(request_id, "positive", "m", "v1")

    everything = store.labelled_rows()
    train = store.labelled_rows("train")
    validation = store.labelled_rows("validation")

    assert len(everything) == 60
    assert len(train) + len(validation) == 60
    assert train and validation
    assert {r["split"] for r in train} == {"train"}
    assert set(everything[0]) == {"id", "text", "llm_label", "pred_label", "split"}


def test_agreement_between_the_served_model_and_the_llm(store):
    for i, (pred, llm) in enumerate(
        [
            ("positive", "positive"),
            ("positive", "negative"),
            ("negative", "negative"),
            ("negative", "negative"),
        ]
    ):
        request_id = add(store, f"review {i}", pred)
        store.mark_labelled(request_id, llm, "m", "v1")
    add(store, "still pending")

    assert store.agreement_stats() == {
        "labelled": 4,
        "agree": 3,
        "disagree": 1,
        "agreement_rate": 0.75,
    }


def test_agreement_with_nothing_labelled(store):
    assert store.agreement_stats() == {
        "labelled": 0,
        "agree": 0,
        "disagree": 0,
        "agreement_rate": None,
    }


# ------------------------------------------------------------------ daily usage


def test_label_calls_and_tokens_are_counted_per_day(store):
    clock = store._clock

    assert store.calls_today() == 0
    assert store.record_label_call(300) == 1
    assert store.record_label_call(200) == 2
    assert (store.calls_today(), store.tokens_today()) == (2, 500)

    clock.advance(days=1)
    assert store.calls_today() == 0  # a new UTC day starts at zero
    assert store.record_label_call(100) == 1


def test_simultaneous_saves_and_counts_do_not_lose_updates(store):
    errors = []

    def worker(number):
        try:
            for i in range(10):
                add(store, f"thread {number} review {i}")
                store.record_label_call(10)
        except Exception as error:  # pragma: no cover
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sum(store.counts_by_status().values()) == 60
    assert (store.calls_today(), store.tokens_today()) == (60, 600)


# ------------------------------------------------------------------ database url and start-up


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "postgresql://u:p@host/db?sslmode=require",
            "postgresql+psycopg://u:p@host/db?sslmode=require",
        ),
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_postgres_urls_are_pointed_at_the_psycopg_driver(given, expected):
    assert feedback_store.normalise_database_url(given) == expected


def test_the_password_is_hidden_when_a_url_is_logged():
    shown = feedback_store.redact_url("postgresql://neon_user:s3cret@ep-1.neon.tech/db")

    assert "s3cret" not in shown
    assert shown == "postgresql://neon_user:***@ep-1.neon.tech/db"


def test_postgres_engine_checks_connections_before_using_them():
    pytest.importorskip("psycopg")

    engine = feedback_store.make_engine("postgresql://u:p@localhost/db")  # does not connect

    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.pool._pre_ping is True  # Neon suspends idle databases


def test_without_a_database_url_a_local_file_is_used_and_the_user_is_told(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(feedback_store, "DEFAULT_SQLITE_PATH", tmp_path / "sub" / "feedback.db")

    with caplog.at_level(logging.WARNING, logger="reviewpulse.store"):
        created = FeedbackStore()

    assert (tmp_path / "sub" / "feedback.db").exists()
    assert any("No DATABASE_URL" in r.getMessage() for r in caplog.records)
    created.dispose()


def test_get_store_can_be_turned_off():
    assert feedback_store.get_store({"FEEDBACK_STORE": "off"}) is None


def test_get_store_uses_database_url(tmp_path):
    created = feedback_store.get_store({"DATABASE_URL": f"sqlite:///{tmp_path / 'env.db'}"})

    assert created is not None and created.dialect == "sqlite"
    assert feedback_store.get_store({"FEEDBACK_STORE": "off"}) is created  # loaded once
    created.dispose()


def test_a_store_that_cannot_start_does_not_stop_the_app(caplog):
    pytest.importorskip("psycopg")
    env = {"DATABASE_URL": "postgresql://u:p@127.0.0.1:9/nothing"}

    with caplog.at_level(logging.ERROR, logger="reviewpulse.store"):
        assert feedback_store.get_store(env) is None

    assert any("could not start" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ module hook


def test_record_prediction_saves_through_the_shared_store(store):
    feedback_store.set_store(store)

    first = feedback_store.record_prediction("nice one", "positive", 0.8, "m", "v1")
    again = feedback_store.record_prediction("nice one", "positive", 0.8, "m", "v1")

    assert first is not None
    assert again is None
    assert store.counts_by_status()[PENDING] == 1


def test_record_prediction_is_a_no_op_when_saving_is_off(monkeypatch):
    monkeypatch.setenv("FEEDBACK_STORE", "off")

    assert feedback_store.record_prediction("text", "positive", 0.8, "m") is None
