"""Fresh reviews for the retraining demo: never a review from the three splits."""

import importlib.util
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fresh_reviews", ROOT / "scripts" / "fresh_reviews.py"
)
fresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fresh)


def raw_pool(count=60):
    return pd.DataFrame(
        {
            "label": [i % 2 for i in range(count)],
            "title": ["t"] * count,
            "content": [f"review number {i} about a product" for i in range(count)],
        }
    )


def test_fresh_reviews_exclude_known_texts_and_keep_only_the_text():
    known = {f"review number {i} about a product" for i in range(10)}

    result = fresh.fresh_reviews(raw_pool(), known, count=20, seed=1)

    assert list(result.columns) == ["review_text"]
    assert len(result) == 20 and result["review_text"].is_unique
    assert not (set(result["review_text"]) & known)


def test_the_same_seed_gives_the_same_reviews():
    first = fresh.fresh_reviews(raw_pool(), set(), count=10, seed=3)
    second = fresh.fresh_reviews(raw_pool(), set(), count=10, seed=3)

    assert first.equals(second)


def test_long_reviews_are_left_out():
    pool = raw_pool(5)
    pool.loc[0, "content"] = "x" * 2000

    result = fresh.fresh_reviews(pool, set(), count=4, seed=1, max_chars=1000)

    assert all(len(text) <= 1000 for text in result["review_text"])


def test_asking_for_more_than_there_is_is_an_error():
    with pytest.raises(ValueError, match="Only 5 fresh reviews"):
        fresh.fresh_reviews(raw_pool(5), set(), count=10)


def test_each_review_is_sent_once_and_counted():
    sent = []

    def post(url, text):
        sent.append((url, text))
        return 200

    accepted, failed = fresh.send_reviews(
        ["a", "b", "c"], "http://x", post=post, sleep=lambda s: None
    )

    assert (accepted, failed) == (3, 0)
    assert [text for _, text in sent] == ["a", "b", "c"]


def test_rejected_requests_are_counted_as_failed():
    accepted, failed = fresh.send_reviews(
        ["a", "b"], "http://x", post=lambda u, t: 422, sleep=lambda s: None
    )

    assert (accepted, failed) == (0, 2)


def test_sending_stops_when_the_api_cannot_be_reached():
    calls = []

    def unreachable(url, text):
        calls.append(text)
        raise urllib.error.URLError("connection refused")

    accepted, failed = fresh.send_reviews(
        ["a"] * 10, "http://x", post=unreachable, sleep=lambda s: None
    )

    assert accepted == 0 and failed == 3 and len(calls) == 3


def test_one_failure_among_good_requests_does_not_stop_the_run():
    answers = iter([200, OSError("blip"), 200])

    def post(url, text):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    accepted, failed = fresh.send_reviews(
        ["a", "b", "c"], "http://x", post=post, sleep=lambda s: None
    )

    assert (accepted, failed) == (2, 1)


def test_the_command_line_writes_a_csv_without_the_known_reviews(tmp_path, monkeypatch):
    splits = {
        name: pd.DataFrame(
            {"review_text": [f"review number {i} about a product" for i in range(a, b)]}
        )
        for name, (a, b) in {"train": (0, 10), "validation": (10, 15), "test": (15, 20)}.items()
    }
    monkeypatch.setattr(fresh, "load_split", lambda name, sample=False: splits[name])
    monkeypatch.setattr(fresh, "load_raw", lambda revision=None: (raw_pool(), raw_pool(3), "rev"))
    out = tmp_path / "fresh.csv"

    code = fresh.main(["--count", "15", "--out", str(out)])

    written = pd.read_csv(out)
    assert code == 0 and len(written) == 15
    assert not any(
        f"review number {i} about" in t for t in written["review_text"] for i in range(20)
    )
