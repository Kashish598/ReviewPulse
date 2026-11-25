import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.labeler import AuthError, LLMResult, PermanentError, RateLimitError, TransientError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_labeler.py"
_spec = importlib.util.spec_from_file_location("eval_labeler", SCRIPT)
ev = importlib.util.module_from_spec(_spec)
sys.modules["eval_labeler"] = ev
_spec.loader.exec_module(ev)


class FakeClient:
    model = "fake-model"

    def __init__(self, *script):
        self.script = list(script)
        self.texts = []

    def complete(self, text):
        self.texts.append(text)
        step = self.script.pop(0) if self.script else LLMResult("positive", 10)
        if isinstance(step, Exception):
            raise step
        return step


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def run(client, reviews, tmp_path, interval=0.0, clock=None):
    clock = clock or Clock()
    return ev.label_reviews(
        client,
        reviews,
        progress_path=tmp_path / "progress.jsonl",
        interval=interval,
        sleep=clock.sleep,
        clock=clock,
        say=lambda *_: None,
    )


REVIEWS = [(1, "love it"), (2, "hate it"), (3, "so so")]


# ------------------------------------------------------------ calling the labeller


def test_a_good_answer_is_kept_and_a_malformed_one_is_counted_not_guessed(tmp_path):
    client = FakeClient(
        LLMResult("Positive.", 12), LLMResult("I think it is negative", 30), LLMResult("", 5)
    )

    results = run(client, REVIEWS, tmp_path)

    assert results[1]["status"] == "ok" and results[1]["label"] == "positive"
    assert results[2]["status"] == "invalid" and results[2]["label"] is None
    assert results[3]["status"] == "invalid"


def test_a_refused_request_is_recorded_and_the_run_goes_on(tmp_path):
    client = FakeClient(PermanentError("HTTP 400"), LLMResult("negative", 5))

    results = run(client, REVIEWS[:2], tmp_path)

    assert results[1]["status"] == "refused"
    assert results[2]["label"] == "negative"


def test_rate_limit_waits_and_asks_again_for_the_same_review(tmp_path):
    clock = Clock()
    client = FakeClient(RateLimitError(20), LLMResult("positive", 5))

    results = run(client, REVIEWS[:1], tmp_path, clock=clock)

    assert results[1]["label"] == "positive"
    assert len(client.texts) == 2
    assert 20 in clock.sleeps


def test_network_trouble_backs_off_and_retries(tmp_path):
    clock = Clock()
    client = FakeClient(TransientError("down"), TransientError("down"), LLMResult("negative", 5))

    results = run(client, REVIEWS[:1], tmp_path, clock=clock)

    assert results[1]["label"] == "negative"
    assert clock.sleeps == [30.0, 60.0]


def test_gives_up_after_too_many_failures_and_keeps_earlier_answers(tmp_path):
    client = FakeClient(LLMResult("positive", 5), *[TransientError("down")] * ev.MAX_RETRIES)

    with pytest.raises(ev.EvalAborted):
        run(client, REVIEWS[:2], tmp_path)

    saved = ev.load_progress(tmp_path / "progress.jsonl", "fake-model", ev.PROMPT_VERSION)
    assert list(saved) == [1]


def test_a_bad_key_stops_at_once(tmp_path):
    client = FakeClient(AuthError("the API refused the key (HTTP 401)"))

    with pytest.raises(ev.EvalAborted, match="401"):
        run(client, REVIEWS, tmp_path)

    assert len(client.texts) == 1


def test_calls_are_spaced_by_the_interval(tmp_path):
    clock = Clock()

    run(FakeClient(), REVIEWS, tmp_path, interval=6.0, clock=clock)

    assert clock.sleeps == [6.0, 6.0]


def test_a_second_run_resumes_and_does_not_ask_again(tmp_path):
    run(FakeClient(), REVIEWS[:2], tmp_path)
    second = FakeClient()

    results = run(second, REVIEWS, tmp_path)

    assert second.texts == ["so so"]
    assert sorted(results) == [1, 2, 3]


def test_progress_from_another_model_or_prompt_version_is_ignored(tmp_path):
    run(FakeClient(), REVIEWS, tmp_path)
    other = FakeClient()
    other.model = "another-model"

    run(other, REVIEWS, tmp_path)

    assert len(other.texts) == 3


def test_damaged_progress_lines_are_skipped(tmp_path):
    path = tmp_path / "progress.jsonl"
    good = {"row_id": 1, "model": "fake-model", "prompt_version": ev.PROMPT_VERSION}
    path.write_text("not json\n" + json.dumps(good) + "\n{}\n", encoding="utf-8")

    assert list(ev.load_progress(path, "fake-model", ev.PROMPT_VERSION)) == [1]


# ---------------------------------------------------------------------- numbers


def record(row_id, label, status="ok", tokens=10):
    return {"row_id": row_id, "status": status, "label": label, "tokens": tokens}


HUMAN = [(1, "positive"), (2, "negative"), (3, "positive"), (4, "negative")]


def test_accuracy_against_the_human_labels_and_unusable_answers_are_separate():
    results = {
        1: record(1, "positive"),
        2: record(2, "positive"),  # wrong
        3: record(3, None, status="invalid"),
        4: record(4, "negative"),
    }

    summary = ev.summarise(HUMAN, results)

    assert summary["complete"] is True
    assert summary["usable_answers"] == 3
    assert summary["invalid_answers"] == 1
    assert summary["usable_share"] == 0.75
    assert summary["labeller_vs_human"]["accuracy"] == pytest.approx(2 / 3)
    assert summary["accuracy_counting_unusable_as_wrong"] == 0.5
    assert summary["total_tokens"] == 40


def test_an_unfinished_run_is_marked_incomplete():
    summary = ev.summarise(HUMAN, {1: record(1, "positive")})

    assert summary["complete"] is False
    assert summary["answered"] == 1


def test_no_usable_answers_gives_no_accuracy():
    summary = ev.summarise(HUMAN, {1: record(1, None, status="invalid")})

    assert summary["labeller_vs_human"] is None


def test_agreement_with_bert_and_who_is_right_when_they_differ():
    results = {
        i: record(i, label)
        for i, label in [(1, "positive"), (2, "positive"), (3, "negative"), (4, "negative")]
    }
    # human: 1 pos, 2 neg, 3 pos, 4 neg.  labeller: 1 ok, 2 wrong, 3 wrong, 4 ok.
    bert = pd.DataFrame(
        {"row_id": [1, 2, 3, 4], "pred": ["positive", "negative", "positive", "positive"]}
    )
    # bert: 1 ok, 2 ok, 3 ok, 4 wrong.

    stats = ev.summarise(HUMAN, results, bert)["labeller_vs_bert"]

    assert stats["reviews"] == 4
    assert stats["agreement_rate"] == 0.25  # only review 1 matches
    assert stats["disagreements"] == 3
    assert stats["labeller_right_when_they_disagree"] == 1  # review 4
    assert stats["bert_right_when_they_disagree"] == 2  # reviews 2 and 3
    assert stats["bert_accuracy_on_these"] == 0.75
    assert stats["labeller_accuracy_on_these"] == 0.5


def test_markdown_shows_the_measured_numbers():
    results = {
        i: record(i, label)
        for i, label in [(1, "positive"), (2, "negative"), (3, "positive"), (4, "negative")]
    }
    meta = {"model": "fake-model", "prompt_version": "v1", "created_at": "2026-11-26T10:00:00"}

    text = ev.render_markdown(ev.summarise(HUMAN, results), meta)

    assert "fake-model" in text and "100.0%" in text
    assert "No BERT predictions" in text


# -------------------------------------------------------------------- output files


def test_outputs_are_written_for_the_usable_answers_only(tmp_path):
    frame = pd.DataFrame(
        {"row_id": [1, 2, 3, 4], "review_text": list("abcd"), "label": [1, 0, 1, 0]}
    )
    results = {
        1: record(1, "positive"),
        2: record(2, None, status="invalid"),
        3: record(3, "negative"),
        4: record(4, "negative"),
    }
    summary = ev.summarise(HUMAN, results)
    meta = {"model": "m", "prompt_version": "v1", "created_at": "now"}

    md, csv = ev.write_outputs(summary, meta, results, frame, tmp_path / "docs", tmp_path / "pred")

    saved = pd.read_csv(csv)
    assert saved["row_id"].tolist() == [1, 3, 4]
    assert saved["pred"].tolist() == ["positive", "negative", "negative"]
    assert json.loads((tmp_path / "docs" / "labeler_eval.json").read_text())["model"] == "m"
    assert md.read_text(encoding="utf-8").startswith("# LLM labeller validation")


def test_the_key_must_come_from_the_environment():
    with pytest.raises(SystemExit):
        ev.build_client_from_env({})

    client = ev.build_client_from_env({"GROQ_API_KEY": "gsk_x", "LABEL_MODEL": "some-model"})
    assert client.model == "some-model"
    assert "gsk_x" not in repr(client)


# ----------------------------------------------- the real row ids are strings like "te385173"


def test_string_row_ids_work_through_the_whole_flow(tmp_path):
    reviews = [("te385173", "love it"), ("te000007", "hate it")]
    client = FakeClient(LLMResult("positive", 5), LLMResult("negative", 5))

    results = run(client, reviews, tmp_path)
    resumed = run(FakeClient(), reviews, tmp_path)  # reads the saved progress back

    assert results["te385173"]["label"] == "positive"
    assert sorted(resumed) == ["te000007", "te385173"]

    human = [("te385173", "positive"), ("te000007", "positive")]
    bert = pd.DataFrame({"row_id": ["te385173", "te000007"], "pred": ["positive", "negative"]})
    summary = ev.summarise(human, results, bert)
    assert summary["complete"] is True
    assert summary["labeller_vs_human"]["accuracy"] == 0.5
    assert summary["labeller_vs_bert"]["agreement_rate"] == 1.0

    frame = pd.DataFrame(
        {"row_id": ["te385173", "te000007"], "review_text": ["a", "b"], "label": [1, 1]}
    )
    _, csv = ev.write_outputs(
        summary,
        {"model": "m", "prompt_version": "v1", "created_at": "now"},
        results,
        frame,
        tmp_path / "docs",
        tmp_path / "pred",
    )
    assert pd.read_csv(csv)["row_id"].tolist() == ["te385173", "te000007"]
