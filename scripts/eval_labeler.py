"""M3: how far can the LLM labeller be trusted? Run it on the human-labelled test reviews.

    python scripts/eval_labeler.py                 # all test reviews (600), about an hour
    python scripts/eval_labeler.py --n 50          # a quick trial: the first 50 in fixed order

It sends each test review to the same Groq model, prompt and settings that src/labeler.py uses in
the app, one call at a time, and keeps the answer only if it is exactly `negative` or `positive`
(a malformed answer is counted, never guessed). Then it reports

  * the labeller's accuracy and macro-F1 against the human labels,
  * how many answers were unusable,
  * how often it agrees with the served BERT model on the same reviews, and, where they disagree,
    which of the two matches the human label.

Nothing here touches the database or the daily cap: it is a one-off batch run with its own
throttle (--interval, default the labeller's 6 s). Every answer is appended to
data/processed/labeler_eval_progress.jsonl (git-ignored), so a stopped or rate-limited run carries
on where it left off when you start it again. Use --fresh to start over. Progress from a different
model or prompt version is ignored.

Needs GROQ_API_KEY in the environment (LABEL_MODEL, LABEL_MAX_TOKENS and LABEL_REASONING_EFFORT are
read too). Results go to docs/labeler_eval.json and docs/labeler_eval.md and the per-review answers
to data/predictions/labeler_test.csv. Partial runs write nothing unless --allow-partial is given.
Never commit a key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    ID_TO_LABEL,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    compute_metrics,
    load_split,
    load_test_subset,
    read_predictions,
    write_predictions,
)
from src.labeler import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_INTERVAL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    PROMPT_VERSION,
    RETIRED_MODELS,
    AuthError,
    ConfigError,
    GroqClient,
    PermanentError,
    RateLimitError,
    TransientError,
    parse_label,
)

DOCS_DIR = ROOT / "docs"
DEFAULT_BERT_PREDICTIONS = PREDICTIONS_DIR / "bert_finetuned_v2_test.csv"
PROGRESS_FILE = PROCESSED_DIR / "labeler_eval_progress.jsonl"
OK, INVALID, REFUSED = "ok", "invalid", "refused"
MAX_RETRIES = 5  # for rate limits and network trouble, per review
MAX_WAIT_SECONDS = 300.0


class EvalAborted(Exception):
    """The run cannot go on (bad key, retired model, too many failures). Progress is kept."""


# ------------------------------------------------------------------ calling the labeller


def load_progress(path: Path, model: str, prompt_version: str) -> dict:
    """Earlier answers from the same model and prompt version. Bad lines are skipped."""
    done: dict = {}
    if not Path(path).exists():
        return done
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record["model"] == model and record["prompt_version"] == prompt_version:
                done[record["row_id"]] = record
        except (ValueError, KeyError, TypeError):
            continue
    return done


def label_reviews(
    client,
    reviews: list[tuple],
    *,
    progress_path: Path,
    interval: float = DEFAULT_MIN_INTERVAL,
    sleep=time.sleep,
    clock=time.monotonic,
    say=print,
) -> dict:
    """Ask the LLM about every review that has no saved answer yet. Returns row_id -> record.

    Raises EvalAborted on a bad key, a missing model or after MAX_RETRIES failures in a row on
    one review; what was done so far is already in the progress file.
    """
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_progress(progress_path, client.model, PROMPT_VERSION)
    todo = [(row_id, text) for row_id, text in reviews if row_id not in results]
    if results:
        say(f"Resuming: {len(results)} answers already saved, {len(todo)} to go.")

    last_call: float | None = None
    with progress_path.open("a", encoding="utf-8") as progress:
        for position, (row_id, text) in enumerate(todo, start=1):
            if last_call is not None:
                wait = interval - (clock() - last_call)
                if wait > 0:
                    sleep(wait)
            record = _ask(client, row_id, text, sleep=sleep, say=say)
            last_call = clock()
            results[row_id] = record
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            if position % 25 == 0 or position == len(todo):
                say(f"  {position}/{len(todo)} done ({_count(results, OK)} usable answers so far)")
    return results


def _count(results: dict, status: str) -> int:
    return sum(1 for record in results.values() if record["status"] == status)


def _ask(client, row_id, text: str, *, sleep, say) -> dict:
    base = {"row_id": row_id, "model": client.model, "prompt_version": PROMPT_VERSION}
    failures = 0
    while True:
        try:
            result = client.complete(text)
        except RateLimitError as error:
            failures += 1
            wait = min(error.retry_after, MAX_WAIT_SECONDS)
            reason = f"rate limited, waiting {wait:.0f}s"
        except TransientError as error:
            failures += 1
            wait = min(30.0 * 2 ** (failures - 1), MAX_WAIT_SECONDS)
            reason = f"{error}, waiting {wait:.0f}s"
        except (AuthError, ConfigError) as error:
            raise EvalAborted(str(error)) from error
        except PermanentError as error:
            return {**base, "status": REFUSED, "label": None, "raw": str(error)[:80], "tokens": 0}
        else:
            label = parse_label(result.raw)
            return {
                **base,
                "status": OK if label else INVALID,
                "label": label,
                "raw": result.raw[:80],
                "tokens": result.tokens,
                "finish_reason": result.finish_reason,
            }
        if failures >= MAX_RETRIES:
            raise EvalAborted(f"review {row_id}: gave up after {failures} failed tries ({reason})")
        say(f"  review {row_id}: {reason}")
        sleep(wait)


# ------------------------------------------------------------------------------- numbers


def summarise(test_rows: list[tuple], results: dict, bert=None) -> dict:
    """The numbers. `test_rows` is (row_id, human label name), `bert` is a predictions frame
    (row_id, pred) for the served model on the same reviews, or None."""
    human = dict(test_rows)
    answered = {i: r for i, r in results.items() if i in human}
    usable = {i: r["label"] for i, r in answered.items() if r["status"] == OK}
    n = len(human)

    summary = {
        "reviews": n,
        "answered": len(answered),
        "complete": len(answered) == n,
        "usable_answers": len(usable),
        "invalid_answers": sum(1 for r in answered.values() if r["status"] == INVALID),
        "refused_requests": sum(1 for r in answered.values() if r["status"] == REFUSED),
        "usable_share": round(len(usable) / n, 4) if n else None,
        "total_tokens": sum(int(r.get("tokens") or 0) for r in answered.values()),
        "labeller_vs_human": None,
        "accuracy_counting_unusable_as_wrong": None,
        "labeller_vs_bert": None,
    }
    if not usable:
        return summary

    ids = sorted(usable)
    metrics = compute_metrics([human[i] for i in ids], [usable[i] for i in ids])
    summary["labeller_vs_human"] = metrics
    correct = sum(1 for i in ids if usable[i] == human[i])
    summary["accuracy_counting_unusable_as_wrong"] = round(correct / len(answered), 4)

    if bert is not None:
        bert_pred = {r: p for r, p in zip(bert["row_id"], bert["pred"], strict=True)}
        shared = [i for i in ids if i in bert_pred]
        if shared:
            agree = sum(1 for i in shared if usable[i] == bert_pred[i])
            disagree = [i for i in shared if usable[i] != bert_pred[i]]
            summary["labeller_vs_bert"] = {
                "reviews": len(shared),
                "agreement_rate": round(agree / len(shared), 4),
                "disagreements": len(disagree),
                "labeller_right_when_they_disagree": sum(
                    1 for i in disagree if usable[i] == human[i]
                ),
                "bert_right_when_they_disagree": sum(
                    1 for i in disagree if bert_pred[i] == human[i]
                ),
                "bert_accuracy_on_these": round(
                    sum(1 for i in shared if bert_pred[i] == human[i]) / len(shared), 4
                ),
                "labeller_accuracy_on_these": round(
                    sum(1 for i in shared if usable[i] == human[i]) / len(shared), 4
                ),
            }
    return summary


def render_markdown(summary: dict, meta: dict) -> str:
    def pct(value) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines = [
        "# LLM labeller validation (M3)",
        "",
        f"Run on {meta['created_at']}. Model `{meta['model']}`, prompt version "
        f"`{meta['prompt_version']}`, temperature 0. Reference: the human labels of the "
        f"{summary['reviews']} test reviews.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Reviews sent | {summary['answered']} of {summary['reviews']} |",
        f"| Usable answers (exactly negative or positive) | {summary['usable_answers']} "
        f"({pct(summary['usable_share'])}) |",
        f"| Unusable answers | {summary['invalid_answers']} |",
        f"| Requests the API refused | {summary['refused_requests']} |",
        f"| Tokens used | {summary['total_tokens']} |",
    ]
    human = summary["labeller_vs_human"]
    if human:
        lines += [
            f"| Labeller accuracy vs human (usable answers) | {pct(human['accuracy'])} |",
            f"| Labeller macro-F1 vs human | {human['macro_f1']:.4f} |",
            f"| Accuracy if unusable answers count as wrong | "
            f"{pct(summary['accuracy_counting_unusable_as_wrong'])} |",
            "",
            "Confusion matrix (rows: human label, columns: labeller; order negative, positive): "
            f"`{human['confusion_matrix']}`",
        ]
    bert = summary["labeller_vs_bert"]
    if bert:
        lines += [
            "",
            f"## Agreement with the served BERT model ({bert['reviews']} reviews)",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Labeller and BERT give the same label | {pct(bert['agreement_rate'])} |",
            f"| Reviews where they differ | {bert['disagreements']} |",
            f"| ...the labeller matches the human label | "
            f"{bert['labeller_right_when_they_disagree']} |",
            f"| ...BERT matches the human label | {bert['bert_right_when_they_disagree']} |",
            f"| BERT accuracy on these reviews | {pct(bert['bert_accuracy_on_these'])} |",
            f"| Labeller accuracy on these reviews | {pct(bert['labeller_accuracy_on_these'])} |",
        ]
    elif human:
        lines += ["", "No BERT predictions were available, so agreement was not computed."]
    lines += [
        "",
        "The labels are LLM opinions. A labeller accuracy below the served model's would mean "
        "that retraining on its labels adds noise, not signal.",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------------------- main


def build_client_from_env(env=os.environ) -> GroqClient:
    key = (env.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise SystemExit("Set GROQ_API_KEY in this window first (never write it in a file).")
    model = (env.get("LABEL_MODEL") or "").strip() or DEFAULT_MODEL
    if model in RETIRED_MODELS:
        print(f"Warning: {model} was shut down by Groq, expect HTTP 404. Use {DEFAULT_MODEL}.")
    effort = (env.get("LABEL_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT).strip().lower()
    try:
        max_tokens = int(env.get("LABEL_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    return GroqClient(
        key,
        model,
        max_tokens=max_tokens,
        reasoning_effort=None if effort in {"", "none", "off"} else effort,
    )


def write_outputs(summary: dict, meta: dict, results: dict, test_frame, docs_dir: Path, pred_dir):
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "labeler_eval.json").write_text(
        json.dumps({**meta, **summary}, indent=2), encoding="utf-8"
    )
    (docs_dir / "labeler_eval.md").write_text(render_markdown(summary, meta), encoding="utf-8")

    ok_ids = [i for i, r in results.items() if r["status"] == OK]
    usable = test_frame[test_frame["row_id"].isin(ok_ids)]
    path = write_predictions(
        Path(pred_dir) / "labeler_test.csv",
        usable["row_id"].tolist(),
        usable["label"].tolist(),
        [results[i]["label"] for i in usable["row_id"]],
    )
    return docs_dir / "labeler_eval.md", path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LLM labeller on the test reviews.")
    parser.add_argument("--n", type=int, default=None, help="only the first N test reviews")
    parser.add_argument("--interval", type=float, default=DEFAULT_MIN_INTERVAL)
    parser.add_argument("--bert-predictions", type=Path, default=DEFAULT_BERT_PREDICTIONS)
    parser.add_argument("--fresh", action="store_true", help="ignore and delete saved progress")
    parser.add_argument("--allow-partial", action="store_true", help="write results if incomplete")
    args = parser.parse_args()

    test = load_test_subset(args.n) if args.n else load_split("test")
    client = build_client_from_env()
    if args.fresh and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    reviews = list(zip(test["row_id"].tolist(), test["review_text"].tolist(), strict=True))
    minutes = len(reviews) * args.interval / 60
    print(
        f"{len(reviews)} reviews, model {client.model}, prompt {PROMPT_VERSION}, "
        f"one call every {args.interval:g} s (about {minutes:.0f} min)."
    )
    try:
        results = label_reviews(
            client, reviews, progress_path=PROGRESS_FILE, interval=args.interval
        )
    except EvalAborted as error:
        print(f"\nStopped: {error}\nAnswers so far are saved. Run the same command to resume.")
        return 1

    bert = None
    if args.bert_predictions.exists():
        bert = read_predictions(args.bert_predictions)
    else:
        print(f"No BERT predictions at {args.bert_predictions}, skipping the agreement part.")

    human = [
        (i, ID_TO_LABEL[int(label)]) for i, label in zip(test["row_id"], test["label"], strict=True)
    ]
    summary = summarise(human, results, bert)
    meta = {
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not summary["complete"] and not args.allow_partial:
        print("The run is incomplete, nothing written (use --allow-partial to override).")
        return 1
    md_path, csv_path = write_outputs(summary, meta, results, test, DOCS_DIR, PREDICTIONS_DIR)
    print("\n" + render_markdown(summary, meta))
    print(f"Saved: {md_path}, {DOCS_DIR / 'labeler_eval.json'}, {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
