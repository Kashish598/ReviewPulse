"""Fresh review texts for the retraining demo (M4): reviews the project has never seen.

The retraining loop needs NEW reviews. Reviews from the train, validation or test split are
dropped by src/retrain.py (they are not new, and test reviews must never be trained on), so the
validation reviews that scripts/simulate_drift.py uses for normal traffic do not work for this.

This script takes reviews from the official Amazon polarity train pool that are in none of the
three splits, keeps only the text (the human label is NOT used: the LLM labeller supplies the
label, which is the point of the loop), and either writes them to a CSV or sends each one exactly
once to /predict, where they are saved and labelled in the background.

Run:  python scripts/fresh_reviews.py --count 120                      (writes a CSV)
      python scripts/fresh_reviews.py --count 120 --send               (sends them to the API)
      python scripts/fresh_reviews.py --count 120 --send --url http://localhost:8000 --delay 0.5

Start the API first (with DATABASE_URL and GROQ_API_KEY set) so the requests are saved and
labelled. The labeller spends about 6 seconds per review.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import DATASET_REVISION, PROCESSED_DIR, load_split  # noqa: E402
from src.prepare_data import clean_reviews, load_raw  # noqa: E402

logger = logging.getLogger("reviewpulse.fresh_reviews")

DEFAULT_OUT = PROCESSED_DIR / "fresh_reviews.csv"
MAX_CHARS = 1000  # keeps the labeller's token use per review small (the API accepts 5000)


def fresh_reviews(
    train_raw: pd.DataFrame,
    known_texts: set[str],
    count: int,
    seed: int = 7,
    max_chars: int = MAX_CHARS,
) -> pd.DataFrame:
    """`count` review texts from the raw train pool that are not in `known_texts`."""
    pool, _ = clean_reviews(train_raw, "tr")
    texts = pool["review_text"].str.strip()
    keep = ~texts.isin(known_texts) & (texts.str.len() <= max_chars)
    candidates = pool.loc[keep, ["review_text"]].assign(review_text=texts[keep])
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} fresh reviews are available, asked for {count}.")
    return candidates.sample(n=count, random_state=seed).reset_index(drop=True)


def post_prediction(url: str, text: str, timeout: float = 60.0) -> int:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/predict",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def send_reviews(
    texts: Sequence[str],
    url: str,
    delay: float = 0.2,
    post: Callable[[str, str], int] = post_prediction,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, int]:
    """Send each text once. Returns (accepted, failed). Stops early if nothing can be reached."""
    accepted = failed = 0
    for index, text in enumerate(texts, start=1):
        try:
            status = post(url, text)
        except (urllib.error.URLError, OSError, ValueError) as error:
            failed += 1
            logger.warning("Request %d failed: %s", index, error)
            if accepted == 0 and failed >= 3:
                logger.error("Nothing could be reached at %s. Is the API running?", url)
                break
            continue
        if status == 200:
            accepted += 1
        else:
            failed += 1
            logger.warning("Request %d answered HTTP %s", index, status)
        if index % 25 == 0:
            logger.info("%d of %d sent", index, len(texts))
        sleep(delay)
    return accepted, failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fresh reviews for the retraining demo.")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--send", action="store_true", help="send them to /predict as well")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    known = {
        str(text).strip()
        for name in ("train", "validation", "test")
        for text in load_split(name)["review_text"]
    }
    train_raw, _, revision = load_raw(DATASET_REVISION)
    try:
        fresh = fresh_reviews(train_raw, known, args.count, args.seed)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(args.out, index=False)
    logger.info(
        "Wrote %d fresh reviews (dataset revision %s) to %s", len(fresh), revision, args.out
    )

    if args.send:
        accepted, failed = send_reviews(fresh["review_text"].tolist(), args.url, args.delay)
        logger.info("Sent: %d accepted, %d failed", accepted, failed)
        return 0 if accepted and not failed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
