"""Send traffic to a running API so the drift detector and the email alert can be shown live.

    python scripts/simulate_drift.py --url http://localhost:8000 --mode normal --count 150
    python scripts/simulate_drift.py --url http://localhost:8000 --mode shifted --count 150

normal   ordinary English reviews (from --csv, else data/processed/validation.csv, else built-in)
shifted  very short text, emoji, Urdu, Hindi, Chinese and Arabic, very long text, and codes
mixed    half of each

After the requests it asks GET /drift for the verdict, if the API has that endpoint.
Uses only the standard library. Exit code 1 when no request succeeded.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("reviewpulse.simulate_drift")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "processed" / "validation.csv"

NORMAL_REVIEWS = [
    "I bought this for my son and he loves it. It works well and arrived on time.",
    "Stopped working after two weeks. Customer service never answered my emails.",
    "Great value for the price. The battery lasts all day and the screen is bright.",
    "The book was a slow read but the ending made it worth it. Would recommend to friends.",
    "Terrible quality. The zipper broke the first time I used it and the fabric feels cheap.",
    "Exactly what I needed. Easy to set up, sturdy, and it looks nice on my desk.",
    "I expected more. The sound is flat and the case scratches easily, not worth the money.",
    "Fast shipping and well packed. The coffee tastes fresh and the flavor is smooth.",
    "My dog chewed through it in a day. Waste of money, I would not buy this again.",
    "Fits perfectly and feels comfortable all day. I have already ordered a second pair.",
    "The instructions were confusing and a part was missing, but support sent a replacement.",
    "Beautiful design and very easy to clean. My kids use it every single day.",
]

SHORT = ["ok", "👍👍👍", "10/10", "meh", "❤️🔥", "bad", "good"]
OTHER_SCRIPTS = [
    "یہ پروڈکٹ بہت اچھی ہے اور میں اسے دوبارہ خریدوں گا",
    "यह उत्पाद बहुत अच्छा है और मैं इसे फिर से खरीदूंगा",
    "这个产品非常好，质量很棒，我会再次购买",
    "هذا المنتج رائع جدا وسأشتريه مرة أخرى",
]
CODES = [
    "SKU 88231 qty 12 lot 4471 batch B-77 ref 0099321 ship 2026-11-21",
    "ERR_CODE 0x80070005 module 7741 rev 12 build 20261121 id 99-1234-5567",
]
LONG_BASE = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "


def shifted_reviews(count: int, rng: random.Random) -> list[str]:
    """Text that looks nothing like the training reviews."""
    long_text = (LONG_BASE * 60)[:4000]
    pools = [SHORT, OTHER_SCRIPTS, [long_text], CODES]
    return [rng.choice(pools[i % len(pools)]) for i in range(count)]


def normal_reviews(count: int, rng: random.Random, csv_path: Path | None = None) -> list[str]:
    path = csv_path or DEFAULT_CSV
    if path.is_file():
        import pandas as pd

        frame = pd.read_csv(path)
        texts = [t for t in frame["review_text"].astype(str) if t.strip()]
        if texts:
            return [rng.choice(texts)[:4000] for _ in range(count)]
    return [rng.choice(NORMAL_REVIEWS) for _ in range(count)]


def build_traffic(mode: str, count: int, seed: int = 42, csv_path: Path | None = None) -> list[str]:
    rng = random.Random(seed)
    if mode == "normal":
        return normal_reviews(count, rng, csv_path)
    if mode == "shifted":
        return shifted_reviews(count, rng)
    if mode == "mixed":
        half = count // 2
        texts = normal_reviews(half, rng, csv_path) + shifted_reviews(count - half, rng)
        rng.shuffle(texts)
        return texts
    raise ValueError(f"Unknown mode {mode!r}")


def request_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Send normal or shifted reviews to the API.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--mode", choices=["normal", "shifted", "mixed"], default="shifted")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, help="CSV with review_text for the normal traffic")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    texts = build_traffic(args.mode, args.count, args.seed, args.csv)
    logger.info("Sending %d %s reviews to %s/predict", len(texts), args.mode, base)

    sent = failed = 0
    for index, text in enumerate(texts, start=1):
        try:
            request_json(f"{base}/predict", {"text": text})
            sent += 1
        except (urllib.error.URLError, OSError, ValueError) as error:
            failed += 1
            logger.warning("Request %d failed: %s", index, error)
        if index % 25 == 0:
            logger.info("%d of %d sent", index, len(texts))
        if args.delay > 0:
            time.sleep(args.delay)

    logger.info("Done: %d succeeded, %d failed", sent, failed)

    try:
        report = request_json(f"{base}/drift")
        logger.info(
            "GET /drift: status=%s window=%s drifted=%s",
            report.get("status"),
            report.get("window"),
            report.get("drifted_features"),
        )
    except urllib.error.HTTPError as error:
        logger.info("GET /drift answered %s (the endpoint may not be wired in yet)", error.code)
    except (urllib.error.URLError, OSError, ValueError) as error:
        logger.info("Could not read /drift: %s", error)

    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
