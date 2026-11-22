"""Build the drift reference snapshot from the training split.

    python scripts/prepare_data.py            # once, if data/processed/ does not exist yet
    python scripts/build_drift_reference.py

Writes data/reference/reference.json (about 60 KB; commit it, the Docker image needs it).
Then replays the validation split through the detector in windows, to show how often clean data
would raise a false alarm. For that check the true labels stand in for predictions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import drift  # noqa: E402
from src.common import load_split  # noqa: E402

logger = logging.getLogger("reviewpulse.drift_reference")


def false_alarm_check(
    reference: drift.Reference, frame: pd.DataFrame, window: int, threshold: float
):
    """Replay clean data in back-to-back windows and count how many would be flagged."""
    positive = {1, "1", "positive"}
    rows = [
        drift.feature_row(text, "positive" if label in positive else "negative", reference.vocab)
        for text, label in zip(frame["review_text"], frame["label"], strict=True)
    ]
    windows = [rows[i : i + window] for i in range(0, len(rows) - window + 1, window)]
    reports = [
        drift.evaluate_rows(
            reference, chunk, min_window=window, window_size=window, psi_threshold=threshold
        )
        for chunk in windows
    ]
    flagged = sum(report.status == "drift" for report in reports)
    worst = {
        name: max((report.features[name].psi for report in reports), default=0.0)
        for name in drift.FEATURES
    }
    return len(reports), flagged, worst


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the data drift reference snapshot.")
    parser.add_argument("--csv", type=Path, help="CSV with review_text and label (default: train)")
    parser.add_argument("--check-csv", type=Path, help="CSV to replay (default: validation)")
    parser.add_argument(
        "--sample", action="store_true", help="use data/sample instead of processed"
    )
    parser.add_argument("--out", type=Path, default=drift.REFERENCE_PATH)
    parser.add_argument("--bins", type=int, default=drift.DEFAULT_BINS)
    parser.add_argument("--vocab-size", type=int, default=drift.DEFAULT_VOCAB_SIZE)
    parser.add_argument("--window-size", type=int, default=drift.DEFAULT_WINDOW_SIZE)
    parser.add_argument("--threshold", type=float, default=drift.DEFAULT_PSI_THRESHOLD)
    args = parser.parse_args(argv)

    train = pd.read_csv(args.csv) if args.csv else load_split("train", sample=args.sample)
    data = drift.build_reference(
        train["review_text"].astype(str).tolist(),
        train["label"].tolist(),
        bins=args.bins,
        vocab_size=args.vocab_size,
        source=str(args.csv) if args.csv else "train split",
    )
    path = drift.save_reference(data, args.out)
    logger.info(
        "Wrote %s from %d reviews (vocabulary %d words)", path, len(train), data["vocab_size"]
    )
    for name in drift.FEATURES:
        spec = data["features"][name]
        logger.info(
            "  %-16s mean %.3f std %.3f edges %s", name, spec["mean"], spec["std"], spec["edges"]
        )

    try:
        check = (
            pd.read_csv(args.check_csv)
            if args.check_csv
            else load_split("validation", sample=args.sample)
        )
    except FileNotFoundError:
        logger.warning("No validation split found, skipping the false alarm check")
        return 0

    reference = drift.load_reference(path)
    total, flagged, worst = false_alarm_check(reference, check, args.window_size, args.threshold)
    if total == 0:
        logger.warning(
            "Validation has fewer than %d reviews, skipping the false alarm check", args.window_size
        )
        return 0
    logger.info(
        "False alarm check: %d of %d clean windows of %d reviews flagged at PSI %s",
        flagged,
        total,
        args.window_size,
        args.threshold,
    )
    for name, value in worst.items():
        logger.info("  highest PSI on clean data, %-16s %.3f", name, value)
    if flagged:
        logger.warning("Clean data is being flagged. Raise DRIFT_PSI_THRESHOLD or the window size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
