"""Data preparation: download, clean, validate, split, save.

Run:  python -m src.prepare_data            (writes data/processed/)
      python -m src.prepare_data --fixture  (also writes the small data/sample/ fixture)

Decisions (see README): content only, no stratification, duplicates removed before
sampling, train and validation come from the official train split, test comes from
the official test split, seed 42.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import (
    DATA_COLUMNS,
    DATASET_NAME,
    DATASET_REVISION,
    PROCESSED_DIR,
    SAMPLE_DIR,
    SEED,
    SPLITS,
    TEST_ORDER_FILE,
)

N_TRAIN = 40_000
N_VALIDATION = 10_000
N_TEST = 10_000
FIXTURE_SIZES = {"train": 100, "validation": 50, "test": 50}


# --------------------------------------------------------------- loading


def resolve_revision(revision: str | None) -> str:
    """Use the pinned revision if given, otherwise look up the current commit hash."""
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(DATASET_NAME).sha
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not resolve dataset revision ({exc}). Recording 'unknown'.")
        return "unknown"


def load_raw(revision: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Download the official train and test splits (columns: label, title, content).

    Returns (train, test, revision). Pass a revision to get exactly that dataset version.
    """
    from datasets import load_dataset  # imported here so tests do not need the library

    revision = resolve_revision(revision)
    kwargs = {} if revision == "unknown" else {"revision": revision}
    ds = load_dataset(DATASET_NAME, **kwargs)
    return ds["train"].to_pandas(), ds["test"].to_pandas(), revision


# -------------------------------------------------------------- cleaning


def clean_reviews(raw: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, dict]:
    """Validate labels, drop missing/empty content, drop exact duplicate reviews.

    row_id is f"{prefix}{position in the raw split}", so it stays stable and traceable.
    Output columns: row_id, review_text, label. The title column is not carried over.
    """
    required = {"content", "label"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Raw data needs columns {required}, got {list(raw.columns)}")

    df = raw.reset_index(drop=True)[["content", "label"]].copy()
    df["row_id"] = [f"{prefix}{i}" for i in range(len(df))]

    if df["label"].isna().any() or not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be only 0 or 1 with no missing values.")
    df["label"] = df["label"].astype(int)

    n_original = len(df)
    missing = df["content"].isna()
    n_missing = int(missing.sum())
    df = df[~missing]

    empty = df["content"].str.strip() == ""
    n_empty = int(empty.sum())
    df = df[~empty]

    dup = df.duplicated(subset="content", keep="first")
    n_duplicates = int(dup.sum())
    df = df[~dup]

    df = df.rename(columns={"content": "review_text"})[DATA_COLUMNS].reset_index(drop=True)
    report = {
        "original_rows": n_original,
        "missing_removed": n_missing,
        "empty_removed": n_empty,
        "duplicates_removed": n_duplicates,
        "rows_after_cleaning": len(df),
    }
    return df, report


# ------------------------------------------------------------- splitting


def build_splits(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    seed: int = SEED,
    n_train: int = N_TRAIN,
    n_validation: int = N_VALIDATION,
    n_test: int = N_TEST,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Clean both official splits, then sample. No stratification.

    1. Clean each official split (missing, empty, duplicates).
    2. Remove from the test pool any review that also appears in the train pool.
    3. Sample n_train + n_validation rows from the train pool, n_test from the test pool.
    """
    train_pool, train_report = clean_reviews(train_raw, "tr")
    test_pool, test_report = clean_reviews(test_raw, "te")

    seen = set(train_pool["review_text"])
    overlap = test_pool["review_text"].isin(seen)
    n_cross = int(overlap.sum())
    test_pool = test_pool[~overlap]

    need = n_train + n_validation
    if len(train_pool) < need:
        raise ValueError(f"Train pool has {len(train_pool)} rows, need {need}.")
    if len(test_pool) < n_test:
        raise ValueError(f"Test pool has {len(test_pool)} rows, need {n_test}.")

    sampled = train_pool.sample(n=need, random_state=seed).reset_index(drop=True)
    splits = {
        "train": sampled.iloc[:n_train].reset_index(drop=True),
        "validation": sampled.iloc[n_train:].reset_index(drop=True),
        "test": test_pool.sample(n=n_test, random_state=seed).reset_index(drop=True),
    }
    validate_splits(splits)

    report = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "sampling": "plain random sampling, no stratification, duplicates removed before sampling",
        "input_column": "content (title is not used)",
        "official_train_cleaning": train_report,
        "official_test_cleaning": test_report,
        "test_rows_removed_for_overlap_with_train": n_cross,
        "final_sizes": {name: len(df) for name, df in splits.items()},
        "label_counts": {
            name: {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
            for name, df in splits.items()
        },
    }
    return splits, report


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Raise ValueError if the splits break any pipeline rule."""
    if set(splits) != set(SPLITS):
        raise ValueError(f"Expected splits {SPLITS}, got {tuple(splits)}")

    all_ids: list[str] = []
    for name, df in splits.items():
        if list(df.columns) != DATA_COLUMNS:
            raise ValueError(f"{name}: columns must be {DATA_COLUMNS}, got {list(df.columns)}")
        if df["review_text"].isna().any() or (df["review_text"].str.strip() == "").any():
            raise ValueError(f"{name}: missing or empty review_text.")
        if not df["label"].isin([0, 1]).all():
            raise ValueError(f"{name}: labels must be only 0 or 1.")
        if df["review_text"].duplicated().any():
            raise ValueError(f"{name}: duplicate reviews inside the split.")
        all_ids.extend(df["row_id"].tolist())

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("row_id values are not unique across splits.")

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(splits[a]["review_text"]) & set(splits[b]["review_text"])
            if shared:
                raise ValueError(f"{len(shared)} reviews appear in both {a} and {b}.")


# ---------------------------------------------------------------- saving


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path, seed: int = SEED) -> None:
    """Write train/validation/test CSVs and test_order.csv (fixed shuffled test order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    order = splits["test"].sample(frac=1.0, random_state=seed)[["row_id"]]
    order.to_csv(out_dir / TEST_ORDER_FILE, index=False)


def make_fixture(
    splits: dict[str, pd.DataFrame],
    out_dir: Path = SAMPLE_DIR,
    sizes: dict[str, int] | None = None,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Small subset of the splits for tests, CI and development. Safe to commit."""
    sizes = sizes or FIXTURE_SIZES
    small = {
        name: df.sample(n=min(sizes[name], len(df)), random_state=seed).reset_index(drop=True)
        for name, df in splits.items()
    }
    validate_splits(small)
    write_splits(small, out_dir, seed=seed)
    return small


# ------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReviewPulse data splits.")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--revision", default=DATASET_REVISION, help="dataset commit hash")
    parser.add_argument("--fixture", action="store_true", help="also write data/sample/")
    args = parser.parse_args()

    train_raw, test_raw, revision = load_raw(args.revision)
    splits, report = build_splits(train_raw, test_raw, seed=args.seed)
    report["dataset_revision"] = revision
    write_splits(splits, args.out, seed=args.seed)
    (args.out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.fixture:
        make_fixture(splits, SAMPLE_DIR, seed=args.seed)
        print(f"Fixture written to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
