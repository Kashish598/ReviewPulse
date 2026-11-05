import pandas as pd
import pytest

from src.common import DATA_COLUMNS, SPLITS, load_split, load_test_subset
from src.prepare_data import (
    build_splits,
    clean_reviews,
    make_fixture,
    validate_splits,
    write_splits,
)


def raw_df(n: int, offset: int = 0, label_pattern=(0, 1)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [label_pattern[i % len(label_pattern)] for i in range(n)],
            "title": [f"title {i}" for i in range(n)],
            "content": [f"review text number {i + offset}" for i in range(n)],
        }
    )


def test_clean_removes_missing_empty_and_duplicates():
    raw = pd.DataFrame(
        {
            "label": [0, 1, 1, 0, 1],
            "title": list("abcde"),
            "content": ["good", None, "   ", "good", "bad"],
        }
    )
    clean, report = clean_reviews(raw, "tr")
    assert clean["review_text"].tolist() == ["good", "bad"]
    assert report["missing_removed"] == 1
    assert report["empty_removed"] == 1
    assert report["duplicates_removed"] == 1
    assert list(clean.columns) == DATA_COLUMNS


def test_clean_keeps_original_text_exactly():
    raw = pd.DataFrame({"label": [1], "title": ["t"], "content": ["Great!!! :) THE Best."]})
    clean, _ = clean_reviews(raw, "tr")
    assert clean["review_text"].iloc[0] == "Great!!! :) THE Best."


def test_clean_rejects_invalid_labels():
    raw = pd.DataFrame({"label": [0, 2], "title": ["a", "b"], "content": ["x", "y"]})
    with pytest.raises(ValueError):
        clean_reviews(raw, "tr")


def test_label_mapping_is_preserved():
    raw = pd.DataFrame({"label": [0, 1], "title": ["a", "b"], "content": ["x", "y"]})
    clean, _ = clean_reviews(raw, "tr")
    assert clean["label"].tolist() == [0, 1]


def small_splits(seed=42):
    return build_splits(
        raw_df(200), raw_df(100, offset=1000), seed=seed, n_train=100, n_validation=40, n_test=30
    )


def test_split_sizes():
    splits, report = small_splits()
    assert {k: len(v) for k, v in splits.items()} == {"train": 100, "validation": 40, "test": 30}
    assert report["final_sizes"]["train"] == 100


def test_no_overlap_between_splits():
    splits, _ = small_splits()
    validate_splits(splits)  # raises if any overlap, bad label, or duplicate row_id


def test_reproducible_with_same_seed():
    a, _ = small_splits(seed=42)
    b, _ = small_splits(seed=42)
    for name in SPLITS:
        assert a[name].equals(b[name])


def test_different_seed_changes_sample():
    a, _ = small_splits(seed=42)
    b, _ = small_splits(seed=7)
    assert not a["train"].equals(b["train"])


def test_test_rows_overlapping_train_are_removed():
    train_raw = raw_df(200)
    test_raw = pd.concat([raw_df(20, offset=0), raw_df(50, offset=5000)], ignore_index=True)
    splits, report = build_splits(train_raw, test_raw, n_train=100, n_validation=40, n_test=30)
    assert report["test_rows_removed_for_overlap_with_train"] == 20
    train_val_text = set(splits["train"]["review_text"]) | set(splits["validation"]["review_text"])
    assert not train_val_text & set(splits["test"]["review_text"])


def test_validate_splits_catches_overlap():
    splits, _ = small_splits()
    splits["test"].loc[0, "review_text"] = splits["train"].loc[0, "review_text"]
    with pytest.raises(ValueError, match="appear in both"):
        validate_splits(splits)


def test_validate_splits_catches_bad_label():
    splits, _ = small_splits()
    splits["test"].loc[0, "label"] = 3
    with pytest.raises(ValueError, match="labels"):
        validate_splits(splits)


def test_not_enough_rows_raises():
    with pytest.raises(ValueError):
        build_splits(raw_df(10), raw_df(10, offset=500), n_train=100, n_validation=40, n_test=30)


def test_fixture_roundtrip_and_test_subset(tmp_path, monkeypatch):
    import src.common as common

    splits, _ = small_splits()
    sizes = {"train": 20, "validation": 10, "test": 12}
    small = make_fixture(splits, out_dir=tmp_path, sizes=sizes)
    monkeypatch.setattr(common, "SAMPLE_DIR", tmp_path)
    assert load_split("train", sample=True).equals(small["train"])
    subset5 = load_test_subset(5, sample=True)
    subset8 = load_test_subset(8, sample=True)
    assert subset5["row_id"].tolist() == subset8["row_id"].tolist()[:5]
    with pytest.raises(ValueError):
        load_test_subset(13, sample=True)


def test_write_splits_creates_files(tmp_path):
    splits, _ = small_splits()
    write_splits(splits, tmp_path)
    for name in SPLITS:
        assert (tmp_path / f"{name}.csv").exists()
    assert (tmp_path / "test_order.csv").exists()
