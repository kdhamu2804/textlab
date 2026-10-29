"""Tests for data cleaning and loading."""

import pandas as pd
import pytest

from ml.config import COLUMNS, TOPICS
from ml.data import add_topic, clean, load_split


def _raw_frame(subreddit: str = "ptsd") -> pd.DataFrame:
    """Build a one-row raw frame with an extra unused column."""
    return pd.DataFrame(
        {"id": ["a1"], "text": ["  hello  "], "label": [1], "subreddit": [subreddit],
         "confidence": [0.8], "extra": [0]}
    )


def test_clean_keeps_columns_and_adds_topic() -> None:
    df = clean(_raw_frame())
    assert list(df.columns) == COLUMNS + ["topic"]
    assert df.loc[0, "text"] == "hello"
    assert df.loc[0, "topic"] == "ptsd"


def test_add_topic_rejects_unknown_subreddit() -> None:
    with pytest.raises(ValueError, match="without a topic"):
        add_topic(_raw_frame("unknown_sub"))


def test_clean_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        clean(_raw_frame().drop(columns=["confidence"]))


@pytest.mark.parametrize(("split", "size"), [("train", 2838), ("test", 715)])
def test_processed_splits(split: str, size: int) -> None:
    df = load_split(split)
    assert len(df) == size
    assert set(df["label"]) == {0, 1}
    assert set(df["topic"]) == set(TOPICS)
    assert df["text"].str.len().gt(0).all()


def test_load_split_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        load_split("validation")
