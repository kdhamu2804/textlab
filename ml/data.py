"""Download, clean and load the Dreaddit dataset."""

from pathlib import Path

import pandas as pd

from ml.config import (
    COLUMNS,
    HF_DATASET,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS,
    SUBREDDIT_TO_TOPIC,
)


def raw_path(split: str) -> Path:
    """Return the path of a raw split CSV."""
    return RAW_DIR / f"dreaddit_{split}.csv"


def processed_path(split: str) -> Path:
    """Return the path of a processed split CSV."""
    return PROCESSED_DIR / f"{split}.csv"


def download_raw(overwrite: bool = False) -> None:
    """Download both Dreaddit splits from Hugging Face and save them as CSV."""
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HF_DATASET)
    for split in SPLITS:
        path = raw_path(split)
        if path.exists() and not overwrite:
            continue
        dataset[split].to_pandas().to_csv(path, index=False)


def add_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``topic`` column; raise if any subreddit has no topic mapping."""
    topics = df["subreddit"].map(SUBREDDIT_TO_TOPIC)
    unknown = sorted(df.loc[topics.isna(), "subreddit"].unique())
    if unknown:
        raise ValueError(f"Subreddits without a topic mapping: {unknown}")
    return df.assign(topic=topics)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the needed columns, enforce types and add the topic column."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[COLUMNS].copy()
    out["id"] = out["id"].astype(str)
    if out["id"].duplicated().any():
        raise ValueError("Post ids must be unique")
    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    out["subreddit"] = out["subreddit"].astype(str).str.lower()
    out["confidence"] = out["confidence"].astype(float)
    return add_topic(out).reset_index(drop=True)


def prepare_split(split: str) -> pd.DataFrame:
    """Clean one raw split and write it to ``data/processed``."""
    df = clean(pd.read_csv(raw_path(split)))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path(split), index=False)
    return df


def load_split(split: str) -> pd.DataFrame:
    """Load a processed split (``train`` or ``test``)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    return pd.read_csv(processed_path(split), dtype={"id": str})


def topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return post counts and percent stress per topic, with a total row."""
    grouped = df.groupby("topic")["label"].agg(posts="size", stress_pct="mean")
    total = pd.DataFrame(
        {"posts": [len(df)], "stress_pct": [df["label"].mean()]}, index=["ALL"]
    )
    summary = pd.concat([grouped, total])
    summary["stress_pct"] = (summary["stress_pct"] * 100).round(1)
    return summary
