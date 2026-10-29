"""Train/validation and leave-one-topic-out splits."""

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import SEED, TOPICS

STANDARD: str = "standard"
NEW_TOPIC: str = "new_topic"


@dataclass(frozen=True)
class Experiment:
    """One train/test setting.

    ``setting`` is ``standard`` (all topics) or ``new_topic`` (one topic held
    out); ``test_set`` is ``all`` or the name of the held-out topic.
    """

    setting: str
    test_set: str
    train: pd.DataFrame
    test: pd.DataFrame


def validation_split(
    df: pd.DataFrame, val_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation, stratified by label."""
    train, val = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def experiments(train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Experiment]:
    """Yield the standard split, then one leave-one-topic-out split per topic.

    For a held-out topic, the model trains on the other four topics of the
    training split and is tested on that topic's posts in the test split.
    The held-out topic never appears in training or validation.
    """
    yield Experiment(STANDARD, "all", train, test)
    for topic in TOPICS:
        yield Experiment(
            NEW_TOPIC,
            topic,
            train[train["topic"] != topic].reset_index(drop=True),
            test[test["topic"] == topic].reset_index(drop=True),
        )
