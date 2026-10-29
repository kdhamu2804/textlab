"""Tests for the manual label check."""

import pandas as pd
import pytest

from ml.annotation import ANNOTATORS, agreement_report, sample_for_labeling
from ml.data import load_split


def test_sample_is_repeatable_and_unlabelled() -> None:
    train = load_split("train")
    a, b = sample_for_labeling(train), sample_for_labeling(train)
    assert len(a) == 50 and a["id"].is_unique
    assert a["id"].tolist() == b["id"].tolist()
    assert list(a.columns) == ["id", "text", "dataset_label", *ANNOTATORS]
    assert a[ANNOTATORS].isna().all().all()


def test_agreement_report() -> None:
    df = pd.DataFrame({
        "dataset_label": [1, 0, 1, 0],
        "dhamodhar": [1, 0, 1, 0],
        "kushvanth": [1, 1, None, 0],
        "uday": [None, None, None, None],
    })
    report = agreement_report(df).set_index(["rater_a", "rater_b"])
    assert report.loc[("dhamodhar", "dataset_label"), "agreement_pct"] == 100
    assert report.loc[("kushvanth", "dataset_label"), "n"] == 3
    assert report.loc[("uday", "dataset_label"), "n"] == 0


def test_agreement_rejects_bad_values() -> None:
    df = pd.DataFrame({"dataset_label": [1], "dhamodhar": [2], "kushvanth": [None], "uday": [None]})
    with pytest.raises(ValueError, match="only 0, 1"):
        agreement_report(df)
