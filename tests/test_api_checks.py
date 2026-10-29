"""Tests for the Phase 7 checks (consistency, latency, error sampling)."""

import numpy as np
import pandas as pd
import pytest

from ml.api_checks import compare_with_offline, latency_summary, sample_errors


@pytest.fixture
def offline() -> pd.DataFrame:
    return pd.DataFrame({"id": ["a", "b", "c", "d"], "pred": [1, 0, 1, 0],
                         "stress_prob": [0.9, 0.2, 0.6, 0.4]}).set_index("id")


def test_compare_with_offline_counts_matches(offline) -> None:
    api = pd.DataFrame({"id": ["a", "b", "c", "d"], "api_pred": [1, 0, 0, 0],
                        "api_stress_prob": [0.9, 0.2, 0.45, 0.4]})
    check = compare_with_offline(api, offline, "tfidf")
    assert check["n_label_match"] == 3 and check["label_match_pct"] == 75
    assert check["max_prob_diff"] == pytest.approx(0.15)


def test_compare_with_offline_rejects_unknown_ids(offline) -> None:
    api = pd.DataFrame({"id": ["zzz"], "api_pred": [1], "api_stress_prob": [0.9]})
    with pytest.raises(ValueError):
        compare_with_offline(api, offline, "tfidf")


def test_latency_summary() -> None:
    summary = latency_summary(np.arange(1, 101, dtype=float), np.full(100, 5.0), "tfidf")
    assert summary["server_mean_ms"] == pytest.approx(50.5)
    assert summary["server_p95_ms"] == pytest.approx(95.05)
    assert summary["meets_goal"] is True


def test_sample_errors_only_returns_mistakes(offline) -> None:
    test = pd.DataFrame({"id": ["a", "b", "c", "d"], "topic": ["ptsd"] * 4,
                         "text": list("wxyz"), "label": [1, 1, 0, 0]})
    errors = sample_errors(test, offline, n=50)
    assert sorted(errors["id"]) == ["b", "c"]
    assert set(errors["error_type"]) == {"false negative", "false positive"}
    assert errors.loc[errors["id"] == "c", "confidence"].item() == pytest.approx(0.6)
    assert errors["group"].isna().all()
