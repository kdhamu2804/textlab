"""Tests for metrics and bootstrap confidence intervals."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.evaluation import bootstrap_ci, compute_metrics, evaluate


@pytest.fixture
def labels() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_pred = np.where(rng.random(300) < 0.75, y_true, 1 - y_true)
    return y_true, y_pred


def test_metrics_match_sklearn(labels) -> None:
    y_true, y_pred = labels
    m = compute_metrics(y_true, y_pred)
    assert m["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert m["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert m["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))


def test_no_positive_predictions_give_zero_f1() -> None:
    m = compute_metrics(np.array([1, 0, 1]), np.array([0, 0, 0]))
    assert m["f1"] == 0.0 and m["precision"] == 0.0


def test_ci_contains_point_estimate_and_is_repeatable(labels) -> None:
    y_true, y_pred = labels
    ci = bootstrap_ci(y_true, y_pred)
    point = compute_metrics(y_true, y_pred)
    for name, (low, high) in ci.items():
        assert low <= point[name] <= high
    assert ci == bootstrap_ci(y_true, y_pred)


def test_evaluate_has_bounds_for_every_metric(labels) -> None:
    row = evaluate(*labels)
    assert row["n"] == 300
    for name in ("f1", "precision", "recall", "accuracy"):
        assert row[f"{name}_low"] <= row[name] <= row[f"{name}_high"]
