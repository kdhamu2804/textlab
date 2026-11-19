"""Deeper analysis of saved predictions: confusion matrices, significance tests
and probability calibration."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ml.config import SEED
from ml.evaluation import _metrics_from_arrays

N_SIGNIFICANCE_RESAMPLES: int = 10_000
N_CALIBRATION_BINS: int = 10


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Counts of true/false positives and negatives (stress = positive class)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "f1",
    n_resamples: int = N_SIGNIFICANCE_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired bootstrap test of ``metric(b) - metric(a)`` on the same test posts.

    Both models are scored on the same resampled posts, so the test accounts
    for the posts they share. ``p_value`` is one-sided: the share of resamples
    in which model b is *not* better than model a.
    """
    y_true, pred_a, pred_b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    diff = (
        _metrics_from_arrays(y_true[idx], pred_b[idx])[metric]
        - _metrics_from_arrays(y_true[idx], pred_a[idx])[metric]
    )
    observed = (
        _metrics_from_arrays(y_true, pred_b)[metric] - _metrics_from_arrays(y_true, pred_a)[metric]
    )
    low, high = np.percentile(diff, (2.5, 97.5))
    return {
        "metric": metric,
        "difference": float(observed),
        "difference_low": float(low),
        "difference_high": float(high),
        "p_value": float(max(np.mean(diff <= 0), 1 / n_resamples)),
        "n_resamples": n_resamples,
    }


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test: do the two models make different numbers of errors?

    ``only_a_correct`` and ``only_b_correct`` count the posts where exactly one
    model is right; under the null hypothesis each is equally likely.
    """
    y_true, pred_a, pred_b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "mcnemar_p": float(p)}


def compare_models(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, name_a: str, name_b: str
) -> dict[str, float | str]:
    """Paired bootstrap on F1 plus McNemar's test for model b versus model a."""
    return {
        "model_a": name_a,
        "model_b": name_b,
        **paired_bootstrap(y_true, pred_a, pred_b),
        **mcnemar(y_true, pred_a, pred_b),
    }


def calibration_bins(
    y_true: np.ndarray, stress_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS
) -> pd.DataFrame:
    """Reliability table: mean predicted P(stress) vs observed stress rate per bin.

    Bins are equal-width on [0, 1]; empty bins are dropped.
    """
    y_true, stress_prob = np.asarray(y_true, dtype=float), np.asarray(stress_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(stress_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            rows.append({
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(stress_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            })
    return pd.DataFrame(rows)


def calibration_summary(
    y_true: np.ndarray, stress_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS
) -> dict[str, float]:
    """Expected calibration error (ECE) and Brier score.

    ECE is the post-weighted average gap between predicted probability and
    observed stress rate across bins (0 = perfectly calibrated).
    """
    y_true, stress_prob = np.asarray(y_true, dtype=float), np.asarray(stress_prob, dtype=float)
    bins = calibration_bins(y_true, stress_prob, n_bins)
    ece = float(np.sum(bins["n"] * (bins["mean_predicted"] - bins["observed_rate"]).abs()) / len(y_true))
    return {"ece": ece, "brier": float(np.mean((stress_prob - y_true) ** 2)), "n": len(y_true)}
