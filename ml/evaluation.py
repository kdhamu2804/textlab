"""Classification metrics with 95% bootstrap confidence intervals."""

import numpy as np

from ml.config import SEED

METRICS: tuple[str, ...] = ("f1", "precision", "recall", "accuracy")


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide element-wise, returning 0 where the denominator is 0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Compute metrics for the stress class (label 1) along the last axis.

    Works for 1-D arrays (one evaluation) and 2-D arrays (one row per
    bootstrap resample), so point estimates and CIs share one formula.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=-1)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=-1)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=-1)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "precision": precision,
        "recall": recall,
        "accuracy": np.mean(y_true == y_pred, axis=-1),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return F1, precision and recall for the stress class, plus accuracy."""
    scores = _metrics_from_arrays(np.asarray(y_true), np.asarray(y_pred))
    return {name: float(value) for name, value in scores.items()}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """Return percentile bootstrap CIs (default 95%) for every metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    samples = _metrics_from_arrays(y_true[idx], y_pred[idx])
    bounds = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {
        name: tuple(float(b) for b in np.percentile(values, bounds))
        for name, values in samples.items()
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a flat dict of each metric with its ``_low``/``_high`` CI bounds."""
    point = compute_metrics(y_true, y_pred)
    ci = bootstrap_ci(y_true, y_pred)
    row: dict[str, float] = {"n": float(len(y_true))}
    for name in METRICS:
        row[name] = point[name]
        row[f"{name}_low"], row[f"{name}_high"] = ci[name]
    return row
