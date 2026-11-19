"""Temperature scaling (Guo et al., 2017): one scalar that softens or sharpens
a classifier's probabilities without changing any predicted label."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of ``labels`` under ``softmax(logits / T)``."""
    log_probs = log_softmax(np.asarray(logits, dtype=float) / temperature, axis=-1)
    return float(-np.mean(log_probs[np.arange(len(labels)), np.asarray(labels)]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the temperature T > 0 that minimizes NLL on held-out data."""
    result = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))), bounds=(-3, 3), method="bounded"
    )
    return float(np.exp(result.x))


def stress_probability(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(stress) for each row of two-class logits after temperature scaling."""
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=-1)[:, 1]
