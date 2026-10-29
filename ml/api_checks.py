"""Phase 7 checks: API vs offline predictions, latency, and an error sample."""

import numpy as np
import pandas as pd

from ml.config import SEED
from ml.predictors import LABEL_NAMES
from ml.splits import STANDARD


def offline_standard_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the standard-split test predictions indexed by post id."""
    return predictions[predictions["setting"] == STANDARD].set_index("id")


def compare_with_offline(api: pd.DataFrame, offline: pd.DataFrame, model_name: str) -> dict:
    """Compare API labels/probabilities with the saved offline predictions.

    ``api`` needs columns ``id``, ``api_pred`` and ``api_stress_prob``.
    """
    merged = api.join(offline[["pred", "stress_prob"]], on="id", how="left")
    if merged["pred"].isna().any():
        raise ValueError("Some API posts have no offline prediction")
    matches = int((merged["api_pred"] == merged["pred"]).sum())
    return {
        "model_name": model_name,
        "n_posts": len(merged),
        "n_label_match": matches,
        "label_match_pct": 100 * matches / len(merged),
        "max_prob_diff": float((merged["api_stress_prob"] - merged["stress_prob"]).abs().max()),
    }


def latency_summary(server_ms: np.ndarray, client_ms: np.ndarray, model_name: str) -> dict:
    """Mean and 95th percentile of server-side and round-trip latency."""
    return {
        "model_name": model_name,
        "n_requests": len(server_ms),
        "server_mean_ms": float(np.mean(server_ms)),
        "server_p95_ms": float(np.percentile(server_ms, 95)),
        "client_mean_ms": float(np.mean(client_ms)),
        "client_p95_ms": float(np.percentile(client_ms, 95)),
        "goal_ms": 1000.0,
        "meets_goal": bool(np.percentile(client_ms, 95) < 1000),
    }


def sample_errors(
    test: pd.DataFrame, offline: pd.DataFrame, n: int = 50, seed: int = SEED
) -> pd.DataFrame:
    """Pick ``n`` random misclassified test posts, with an empty ``group`` column."""
    df = test.join(offline[["pred", "stress_prob"]], on="id")
    wrong = df[df["label"] != df["pred"]]
    sample = wrong.sample(n=min(n, len(wrong)), random_state=seed)
    confidence = np.where(sample["pred"] == 1, sample["stress_prob"], 1 - sample["stress_prob"])
    return pd.DataFrame({
        "id": sample["id"],
        "topic": sample["topic"],
        "true_label": sample["label"].map(LABEL_NAMES),
        "predicted_label": sample["pred"].map(LABEL_NAMES),
        "confidence": confidence.round(4),
        "error_type": np.where(sample["pred"] == 1, "false positive", "false negative"),
        "text": sample["text"],
        "group": pd.NA,
    }).reset_index(drop=True)
