"""Save per-post predictions and turn them into a metrics table."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import PREDICTIONS_DIR
from ml.evaluation import evaluate
from ml.splits import STANDARD

MODEL_ORDER: list[str] = ["majority", "tfidf", "roberta"]
MODEL_LABELS: dict[str, str] = {
    "majority": "Majority class",
    "tfidf": "TF-IDF + LR",
    "roberta": "RoBERTa",
}
PREDICTION_COLUMNS: list[str] = [
    "setting", "test_set", "id", "topic", "label", "stress_prob", "pred",
]


def prediction_frame(
    setting: str, test_set: str, test: pd.DataFrame, stress_prob: np.ndarray
) -> pd.DataFrame:
    """Build the per-post prediction table for one experiment."""
    return pd.DataFrame(
        {
            "setting": setting,
            "test_set": test_set,
            "id": test["id"].to_numpy(),
            "topic": test["topic"].to_numpy(),
            "label": test["label"].to_numpy(),
            "stress_prob": stress_prob,
            "pred": (np.asarray(stress_prob) > 0.5).astype(int),
        }
    )


def predictions_path(model: str, directory: Path = PREDICTIONS_DIR) -> Path:
    """Return the CSV path that stores one model's predictions."""
    return directory / f"{model}.csv"


def save_predictions(model: str, frames: list[pd.DataFrame]) -> Path:
    """Concatenate and save all predictions for one model."""
    path = predictions_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS].to_csv(path, index=False)
    return path


def load_all_predictions(directory: Path = PREDICTIONS_DIR) -> dict[str, pd.DataFrame]:
    """Load every model's predictions that exist, in a fixed model order."""
    loaded = {}
    for model in MODEL_ORDER:
        path = predictions_path(model, directory)
        if path.exists():
            loaded[model] = pd.read_csv(path, dtype={"id": str})
    return loaded


def _metric_row(model: str, setting: str, test_set: str, df: pd.DataFrame) -> dict:
    """Evaluate one group of predictions and tag it with its identifiers."""
    return {
        "model": model,
        "setting": setting,
        "test_set": test_set,
        **evaluate(df["label"].to_numpy(), df["pred"].to_numpy()),
    }


def metrics_table(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute metrics with bootstrap CIs for every model and experiment.

    Besides the stored experiments, it adds ``in_topic`` rows: the standard
    model scored on each topic's test posts. They are the same posts as the
    matching ``new_topic`` rows, so the two can be compared directly.
    """
    rows = []
    for model, preds in predictions.items():
        for (setting, test_set), group in preds.groupby(["setting", "test_set"], sort=False):
            rows.append(_metric_row(model, setting, test_set, group))
        standard = preds[preds["setting"] == STANDARD]
        for topic, group in standard.groupby("topic"):
            rows.append(_metric_row(model, "in_topic", topic, group))
    table = pd.DataFrame(rows)
    table["n"] = table["n"].astype(int)
    return table
