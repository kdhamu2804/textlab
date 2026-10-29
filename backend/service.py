"""Model access and results loading used by the API routes."""

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ml.predictors import Prediction, Predictor, RobertaPredictor

METRICS_FILE = "metrics.csv"
API_CHECK_FILE = "api_check.csv"
LATENCY_FILE = "latency.csv"
ERRORS_FILE = "errors.csv"


def timed_predict(predictor: Predictor, texts: Sequence[str]) -> tuple[list[Prediction], float]:
    """Run the model and return predictions with the elapsed time in milliseconds."""
    start = time.perf_counter()
    predictions = predictor.predict(texts)
    return predictions, (time.perf_counter() - start) * 1000


def _records(path: Path) -> list[dict[str, Any]] | None:
    """Read a CSV as JSON-safe records (NaN becomes None); None if the file is missing."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _first(path: Path) -> dict[str, Any] | None:
    """Return the first record of a one-row summary CSV."""
    records = _records(path)
    return records[0] if records else None


def load_metrics(results_dir: Path) -> dict[str, Any]:
    """Collect all saved evaluation results for the frontend."""
    return {
        "metrics": _records(results_dir / METRICS_FILE) or [],
        "api_check": _first(results_dir / API_CHECK_FILE),
        "latency": _first(results_dir / LATENCY_FILE),
        "errors": _records(results_dir / ERRORS_FILE),
    }


def model_info(predictor: Predictor, results_dir: Path) -> dict[str, Any]:
    """Describe the loaded model, including its test F1 if it has been evaluated."""
    test_f1, test_f1_ci = None, None
    for row in _records(results_dir / METRICS_FILE) or []:
        if (row["model"], row["setting"], row["test_set"]) == (predictor.name, "standard", "all"):
            test_f1, test_f1_ci = row["f1"], (row["f1_low"], row["f1_high"])
    return {
        "model_name": predictor.name,
        "display_name": predictor.display_name,
        "is_fallback": not isinstance(predictor, RobertaPredictor),
        "test_f1": test_f1,
        "test_f1_ci": test_f1_ci,
        "trained_at": predictor.metadata.get("trained_at"),
    }
