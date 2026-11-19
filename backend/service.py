"""Model access and results loading used by the API routes."""

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ml.annotation import agreement_report
from ml.predictors import Prediction, Predictor, RobertaPredictor

METRICS_FILE = "metrics.csv"
API_CHECK_FILE = "api_check.csv"
LATENCY_FILE = "latency.csv"
TEMPERATURE_FILE = "temperature.csv"
# Tables written by scripts.analyze and scripts.label_agreement, served as lists of rows.
TABLE_FILES: tuple[str, ...] = (
    "confusion", "significance", "calibration", "calibration_summary", "label_agreement",
)
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


def _errors_with_groups(results_dir: Path, groups_file: Path | None) -> list[dict[str, Any]] | None:
    """Misclassified posts, with the manual error group (and who reviewed it) joined by post id."""
    errors = _records(results_dir / ERRORS_FILE)
    if errors is None or groups_file is None or not groups_file.exists():
        return errors
    groups = pd.read_csv(groups_file, dtype=str).fillna("").set_index("id")
    for row in errors:
        key = str(row["id"])
        if key in groups.index:
            row["group"] = groups.at[key, "group"] or None
            row["reviewed_by"] = groups.at[key, "reviewed_by"] or None
    return errors


def _live_agreement(label_check_file: Path | None) -> list[dict[str, Any]] | None:
    """Agreement table computed from the label-check file, or None if nobody has labelled yet."""
    if label_check_file is None or not label_check_file.exists():
        return None
    report = agreement_report(pd.read_csv(label_check_file, dtype={"id": str}))
    if report["n"].eq(0).all():
        return None
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in report.to_dict(orient="records")
    ]


def load_metrics(
    results_dir: Path, groups_file: Path | None = None, label_check_file: Path | None = None
) -> dict[str, Any]:
    """Collect all saved evaluation results for the frontend (missing files become None).

    The label-check agreement is computed live from ``label_check_file`` when given,
    so the Results page updates as the team labels.
    """
    live = _live_agreement(label_check_file)
    return {
        "metrics": _records(results_dir / METRICS_FILE) or [],
        "api_check": _first(results_dir / API_CHECK_FILE),
        "latency": _first(results_dir / LATENCY_FILE),
        "errors": _errors_with_groups(results_dir, groups_file),
        **{name: _records(results_dir / f"{name}.csv") for name in TABLE_FILES},
        "temperature": _first(results_dir / TEMPERATURE_FILE),
        **({"label_agreement": live} if live else {}),
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
        "temperature": getattr(predictor, "temperature", None),
    }
