"""Manual label check: sample posts for the team and measure agreement."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ml.config import DATA_DIR, SEED

LABEL_CHECK_FILE: Path = DATA_DIR / "label_check.csv"
ANNOTATORS: list[str] = ["dhamodhar", "kushvanth", "uday"]


def sample_for_labeling(train: pd.DataFrame, n: int = 50, seed: int = SEED) -> pd.DataFrame:
    """Pick ``n`` random training posts with empty columns for each annotator."""
    sample = train.sample(n=n, random_state=seed)
    out = pd.DataFrame({
        "id": sample["id"].to_numpy(),
        "text": sample["text"].to_numpy(),
        "dataset_label": sample["label"].to_numpy(),
    })
    for name in ANNOTATORS:
        out[name] = pd.NA
    return out


def _clean_labels(column: pd.Series) -> pd.Series:
    """Parse an annotator column into 0/1 values (blank cells become NaN)."""
    values = pd.to_numeric(column, errors="coerce")
    bad = values.notna() & ~values.isin([0, 1])
    if bad.any():
        raise ValueError(f"Column {column.name!r} must contain only 0, 1 or blank")
    return values


def _agreement(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Percent agreement and Cohen's kappa on the rows both have labelled."""
    both = a.notna() & b.notna()
    n = int(both.sum())
    if n == 0:
        return {"n": 0, "agreement_pct": float("nan"), "cohen_kappa": float("nan")}
    x, y = a[both].astype(int), b[both].astype(int)
    kappa = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else float("nan")
    return {"n": n, "agreement_pct": 100 * float((x == y).mean()), "cohen_kappa": float(kappa)}


def agreement_report(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement of each annotator with the dataset label and with each other."""
    labels = {name: _clean_labels(df[name]) for name in ANNOTATORS}
    rows = [
        {"rater_a": name, "rater_b": "dataset_label", **_agreement(labels[name], df["dataset_label"])}
        for name in ANNOTATORS
    ]
    for i, a in enumerate(ANNOTATORS):
        for b in ANNOTATORS[i + 1:]:
            rows.append({"rater_a": a, "rater_b": b, **_agreement(labels[a], labels[b])})
    return pd.DataFrame(rows)
