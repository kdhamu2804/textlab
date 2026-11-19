"""Confusion matrices, significance tests and calibration for every saved model.

Reads results/predictions/*.csv (standard split only) and writes:
    results/confusion.csv            TN / FP / FN / TP per model
    results/significance.csv         paired bootstrap on F1 + McNemar, model pairs
    results/calibration.csv          reliability-diagram bins per model
    results/calibration_summary.csv  ECE and Brier score per model
    results/confusion_matrices.png, results/calibration.png

Run from the project root, after scripts.evaluate:
    python -m scripts.analyze
"""

import pandas as pd

from ml.analysis import calibration_bins, calibration_summary, compare_models, confusion
from ml.charts import plot_calibration, plot_confusion
from ml.config import RESULTS_DIR
from ml.reporting import load_all_predictions
from ml.splits import STANDARD

# Each pair is (baseline, challenger): is the challenger better?
COMPARISONS: list[tuple[str, str]] = [("majority", "tfidf"), ("tfidf", "roberta")]


def main() -> None:
    """Run all analyses on the standard-split predictions and save CSVs and charts."""
    standard = {
        model: preds[preds["setting"] == STANDARD].sort_values("id").reset_index(drop=True)
        for model, preds in load_all_predictions().items()
    }
    if not standard:
        raise SystemExit("No predictions found. Run `python -m scripts.train_baselines` first.")

    conf = pd.DataFrame([{"model": m, **confusion(p["label"], p["pred"])} for m, p in standard.items()])
    sig = pd.DataFrame([
        compare_models(standard[a]["label"], standard[a]["pred"], standard[b]["pred"], a, b)
        for a, b in COMPARISONS
        if a in standard and b in standard
    ])
    probabilistic = {m: p for m, p in standard.items() if m != "majority"}
    bins = pd.concat(
        [calibration_bins(p["label"], p["stress_prob"]).assign(model=m) for m, p in probabilistic.items()],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        [{"model": m, **calibration_summary(p["label"], p["stress_prob"])} for m, p in probabilistic.items()]
    )

    for name, table in [("confusion", conf), ("significance", sig),
                        ("calibration", bins), ("calibration_summary", summary)]:
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    plot_confusion(conf, RESULTS_DIR / "confusion_matrices.png")
    plot_calibration(bins, summary, RESULTS_DIR / "calibration.png")

    print("Confusion matrices:\n", conf.to_string(index=False))
    print("\nSignificance (b vs a, F1):\n", sig.round(4).to_string(index=False))
    print("\nCalibration:\n", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
