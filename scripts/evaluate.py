"""Score every saved prediction file and draw the charts.

Reads results/predictions/{majority,tfidf,roberta}.csv (whichever exist),
writes results/metrics.csv and the PNG charts, and prints a summary.

Run from the project root:
    python -m scripts.evaluate
"""

from ml.charts import plot_model_comparison, plot_new_topic
from ml.config import RESULTS_DIR
from ml.reporting import load_all_predictions, metrics_table

METRICS_FILE = RESULTS_DIR / "metrics.csv"


def main() -> None:
    """Compute metrics with bootstrap CIs, save them and plot them."""
    predictions = load_all_predictions()
    if not predictions:
        raise SystemExit("No predictions found. Run `python -m scripts.train_baselines` first.")
    table = metrics_table(predictions)
    table.to_csv(METRICS_FILE, index=False)

    plot_model_comparison(table, RESULTS_DIR / "model_comparison.png")
    for model in predictions:
        if model != "majority":
            plot_new_topic(table, model, RESULTS_DIR / f"new_topic_{model}.png")

    cols = ["model", "setting", "test_set", "n", "f1", "f1_low", "f1_high",
            "precision", "recall", "accuracy"]
    print(f"\nSaved {METRICS_FILE}")
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
