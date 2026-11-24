"""Extension: score the emotion models on the GoEmotions test split.

Applies each model's validation-tuned thresholds to its saved test probabilities.

Writes:
    results/emotions_metrics.csv       macro / micro F1 with 95% bootstrap CIs per model
    results/emotions_per_label.csv     precision, recall, F1 per emotion and model
    results/emotions_significance.csv  paired bootstrap of macro F1, RoBERTa vs TF-IDF
    results/emotions_per_label.png

Run from the project root, after scripts.train_emotions:
    python -m scripts.evaluate_emotions
"""

import numpy as np
import pandas as pd

from ml.charts import plot_emotion_f1
from ml.config import PREDICTIONS_DIR, RESULTS_DIR
from ml.emotions import (
    EMOTION_ROBERTA_DIR,
    EMOTION_TFIDF_DIR,
    EMOTIONS,
    bootstrap_f1,
    load_emotions,
    paired_bootstrap_macro,
    per_label_report,
    read_emotion_metadata,
)

MODELS = {"tfidf": EMOTION_TFIDF_DIR, "roberta": EMOTION_ROBERTA_DIR}


def predictions(model: str, ids: pd.Series) -> np.ndarray | None:
    """Thresholded 0/1 test predictions of one model, aligned to ``ids`` (None if missing)."""
    path = PREDICTIONS_DIR / f"emotions_{model}.csv"
    if not path.exists():
        return None
    probs = pd.read_csv(path, dtype={"id": str}).set_index("id").loc[ids, EMOTIONS].to_numpy()
    thresholds = np.array(read_emotion_metadata(MODELS[model])["thresholds"])
    return (probs >= thresholds).astype(int)


def main() -> None:
    """Score every trained emotion model and compare them."""
    test, y = load_emotions("test")
    preds = {m: p for m in MODELS if (p := predictions(m, test["id"])) is not None}
    if not preds:
        raise SystemExit("No emotion predictions found. Run `python -m scripts.train_emotions` first.")

    metrics = pd.DataFrame([{"model": m, "n": len(y), **bootstrap_f1(y, p)} for m, p in preds.items()])
    per_label = pd.concat(
        [per_label_report(y, p, np.array(read_emotion_metadata(MODELS[m])["thresholds"])).assign(model=m)
         for m, p in preds.items()],
        ignore_index=True,
    )
    metrics.to_csv(RESULTS_DIR / "emotions_metrics.csv", index=False)
    per_label.to_csv(RESULTS_DIR / "emotions_per_label.csv", index=False)
    plot_emotion_f1(per_label, RESULTS_DIR / "emotions_per_label.png")

    if {"tfidf", "roberta"} <= preds.keys():
        sig = paired_bootstrap_macro(y, preds["tfidf"], preds["roberta"])
        pd.DataFrame([{"model_a": "tfidf", "model_b": "roberta", "metric": "macro_f1", **sig}]).to_csv(
            RESULTS_DIR / "emotions_significance.csv", index=False)
        print(f"RoBERTa vs TF-IDF macro F1: {sig['difference']:+.3f} "
              f"[{sig['difference_low']:+.3f}, {sig['difference_high']:+.3f}], p = {sig['p_value']:.4f}")

    print(metrics.round(3).to_string(index=False))
    best = per_label[per_label["model"] == list(preds)[-1]].sort_values("f1", ascending=False)
    print("\nBest emotions:", ", ".join(f"{r.emotion} {r.f1:.2f}" for r in best.head(5).itertuples()))
    print("Hardest emotions:", ", ".join(f"{r.emotion} {r.f1:.2f}" for r in best.tail(5).itertuples()))


if __name__ == "__main__":
    main()
