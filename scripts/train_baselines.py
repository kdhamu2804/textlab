"""Phase 2: train the majority and TF-IDF baselines and save their predictions.

For every experiment (standard split + 5 new-topic splits) each baseline is
trained on training data only (TF-IDF tunes C on a validation split) and
predicts the test posts. The standard TF-IDF model is saved to models/tfidf/.

Run from the project root:
    python -m scripts.train_baselines
"""

from datetime import UTC, datetime

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.baselines import C_GRID, fit_majority, fit_tfidf
from ml.config import RESULTS_DIR, SEED, TFIDF_DIR
from ml.data import load_split
from ml.predictors import MODEL_FILE, write_metadata
from ml.reporting import prediction_frame, save_predictions
from ml.splits import STANDARD, experiments
from scripts.evaluate import main as evaluate_all


def save_tfidf_model(model: Pipeline, best_c: float, n_train: int) -> None:
    """Save the standard TF-IDF pipeline and its metadata."""
    TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TFIDF_DIR / MODEL_FILE)
    write_metadata(TFIDF_DIR, {
        "model_name": "tfidf",
        "description": "TF-IDF (word 1-2 grams) + logistic regression",
        "C": best_c,
        "c_grid": list(C_GRID),
        "n_train": n_train,
        "seed": SEED,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })


def main() -> None:
    """Run all baseline experiments, save predictions and evaluate."""
    train, test = load_split("train"), load_split("test")
    majority_frames, tfidf_frames, tuning = [], [], []

    for exp in experiments(train, test):
        majority = fit_majority(exp.train)
        majority_frames.append(prediction_frame(
            exp.setting, exp.test_set, exp.test, majority.predict_proba(exp.test["text"])[:, 1]))

        tfidf, best_c, table = fit_tfidf(exp.train)
        tfidf_frames.append(prediction_frame(
            exp.setting, exp.test_set, exp.test, tfidf.predict_proba(exp.test["text"])[:, 1]))
        tuning.append(table.assign(setting=exp.setting, test_set=exp.test_set,
                                   chosen=table["C"] == best_c))
        print(f"{exp.setting:>9} / {exp.test_set:<13} train={len(exp.train):4d} "
              f"test={len(exp.test):3d} best C={best_c}")

        if exp.setting == STANDARD:
            save_tfidf_model(tfidf, best_c, len(exp.train))

    save_predictions("majority", majority_frames)
    save_predictions("tfidf", tfidf_frames)
    pd.concat(tuning, ignore_index=True).to_csv(RESULTS_DIR / "tfidf_tuning.csv", index=False)
    evaluate_all()


if __name__ == "__main__":
    main()
