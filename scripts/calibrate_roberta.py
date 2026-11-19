"""Fit a temperature for the saved RoBERTa model on its validation split.

The validation split is the same 20% of the training posts that was used for
early stopping (seed 42), so no test post is used. Labels do not change; only
the probabilities shown to users become better calibrated.

Writes:
    models/roberta/metadata.json      adds "temperature"
    results/predictions/roberta.csv   standard-split stress_prob after scaling
    results/temperature.csv           T, and NLL / ECE before and after

Run from the project root, after scripts.train_roberta:
    python -m scripts.calibrate_roberta
    python -m scripts.evaluate && python -m scripts.analyze
"""

import numpy as np
import pandas as pd

from ml.analysis import calibration_summary
from ml.calibration import fit_temperature, nll, stress_probability
from ml.config import RESULTS_DIR, ROBERTA_DIR
from ml.data import load_split
from ml.predictors import RobertaPredictor, read_metadata, write_metadata
from ml.reporting import PREDICTION_COLUMNS, predictions_path
from ml.splits import STANDARD, validation_split


def main() -> None:
    """Fit T on validation logits and rescale the saved test probabilities."""
    train, test = load_split("train"), load_split("test")
    _, val = validation_split(train)
    model = RobertaPredictor(ROBERTA_DIR, temperature=1.0, device="auto")
    val_logits, test_logits = model.logits(val["text"].tolist()), model.logits(test["text"].tolist())

    temperature = fit_temperature(val_logits, val["label"].to_numpy())
    raw_test, scaled_test = stress_probability(test_logits), stress_probability(test_logits, temperature)
    report = {
        "temperature": temperature,
        "n_val": len(val),
        "val_nll_before": nll(val_logits, val["label"]),
        "val_nll_after": nll(val_logits, val["label"], temperature),
        "test_ece_before": calibration_summary(test["label"], raw_test)["ece"],
        "test_ece_after": calibration_summary(test["label"], scaled_test)["ece"],
    }
    pd.DataFrame([report]).to_csv(RESULTS_DIR / "temperature.csv", index=False)
    write_metadata(ROBERTA_DIR, {**read_metadata(ROBERTA_DIR), "temperature": temperature})

    path = predictions_path("roberta")
    preds = pd.read_csv(path, dtype={"id": str})
    standard = preds["setting"] == STANDARD
    new_prob = pd.Series(scaled_test, index=test["id"]).loc[preds.loc[standard, "id"]].to_numpy()
    if not np.array_equal((new_prob > 0.5).astype(int), preds.loc[standard, "pred"].to_numpy()):
        raise SystemExit("Scaled probabilities changed some labels; the saved predictions do not match the model.")
    preds.loc[standard, "stress_prob"] = new_prob
    preds[PREDICTION_COLUMNS].to_csv(path, index=False)

    print(pd.Series(report).round(4).to_string())
    print(f"\nSaved temperature to {ROBERTA_DIR / 'metadata.json'} and updated {path}")


if __name__ == "__main__":
    main()
