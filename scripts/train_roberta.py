"""Fine-tune roberta-base on a local GPU (same recipe as notebooks/train_roberta_colab.ipynb).

Trains the standard model and, unless ``--standard-only`` is given, one model per
held-out topic. Each model picks its best epoch by stress-class F1 on a validation
split of its own training data; test posts are only used for the final predictions.

Writes:
    models/roberta/                      the standard model (used by the API)
    results/predictions/roberta.csv      per-post test predictions
    results/roberta_training.csv         best epoch and validation F1 per experiment

Run from the project root (needs a CUDA build of PyTorch for reasonable speed):
    python -m scripts.train_roberta
    python -m scripts.evaluate
"""

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ml.config import RESULTS_DIR, ROBERTA_DIR, SEED
from ml.data import load_split
from ml.reporting import prediction_frame, save_predictions
from ml.splits import STANDARD, experiments, validation_split

BASE_MODEL = "roberta-base"
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
BATCH_SIZE = 16  # effective batch size, reached with gradient accumulation on small GPUs
MAX_EPOCHS = 6
PATIENCE = 2
TRAINING_FILE = RESULTS_DIR / "roberta_training.csv"


class PostDataset(torch.utils.data.Dataset):
    """Tokenized posts and labels for the Trainer."""

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> None:
        self.encodings = tokenizer(df["text"].tolist(), truncation=True, max_length=MAX_LENGTH)
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def _val_f1(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    return {"f1": float(f1_score(labels, logits.argmax(-1), pos_label=1))}


def train_one(
    train_part: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, run_dir: Path, per_device_batch: int
) -> tuple[Trainer, dict[str, Any]]:
    """Fine-tune on ``train_part`` and keep the epoch with the best validation F1."""
    set_seed(SEED)
    fit_part, val_part = validation_split(train_part)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    accumulation = max(1, BATCH_SIZE // per_device_batch)
    steps_per_epoch = int(np.ceil(len(fit_part) / BATCH_SIZE))
    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=LEARNING_RATE,
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=accumulation,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        warmup_steps=int(0.1 * MAX_EPOCHS * steps_per_epoch),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        logging_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=PostDataset(fit_part, tokenizer),
        eval_dataset=PostDataset(val_part, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_val_f1,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer.train()
    history = [log for log in trainer.state.log_history if "eval_f1" in log]
    best = max(history, key=lambda log: log["eval_f1"])
    info = {
        "best_epoch": best["epoch"],
        "val_f1": best["eval_f1"],
        "epochs_run": history[-1]["epoch"],
        "n_fit": len(fit_part),
        "n_val": len(val_part),
    }
    return trainer, info


def stress_probabilities(trainer: Trainer, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    """Return P(stress) for each post."""
    logits = trainer.predict(PostDataset(df, tokenizer)).predictions
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


def save_standard_model(trainer: Trainer, tokenizer: PreTrainedTokenizerBase, info: dict[str, Any]) -> None:
    """Save the standard model for the API, replacing any previous one."""
    if ROBERTA_DIR.exists():
        shutil.rmtree(ROBERTA_DIR)
    trainer.save_model(str(ROBERTA_DIR))
    tokenizer.save_pretrained(str(ROBERTA_DIR))
    metadata = {
        "model_name": "roberta",
        "base_model": BASE_MODEL,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        **info,
    }
    (ROBERTA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standard-only", action="store_true", help="skip the 5 new-topic models")
    parser.add_argument("--per-device-batch", type=int, default=8, help="lower it if the GPU runs out of memory")
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow!)"
    print(f"Training on: {device}")
    train, test = load_split("train"), load_split("test")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    frames, rows = [], []
    with tempfile.TemporaryDirectory(prefix="textlab_roberta_") as tmp:
        for exp in experiments(train, test):
            if args.standard_only and exp.setting != STANDARD:
                continue
            print(f"\n=== {exp.setting} / {exp.test_set}: train {len(exp.train)}, test {len(exp.test)} ===")
            trainer, info = train_one(
                exp.train, tokenizer, Path(tmp) / f"{exp.setting}_{exp.test_set}", args.per_device_batch
            )
            probs = stress_probabilities(trainer, exp.test, tokenizer)
            frame = prediction_frame(exp.setting, exp.test_set, exp.test, probs)
            frames.append(frame)
            test_f1 = f1_score(frame["label"], frame["pred"])
            rows.append({"setting": exp.setting, "test_set": exp.test_set, **info, "test_f1": test_f1})
            print(f"best epoch {info['best_epoch']:.0f} | val F1 {info['val_f1']:.3f} | test F1 {test_f1:.3f}")
            if exp.setting == STANDARD:
                save_standard_model(trainer, tokenizer, info)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    path = save_predictions("roberta", frames)
    pd.DataFrame(rows).to_csv(TRAINING_FILE, index=False)
    print(f"\nSaved {ROBERTA_DIR}, {path} and {TRAINING_FILE}")
    print("Next: python -m scripts.evaluate")


if __name__ == "__main__":
    main()
