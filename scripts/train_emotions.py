"""Extension: train the emotion models on GoEmotions.

For each model: fit on the training split, choose per-emotion thresholds on the
validation split, and save test-set probabilities. The test split is never used
for training or for choosing anything.

Writes:
    models/emotions_tfidf/   TF-IDF + logistic regression (model.joblib, metadata.json)
    models/emotions/         fine-tuned RoBERTa (save_pretrained, metadata.json)
    results/predictions/emotions_{tfidf,roberta}.csv   test probabilities

Run from the project root:
    python -m scripts.train_emotions --model tfidf
    python -m scripts.train_emotions --model roberta     # needs a GPU, ~30 min
    python -m scripts.evaluate_emotions
"""

import argparse
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.config import PREDICTIONS_DIR, SEED
from ml.emotions import (
    C_GRID,
    EMOTION_ROBERTA_DIR,
    EMOTION_TFIDF_DIR,
    EMOTIONS,
    build_tfidf_emotions,
    load_emotions,
    macro_micro_f1,
    save_thresholds,
    tune_thresholds,
)

BASE_MODEL = "roberta-base"
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
MAX_EPOCHS = 4


def save_test_probabilities(name: str, ids: pd.Series, probs: np.ndarray) -> Path:
    """Save test probabilities, one column per emotion."""
    path = PREDICTIONS_DIR / f"emotions_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(np.round(probs, 4), columns=EMOTIONS)
    out.insert(0, "id", ids.to_numpy())
    out.to_csv(path, index=False)
    return path


def train_tfidf() -> None:
    """Choose C by validation macro F1 (with tuned thresholds), refit on train, save."""
    train, y_train = load_emotions("train")
    val, y_val = load_emotions("validation")
    test, _ = load_emotions("test")
    best = None
    for c in C_GRID:
        model = build_tfidf_emotions(c).fit(train["text"], y_train)
        p_val = model.predict_proba(val["text"])
        thresholds = tune_thresholds(y_val, p_val)
        macro, micro = macro_micro_f1(y_val, (p_val >= thresholds).astype(int))
        print(f"C={c:<5} validation macro F1 {macro:.3f} · micro F1 {micro:.3f}")
        if best is None or macro > best[0]:
            best = (macro, c, model, thresholds)
    _, c, model, thresholds = best
    if EMOTION_TFIDF_DIR.exists():
        shutil.rmtree(EMOTION_TFIDF_DIR)
    EMOTION_TFIDF_DIR.mkdir(parents=True)
    joblib.dump(model, EMOTION_TFIDF_DIR / "model.joblib")
    save_thresholds(EMOTION_TFIDF_DIR, thresholds, {"model_name": "emotions_tfidf", "C": c, "seed": SEED})
    path = save_test_probabilities("tfidf", test["id"], model.predict_proba(test["text"]))
    print(f"Best C={c}. Saved {EMOTION_TFIDF_DIR} and {path}")


def train_roberta(per_device_batch: int) -> None:
    """Fine-tune roberta-base with one sigmoid output per emotion; keep the epoch with lowest validation loss."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from ml.emotions import RobertaEmotionPredictor

    print("Training on:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow!)")
    set_seed(SEED)
    train, y_train = load_emotions("train")
    val, y_val = load_emotions("validation")
    test, _ = load_emotions("test")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    class Comments(torch.utils.data.Dataset):
        def __init__(self, texts: pd.Series, y: np.ndarray) -> None:
            self.enc = tokenizer(texts.tolist(), truncation=True, max_length=MAX_LENGTH)
            self.y = y.astype(np.float32)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i: int) -> dict:
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(EMOTIONS), problem_type="multi_label_classification",
        id2label=dict(enumerate(EMOTIONS)), label2id={e: i for i, e in enumerate(EMOTIONS)},
    )
    steps_per_epoch = int(np.ceil(len(train) / BATCH_SIZE))
    with tempfile.TemporaryDirectory(prefix="textlab_emotions_") as tmp:
        args = TrainingArguments(
            output_dir=tmp, learning_rate=LEARNING_RATE, num_train_epochs=MAX_EPOCHS,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=max(1, BATCH_SIZE // per_device_batch),
            per_device_eval_batch_size=64, weight_decay=0.01,
            warmup_steps=int(0.1 * MAX_EPOCHS * steps_per_epoch),
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=1,
            fp16=torch.cuda.is_available(), seed=SEED, data_seed=SEED, logging_strategy="epoch",
            report_to="none", dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=Comments(train["text"], y_train),
            eval_dataset=Comments(val["text"], y_val), data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )
        trainer.train()
        history = [log for log in trainer.state.log_history if "eval_loss" in log]
        best = min(history, key=lambda log: log["eval_loss"])
        if EMOTION_ROBERTA_DIR.exists():
            shutil.rmtree(EMOTION_ROBERTA_DIR)
        trainer.save_model(str(EMOTION_ROBERTA_DIR))
        tokenizer.save_pretrained(str(EMOTION_ROBERTA_DIR))

    # Thresholds need the saved model; write placeholder metadata first so the predictor can load.
    save_thresholds(EMOTION_ROBERTA_DIR, np.full(len(EMOTIONS), 0.5))
    predictor = RobertaEmotionPredictor(EMOTION_ROBERTA_DIR, batch_size=64, device="auto")
    thresholds = tune_thresholds(y_val, predictor.probabilities(val["text"].tolist()))
    save_thresholds(EMOTION_ROBERTA_DIR, thresholds, {
        "model_name": "emotions_roberta", "base_model": BASE_MODEL, "seed": SEED,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "best_epoch": best["epoch"],
        "val_loss": best["eval_loss"], "epochs_run": history[-1]["epoch"],
    })
    path = save_test_probabilities("roberta", test["id"], predictor.probabilities(test["text"].tolist()))
    print(f"Best epoch {best['epoch']:.0f} (validation loss {best['eval_loss']:.4f}). "
          f"Saved {EMOTION_ROBERTA_DIR} and {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["tfidf", "roberta", "all"], default="all")
    parser.add_argument("--per-device-batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    args = parser.parse_args()
    if args.model in ("tfidf", "all"):
        train_tfidf()
    if args.model in ("roberta", "all"):
        train_roberta(args.per_device_batch)


if __name__ == "__main__":
    main()
