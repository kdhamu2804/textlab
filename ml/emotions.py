"""Extension: multi-label emotion detection on GoEmotions (27 emotions + neutral).

The stress study (Dreaddit) is unchanged; this module adds a second, separate
task. One comment can carry several emotions, so every emotion is a yes/no
decision with its own probability threshold, chosen on the validation split.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from ml.config import MODELS_DIR, RAW_DIR, SEED

EMOTION_DATASET: str = "google-research-datasets/go_emotions"
EMOTION_CONFIG: str = "simplified"
EMOTION_SPLITS: tuple[str, ...] = ("train", "validation", "test")
EMOTIONS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
EMOTION_ROBERTA_DIR: Path = MODELS_DIR / "emotions"
EMOTION_TFIDF_DIR: Path = MODELS_DIR / "emotions_tfidf"
THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.05, 0.951, 0.05), 2)
C_GRID: tuple[float, ...] = (1.0, 3.0, 10.0)


# ---------------------------------------------------------------- data

def emotion_raw_path(split: str) -> Path:
    """CSV path of one GoEmotions split."""
    return RAW_DIR / f"go_emotions_{split}.csv"


def download_emotions(overwrite: bool = False) -> None:
    """Download GoEmotions (simplified) and save each split as CSV (labels space-separated)."""
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(EMOTION_DATASET, EMOTION_CONFIG)
    names = dataset["train"].features["labels"].feature.names
    if names != EMOTIONS:
        raise ValueError("GoEmotions label order differs from EMOTIONS")
    for split in EMOTION_SPLITS:
        path = emotion_raw_path(split)
        if path.exists() and not overwrite:
            continue
        df = dataset[split].to_pandas()
        df["labels"] = df["labels"].map(lambda ids: " ".join(str(i) for i in ids))
        df[["id", "text", "labels"]].to_csv(path, index=False)


def multi_hot(label_lists: Sequence[Sequence[int]], n_labels: int = len(EMOTIONS)) -> np.ndarray:
    """Turn lists of label ids into a 0/1 matrix (rows = texts, columns = emotions)."""
    y = np.zeros((len(label_lists), n_labels), dtype=int)
    for row, ids in enumerate(label_lists):
        y[row, list(ids)] = 1
    return y


def load_emotions(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one split as (dataframe with id/text, multi-hot label matrix)."""
    if split not in EMOTION_SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    df = pd.read_csv(emotion_raw_path(split), dtype={"id": str, "labels": str})
    ids = [[int(i) for i in s.split()] for s in df["labels"]]
    return df[["id", "text"]], multi_hot(ids)


# ---------------------------------------------------------------- metrics

def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def _counts(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, false positives and false negatives per emotion (last axis)."""
    tp = ((y == 1) & (pred == 1)).sum(axis=-2)
    fp = ((y == 0) & (pred == 1)).sum(axis=-2)
    fn = ((y == 1) & (pred == 0)).sum(axis=-2)
    return tp, fp, fn


def macro_micro_f1(y: np.ndarray, pred: np.ndarray) -> tuple[Any, Any]:
    """Macro F1 (mean over emotions) and micro F1 (pooled over all decisions).

    Works on (n, k) arrays and on stacks of resamples (r, n, k).
    """
    tp, fp, fn = _counts(y, pred)
    macro = _f1(tp, fp, fn).mean(axis=-1)
    micro = _f1(tp.sum(axis=-1), fp.sum(axis=-1), fn.sum(axis=-1))
    return macro, micro


def per_label_report(y: np.ndarray, pred: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Precision, recall, F1 and support for every emotion."""
    tp, fp, fn = _counts(y, pred)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(len(tp)), where=(tp + fn) > 0)
    return pd.DataFrame({
        "emotion": EMOTIONS, "support": y.sum(axis=0), "threshold": thresholds,
        "precision": precision, "recall": recall, "f1": _f1(tp, fp, fn),
    })


def tune_thresholds(y_val: np.ndarray, p_val: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """Per-emotion probability threshold that maximizes that emotion's F1 on validation data."""
    best = np.full(y_val.shape[1], 0.5)
    for k in range(y_val.shape[1]):
        if y_val[:, k].sum() == 0:
            continue
        scores = []
        for t in grid:
            pred = p_val[:, k] >= t
            tp = np.sum(pred & (y_val[:, k] == 1))
            fp = np.sum(pred & (y_val[:, k] == 0))
            fn = np.sum(~pred & (y_val[:, k] == 1))
            scores.append(_f1(np.array(tp), np.array(fp), np.array(fn)))
        best[k] = grid[int(np.argmax(scores))]
    return best


def bootstrap_f1(
    y: np.ndarray, pred: np.ndarray, n_resamples: int = 1000, seed: int = SEED
) -> dict[str, float]:
    """Macro and micro F1 with 95% percentile bootstrap intervals (resampling comments)."""
    rng = np.random.default_rng(seed)
    macro, micro = macro_micro_f1(y, pred)
    macros, micros = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        m, u = macro_micro_f1(y[idx], pred[idx])
        macros.append(m)
        micros.append(u)
    return {
        "macro_f1": float(macro), "macro_f1_low": float(np.percentile(macros, 2.5)),
        "macro_f1_high": float(np.percentile(macros, 97.5)),
        "micro_f1": float(micro), "micro_f1_low": float(np.percentile(micros, 2.5)),
        "micro_f1_high": float(np.percentile(micros, 97.5)),
    }


def paired_bootstrap_macro(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_resamples: int = 2000, seed: int = SEED
) -> dict[str, float]:
    """Paired bootstrap of macro F1(b) - macro F1(a); one-sided p = share of resamples where b is not better."""
    rng = np.random.default_rng(seed)
    observed = macro_micro_f1(y, pred_b)[0] - macro_micro_f1(y, pred_a)[0]
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        diffs[r] = macro_micro_f1(y[idx], pred_b[idx])[0] - macro_micro_f1(y[idx], pred_a[idx])[0]
    return {
        "difference": float(observed),
        "difference_low": float(np.percentile(diffs, 2.5)),
        "difference_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(max(np.mean(diffs <= 0), 1 / n_resamples)),
        "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------- TF-IDF baseline

def build_tfidf_emotions(c: float = 3.0) -> Pipeline:
    """TF-IDF (word 1-2 grams) + one logistic regression per emotion."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=SEED), n_jobs=-1)),
    ])


def save_thresholds(model_dir: Path, thresholds: np.ndarray, extra: dict[str, Any] | None = None) -> None:
    """Store per-emotion thresholds (and any extra metadata) in ``metadata.json``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {"emotions": EMOTIONS, "thresholds": [float(t) for t in thresholds], **(extra or {})}
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_emotion_metadata(model_dir: Path) -> dict[str, Any]:
    """Read ``metadata.json`` of an emotion model."""
    return json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- predictors

class EmotionPredictor:
    """Base class: subclasses return an (n, 28) matrix of emotion probabilities."""

    name: str
    display_name: str

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.metadata = read_emotion_metadata(model_dir)
        self.thresholds = np.array(self.metadata["thresholds"])

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, text: str) -> list[dict[str, Any]]:
        """All emotions for one text, most likely first, with whether each passes its threshold."""
        probs = self.probabilities([text])[0]
        rows = [
            {"emotion": e, "probability": round(float(p), 4), "threshold": float(t), "detected": bool(p >= t)}
            for e, p, t in zip(EMOTIONS, probs, self.thresholds, strict=True)
        ]
        return sorted(rows, key=lambda r: r["probability"], reverse=True)


class TfidfEmotionPredictor(EmotionPredictor):
    """TF-IDF + one-vs-rest logistic regression emotion model."""

    name = "emotions_tfidf"
    display_name = "TF-IDF + Logistic Regression (GoEmotions)"

    def __init__(self, model_dir: Path = EMOTION_TFIDF_DIR) -> None:
        super().__init__(model_dir)
        self.pipeline = joblib.load(model_dir / "model.joblib")

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))


class RobertaEmotionPredictor(EmotionPredictor):
    """RoBERTa fine-tuned for multi-label emotions (sigmoid output per emotion)."""

    name = "emotions_roberta"
    display_name = "RoBERTa (fine-tuned on GoEmotions)"

    def __init__(self, model_dir: Path = EMOTION_ROBERTA_DIR, batch_size: int = 32, max_length: int = 128,
                 device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        super().__init__(model_dir)
        self._torch = torch
        self.batch_size, self.max_length = batch_size, max_length
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        texts, rows = list(texts), []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[start : start + self.batch_size], truncation=True,
                                       max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                rows.append(self._torch.sigmoid(self.model(**batch).logits.float()).cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, len(EMOTIONS)))


def load_emotion_predictor(
    roberta_dir: Path = EMOTION_ROBERTA_DIR, tfidf_dir: Path = EMOTION_TFIDF_DIR
) -> EmotionPredictor | None:
    """Best available emotion model (RoBERTa, then TF-IDF), or None if neither was trained."""
    if (roberta_dir / "config.json").exists():
        try:
            return RobertaEmotionPredictor(roberta_dir)
        except ImportError:
            pass
    if (tfidf_dir / "model.joblib").exists():
        return TfidfEmotionPredictor(tfidf_dir)
    return None
