"""Load saved models and turn texts into stress predictions.

The same predictor classes are used by the offline scripts and the API, so
both produce identical outputs for the same model.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.config import ROBERTA_DIR, TFIDF_DIR

LABEL_NAMES: dict[int, str] = {0: "no stress", 1: "stress"}
MODEL_FILE: str = "model.joblib"
METADATA_FILE: str = "metadata.json"


@dataclass(frozen=True)
class Prediction:
    """A single prediction: the label and the model's confidence in it."""

    label: str
    label_id: int
    confidence: float
    stress_probability: float


def to_prediction(stress_probability: float) -> Prediction:
    """Convert a stress probability into a labelled prediction.

    A post is labelled stress only when p > 0.5, matching ``argmax``.
    """
    label_id = int(stress_probability > 0.5)
    confidence = stress_probability if label_id else 1.0 - stress_probability
    return Prediction(LABEL_NAMES[label_id], label_id, float(confidence), float(stress_probability))


def read_metadata(model_dir: Path) -> dict[str, Any]:
    """Read ``metadata.json`` from a model folder (empty dict if missing)."""
    path = model_dir / METADATA_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_metadata(model_dir: Path, metadata: dict[str, Any]) -> None:
    """Write ``metadata.json`` to a model folder."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class Predictor(ABC):
    """Base class: subclasses only need to return stress probabilities."""

    name: str
    display_name: str

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.metadata = read_metadata(model_dir)

    @abstractmethod
    def stress_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Return P(stress) for each text."""

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Return one prediction per text."""
        return [to_prediction(p) for p in self.stress_probabilities(texts)]


class TfidfPredictor(Predictor):
    """TF-IDF + logistic regression model saved with joblib."""

    name = "tfidf"
    display_name = "TF-IDF + Logistic Regression"

    def __init__(self, model_dir: Path = TFIDF_DIR) -> None:
        super().__init__(model_dir)
        self.pipeline = joblib.load(model_dir / MODEL_FILE)

    def stress_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]


class RobertaPredictor(Predictor):
    """Fine-tuned RoBERTa model saved with ``save_pretrained``."""

    name = "roberta"
    display_name = "RoBERTa (fine-tuned)"

    def __init__(
        self, model_dir: Path = ROBERTA_DIR, batch_size: int = 16, max_length: int = 256
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        super().__init__(model_dir)
        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    def stress_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        probs: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start : start + self.batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(**batch).logits
                probs.append(self._torch.softmax(logits, dim=-1)[:, 1].numpy())
        return np.concatenate(probs) if probs else np.empty(0)


def roberta_available(model_dir: Path = ROBERTA_DIR) -> bool:
    """Return True when a saved RoBERTa model exists."""
    return (model_dir / "config.json").exists()


def load_predictor(
    roberta_dir: Path = ROBERTA_DIR, tfidf_dir: Path = TFIDF_DIR
) -> Predictor:
    """Load RoBERTa if it was saved, otherwise fall back to TF-IDF."""
    if roberta_available(roberta_dir):
        return RobertaPredictor(roberta_dir)
    if (tfidf_dir / MODEL_FILE).exists():
        return TfidfPredictor(tfidf_dir)
    raise FileNotFoundError(
        f"No model found in {roberta_dir} or {tfidf_dir}. "
        "Run `python -m scripts.train_baselines` first."
    )
