"""Load saved models and turn texts into stress predictions and explanations.

The same predictor classes are used by the offline scripts and the API, so
both produce identical outputs for the same model.
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.calibration import stress_probability
from ml.config import ROBERTA_DIR, ROBERTA_ONNX_DIR, TFIDF_DIR

LABEL_NAMES: dict[int, str] = {0: "no stress", 1: "stress"}
MODEL_FILE: str = "model.joblib"
METADATA_FILE: str = "metadata.json"
# Words for explanations: letters/digits, optionally with an apostrophe part (can't, I'm).
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")
# Sentences: text up to and including . ! ? (or a line break), or the rest of the post.
SENTENCE_PATTERN = re.compile(r"[^.!?\n]*[^.!?\n\s][^.!?\n]*(?:[.!?]+|$)")
# Clauses, used when a post has fewer than MIN_SEGMENTS sentences.
CLAUSE_PATTERN = re.compile(r"[^.!?,;:\n]*[^.!?,;:\n\s][^.!?,;:\n]*(?:[.!?,;:]+|$)")
MIN_SEGMENTS = 3


@dataclass(frozen=True)
class Prediction:
    """A single prediction: the label and the model's confidence in it."""

    label: str
    label_id: int
    confidence: float
    stress_probability: float


@dataclass(frozen=True)
class Segment:
    """How much one word or sentence pushes the prediction toward stress (+) or no stress (-).

    ``score`` is scaled so the strongest segment in the post has |score| = 1.
    ``delta`` is the drop in P(stress) when the segment is removed (occlusion
    only). ``start``/``end`` are character offsets in the original text.
    """

    text: str
    start: int
    end: int
    score: float
    delta: float | None = None


@dataclass(frozen=True)
class Explanation:
    """Segments of a post with their influence on the prediction."""

    unit: str  # "word" or "sentence"
    method: str
    segments: list[Segment]


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


Span = tuple[str, int, int]


def _spans(pattern: re.Pattern[str], text: str) -> list[Span]:
    """Matches of ``pattern`` as (text, start, end), with surrounding spaces trimmed."""
    spans = []
    for m in pattern.finditer(text):
        stripped = m.group().strip()
        if stripped:
            start = m.start() + m.group().index(stripped)
            spans.append((stripped, start, start + len(stripped)))
    return spans


def word_spans(text: str) -> list[Span]:
    """Split a post into words with their character offsets."""
    return _spans(WORD_PATTERN, text)


def sentence_spans(text: str) -> list[Span]:
    """Split a post into sentences, or into clauses if it has fewer than three sentences."""
    sentences = _spans(SENTENCE_PATTERN, text)
    return sentences if len(sentences) >= MIN_SEGMENTS else _spans(CLAUSE_PATTERN, text)


def _normalized(spans: list[Span], raw: np.ndarray) -> list[Segment]:
    """Scale raw attributions to [-1, 1] by the largest absolute value."""
    peak = float(np.max(np.abs(raw))) if len(raw) else 0.0
    scale = peak if peak > 0 else 1.0
    return [Segment(t, s, e, round(float(r) / scale, 4)) for (t, s, e), r in zip(spans, raw, strict=True)]


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

    def explain(self, text: str) -> Explanation:
        """Sentence occlusion: how much P(stress) drops when each sentence is removed.

        Model-agnostic and faithful to the model's actual output; subclasses may
        override it with a cheaper exact method.
        """
        spans = sentence_spans(text)
        method = "Sentence occlusion: change in P(stress) when the sentence is removed"
        if len(spans) < 2:
            return Explanation("sentence", method, _normalized(spans, np.zeros(len(spans))))
        variants = [text[:s] + text[e:] for _, s, e in spans]
        probs = self.stress_probabilities([text, *variants])
        deltas = probs[0] - probs[1:]
        segments = [
            Segment(seg.text, seg.start, seg.end, seg.score, round(float(d), 4))
            for seg, d in zip(_normalized(spans, deltas), deltas, strict=True)
        ]
        return Explanation("sentence", method, segments)


class TfidfPredictor(Predictor):
    """TF-IDF + logistic regression model saved with joblib."""

    name = "tfidf"
    display_name = "TF-IDF + Logistic Regression"

    def __init__(self, model_dir: Path = TFIDF_DIR) -> None:
        super().__init__(model_dir)
        self.pipeline = joblib.load(model_dir / MODEL_FILE)

    def stress_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def explain(self, text: str) -> Explanation:
        """Exact contribution of each word to the logit (n-grams split equally over their words)."""
        method = "Linear contribution: TF-IDF weight x logistic-regression coefficient"
        spans = word_spans(text)
        if not spans:
            return Explanation("word", method, [])
        vectorizer, clf = self.pipeline.named_steps["tfidf"], self.pipeline.named_steps["clf"]
        row = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        per_word: dict[str, float] = {}
        for index, weight in zip(row.indices, row.data, strict=True):
            tokens = names[index].split()
            for token in tokens:
                per_word[token] = per_word.get(token, 0.0) + weight * clf.coef_[0, index] / len(tokens)
        counts: dict[str, int] = {}
        for word, _, _ in spans:
            counts[word.lower()] = counts.get(word.lower(), 0) + 1
        raw = np.array([per_word.get(w.lower(), 0.0) / counts[w.lower()] for w, _, _ in spans])
        return Explanation("word", method, _normalized(spans, raw))


class RobertaPredictor(Predictor):
    """Fine-tuned RoBERTa model saved with ``save_pretrained``.

    Probabilities use the temperature stored in ``metadata.json`` (1.0 if the
    model has not been calibrated), which never changes the predicted label.
    """

    name = "roberta"
    display_name = "RoBERTa (fine-tuned)"

    def __init__(
        self,
        model_dir: Path = ROBERTA_DIR,
        batch_size: int = 16,
        max_length: int = 256,
        temperature: float | None = None,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        super().__init__(model_dir)
        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.temperature = float(temperature if temperature is not None else self.metadata.get("temperature", 1.0))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    def _encode(self, texts: list[str], **kwargs: Any) -> Any:
        return self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                              return_tensors="pt", **kwargs)

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Raw two-class logits, one row per text."""
        texts = list(texts)
        rows: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self._encode(texts[start : start + self.batch_size]).to(self.device)
                rows.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, 2))

    def stress_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        return stress_probability(self.logits(texts), self.temperature) if len(texts) else np.empty(0)


class OnnxPredictor(Predictor):
    """The same fine-tuned RoBERTa, exported to 8-bit ONNX (``scripts.export_onnx``).

    Needs only ``onnxruntime`` and ``tokenizers`` (no PyTorch), so it fits on small
    servers. It reports itself as ``roberta``; ``results/onnx_check.csv`` records how
    closely it matches the full model.
    """

    name = "roberta"
    display_name = "RoBERTa (fine-tuned, 8-bit ONNX)"

    def __init__(
        self, model_dir: Path = ROBERTA_ONNX_DIR, batch_size: int = 16, max_length: int = 256
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        super().__init__(model_dir)
        self.batch_size = batch_size
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        # No memory arena: buffers are returned after each call, which keeps peak memory low.
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Raw two-class logits, one row per text."""
        texts = list(texts)
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encodings = self.tokenizer.encode_batch(texts[start : start + self.batch_size])
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            rows.append(self.session.run(["logits"], feed)[0])
        return np.concatenate(rows) if rows else np.empty((0, 2))

    def stress_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        return stress_probability(self.logits(texts), self.temperature) if len(texts) else np.empty(0)


def roberta_available(model_dir: Path = ROBERTA_DIR) -> bool:
    """Return True when a saved RoBERTa model exists and PyTorch is installed."""
    if not (model_dir / "config.json").exists():
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def load_predictor(
    roberta_dir: Path = ROBERTA_DIR, tfidf_dir: Path = TFIDF_DIR, onnx_dir: Path = ROBERTA_ONNX_DIR
) -> Predictor:
    """Load the best available model: full RoBERTa, then 8-bit ONNX RoBERTa, then TF-IDF."""
    if roberta_available(roberta_dir):
        return RobertaPredictor(roberta_dir)
    if (onnx_dir / "model.onnx").exists():
        return OnnxPredictor(onnx_dir)
    if (tfidf_dir / MODEL_FILE).exists():
        return TfidfPredictor(tfidf_dir)
    raise FileNotFoundError(
        f"No model found in {roberta_dir}, {onnx_dir} or {tfidf_dir}. "
        "Run `python -m scripts.train_baselines` first."
    )
