"""Export the fine-tuned RoBERTa to a small 8-bit ONNX model for low-memory hosting.

Free hosting (Render, 512 MB RAM) cannot hold PyTorch plus the full RoBERTa model.
ONNX Runtime with dynamic int8 quantization needs no PyTorch at serving time and
is about 4x smaller. This script exports, quantizes, and then checks the compact
model against the full model on all 715 test posts, so any difference is reported.

Writes:
    models/roberta_onnx/model.onnx (+ tokenizer files, metadata.json)
    results/onnx_check.csv

Run from the project root, after scripts.train_roberta and scripts.calibrate_roberta:
    python -m scripts.export_onnx
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.config import RESULTS_DIR, ROBERTA_DIR, ROBERTA_ONNX_DIR
from ml.data import load_split
from ml.predictors import OnnxPredictor, RobertaPredictor, read_metadata, write_metadata

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def export(model_dir: Path, onnx_path: Path) -> None:
    """Trace the classifier to ONNX with dynamic batch and sequence length."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = tokenizer(["A short example post."], return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )


def main() -> None:
    """Export, quantize, copy tokenizer + metadata, and compare with the full model."""
    if ROBERTA_ONNX_DIR.exists():
        shutil.rmtree(ROBERTA_ONNX_DIR)
    ROBERTA_ONNX_DIR.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        export(ROBERTA_DIR, fp32)
        # Per-channel scales keep F1 unchanged on the test set; per-tensor scales lost ~1.4 F1 points.
        quantize_dynamic(str(fp32), str(ROBERTA_ONNX_DIR / "model.onnx"),
                         weight_type=QuantType.QInt8, per_channel=True)
    for name in TOKENIZER_FILES:
        shutil.copy(ROBERTA_DIR / name, ROBERTA_ONNX_DIR / name)
    write_metadata(ROBERTA_ONNX_DIR, {**read_metadata(ROBERTA_DIR), "format": "onnx-int8"})

    test = load_split("test")
    texts, labels = test["text"].tolist(), test["label"].to_numpy()
    full, compact = RobertaPredictor(), OnnxPredictor()
    p_full = full.stress_probabilities(texts)
    start = time.perf_counter()
    p_compact = compact.stress_probabilities(texts)
    ms_per_post = (time.perf_counter() - start) * 1000 / len(texts)
    full_pred, compact_pred = (p_full > 0.5).astype(int), (p_compact > 0.5).astype(int)
    report = {
        "size_mb_full": sum(f.stat().st_size for f in ROBERTA_DIR.glob("model.safetensors")) / 1e6,
        "size_mb_onnx_int8": (ROBERTA_ONNX_DIR / "model.onnx").stat().st_size / 1e6,
        "n_posts": len(texts),
        "label_match": int((full_pred == compact_pred).sum()),
        "max_prob_diff": float(np.abs(p_full - p_compact).max()),
        "f1_full": f1_score(labels, full_pred),
        "f1_onnx_int8": f1_score(labels, compact_pred),
        "ms_per_post_onnx_batch16": ms_per_post,
    }
    pd.DataFrame([report]).to_csv(RESULTS_DIR / "onnx_check.csv", index=False)
    print(pd.Series(report).round(4).to_string())


if __name__ == "__main__":
    main()
