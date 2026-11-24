"""Tests for the GoEmotions extension: data, metrics, thresholds, predictors and the API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from ml.emotions import (
    EMOTION_TFIDF_DIR,
    EMOTIONS,
    emotion_raw_path,
    load_emotions,
    macro_micro_f1,
    multi_hot,
    paired_bootstrap_macro,
    per_label_report,
    tune_thresholds,
)

needs_data = pytest.mark.skipif(not emotion_raw_path("test").exists(), reason="run scripts.prepare_emotions first")
needs_tfidf = pytest.mark.skipif(not (EMOTION_TFIDF_DIR / "model.joblib").exists(),
                                 reason="run scripts.train_emotions --model tfidf first")


def test_multi_hot_marks_every_label() -> None:
    y = multi_hot([[0, 27], [3], []], n_labels=28)
    assert y.shape == (3, 28)
    assert y[0, 0] == y[0, 27] == y[1, 3] == 1 and y.sum() == 3


def test_macro_and_micro_f1() -> None:
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    pred = np.array([[1, 0], [0, 0], [0, 1], [0, 1]])
    macro, micro = macro_micro_f1(y, pred)
    # label 0: tp1 fn1 -> F1 2/3 ; label 1: tp1 fp1 -> F1 2/3 ; pooled tp2 fp1 fn1 -> 2/3
    assert macro == pytest.approx(2 / 3) and micro == pytest.approx(2 / 3)


def test_thresholds_are_tuned_per_emotion() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (400, 2))
    probs = np.column_stack([0.2 + 0.1 * y[:, 0], 0.6 + 0.3 * y[:, 1]])  # label 0 separable at 0.25, label 1 at 0.75
    thresholds = tune_thresholds(y, probs)
    assert 0.2 < thresholds[0] <= 0.3 and 0.6 < thresholds[1] <= 0.9
    assert macro_micro_f1(y, (probs >= thresholds).astype(int))[0] == pytest.approx(1.0)


def test_paired_bootstrap_prefers_the_better_model() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, (300, 3))
    noisy = np.where(rng.random(y.shape) < 0.4, 1 - y, y)
    result = paired_bootstrap_macro(y, noisy, y, n_resamples=300)
    assert result["difference"] > 0 and result["p_value"] < 0.01


def test_per_label_report_has_one_row_per_emotion() -> None:
    y = np.zeros((4, len(EMOTIONS)), dtype=int)
    report = per_label_report(y, y, np.full(len(EMOTIONS), 0.5))
    assert list(report["emotion"]) == EMOTIONS


@needs_data
def test_goemotions_splits_load_with_28_labels() -> None:
    df, y = load_emotions("test")
    assert len(df) == len(y) == 5427 and y.shape[1] == 28
    assert (y.sum(axis=1) >= 1).all()  # every comment has at least one label (possibly neutral)


@needs_tfidf
def test_emotions_endpoint_returns_all_labels(tmp_path) -> None:
    settings = Settings(roberta_dir=tmp_path / "none", onnx_dir=tmp_path / "none",
                        emotion_roberta_dir=tmp_path / "none")
    with TestClient(create_app(settings)) as client:
        body = client.post("/emotions", json={"text": "Thank you so much, this made my day!"}).json()
        assert client.get("/model-info").json()["emotion_model"] == "emotions_tfidf"
    assert body["available"] and len(body["emotions"]) == 28
    probs = [e["probability"] for e in body["emotions"]]
    assert probs == sorted(probs, reverse=True)
    assert body["emotions"][0]["emotion"] == "gratitude"


def test_emotions_endpoint_without_a_model(tmp_path) -> None:
    settings = Settings(roberta_dir=tmp_path / "none", onnx_dir=tmp_path / "none",
                        emotion_roberta_dir=tmp_path / "none", emotion_tfidf_dir=tmp_path / "none")
    with TestClient(create_app(settings)) as client:
        assert client.post("/emotions", json={"text": "hello"}).json() == {
            "available": False, "model_name": None, "display_name": None, "emotions": [], "latency_ms": None,
        }
        assert client.post("/emotions", json={"text": " "}).status_code == 422
