"""Tests for splits, baselines and saved-model predictions."""

import numpy as np
import pandas as pd
import pytest

from ml.baselines import build_tfidf, fit_majority
from ml.config import RESULTS_DIR, ROBERTA_ONNX_DIR, TFIDF_DIR, TOPICS
from ml.data import load_split
from ml.predictors import OnnxPredictor, TfidfPredictor, load_predictor, to_prediction
from ml.reporting import load_all_predictions
from ml.splits import NEW_TOPIC, STANDARD, experiments, validation_split


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return load_split("train")


@pytest.fixture(scope="module")
def test() -> pd.DataFrame:
    return load_split("test")


def test_validation_split_is_disjoint_and_stratified(train) -> None:
    fit_part, val = validation_split(train)
    assert set(fit_part["id"]).isdisjoint(val["id"])
    assert len(fit_part) + len(val) == len(train)
    assert abs(fit_part["label"].mean() - val["label"].mean()) < 0.02


def test_new_topic_splits_hold_the_topic_out(train, test) -> None:
    exps = list(experiments(train, test))
    assert [e.setting for e in exps] == [STANDARD] + [NEW_TOPIC] * len(TOPICS)
    for exp in exps[1:]:
        assert exp.test_set not in set(exp.train["topic"])
        assert set(exp.test["topic"]) == {exp.test_set}


def test_train_and_test_do_not_overlap(train, test) -> None:
    assert set(train["id"]).isdisjoint(test["id"])


def test_majority_predicts_one_class(train) -> None:
    model = fit_majority(train)
    assert set(model.predict(["any text", "more text"])) == {train["label"].mode()[0]}


def test_tfidf_learns_simple_pattern() -> None:
    texts = ["i am so stressed and scared"] * 10 + ["what a calm happy day"] * 10
    model = build_tfidf().fit(texts, [1] * 10 + [0] * 10)
    assert list(model.predict(["stressed and scared", "calm happy"])) == [1, 0]


@pytest.mark.parametrize(("prob", "label", "conf"), [(0.9, "stress", 0.9), (0.2, "no stress", 0.8),
                                                     (0.5, "no stress", 0.5)])
def test_to_prediction(prob: float, label: str, conf: float) -> None:
    pred = to_prediction(prob)
    assert pred.label == label and pred.confidence == pytest.approx(conf)


needs_tfidf = pytest.mark.skipif(not (TFIDF_DIR / "model.joblib").exists(),
                                 reason="run scripts.train_baselines first")


@needs_tfidf
def test_saved_tfidf_matches_saved_predictions(test) -> None:
    saved = load_all_predictions()["tfidf"]
    saved = saved[saved["setting"] == STANDARD].set_index("id").loc[test["id"]]
    probs = TfidfPredictor().stress_probabilities(test["text"])
    np.testing.assert_allclose(probs, saved["stress_prob"].to_numpy(), atol=1e-9)


@needs_tfidf
def test_load_predictor_falls_back_to_tfidf(tmp_path) -> None:
    assert load_predictor(roberta_dir=tmp_path, onnx_dir=tmp_path).name == "tfidf"


@pytest.mark.skipif(not (ROBERTA_ONNX_DIR / "model.onnx").exists(), reason="run scripts.export_onnx first")
def test_onnx_model_is_used_without_full_roberta_and_matches_saved_check(tmp_path, test) -> None:
    predictor = load_predictor(roberta_dir=tmp_path)
    assert isinstance(predictor, OnnxPredictor) and predictor.name == "roberta"
    probs = predictor.stress_probabilities(test["text"].head(8))
    assert probs.shape == (8,) and ((probs >= 0) & (probs <= 1)).all()
    check = pd.read_csv(RESULTS_DIR / "onnx_check.csv").iloc[0]
    assert check["label_match"] / check["n_posts"] > 0.95
    assert abs(check["f1_onnx_int8"] - check["f1_full"]) < 0.01
