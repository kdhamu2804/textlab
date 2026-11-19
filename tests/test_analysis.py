"""Tests for confusion matrices, significance tests, calibration and explanations."""

import numpy as np
import pytest

from ml.analysis import calibration_bins, calibration_summary, confusion, mcnemar, paired_bootstrap
from ml.calibration import fit_temperature, nll, stress_probability
from ml.config import TFIDF_DIR
from ml.predictors import Predictor, TfidfPredictor, sentence_spans, word_spans

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])


def test_confusion_counts() -> None:
    pred = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    assert confusion(Y, pred) == {"tn": 4, "fp": 1, "fn": 2, "tp": 3}


def test_paired_bootstrap_detects_a_clearly_better_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    weak = np.where(rng.random(400) < 0.6, y, 1 - y)     # ~60% accurate
    strong = np.where(rng.random(400) < 0.95, y, 1 - y)  # ~95% accurate
    result = paired_bootstrap(y, weak, strong, n_resamples=2000)
    assert result["difference"] > 0 and result["difference_low"] > 0
    assert result["p_value"] < 0.01


def test_paired_bootstrap_identical_models_is_not_significant() -> None:
    result = paired_bootstrap(Y, Y, Y, n_resamples=500)
    assert result["difference"] == 0 and result["p_value"] == 1.0


def test_mcnemar_counts_disagreements() -> None:
    a = Y.copy()
    b = Y.copy()
    b[:3] = 1 - b[:3]  # b wrong on 3 posts where a is right
    result = mcnemar(Y, a, b)
    assert (result["only_a_correct"], result["only_b_correct"]) == (3, 0)
    assert result["mcnemar_p"] == pytest.approx(0.25)


def test_calibration_of_perfect_probabilities_is_zero() -> None:
    y = np.array([0, 0, 1, 1])
    assert calibration_summary(y, y.astype(float)) == {"ece": 0.0, "brier": 0.0, "n": 4}


def test_calibration_bins_cover_every_post() -> None:
    probs = np.linspace(0, 1, len(Y))
    bins = calibration_bins(Y, probs)
    assert bins["n"].sum() == len(Y)
    assert (bins["mean_predicted"].between(bins["bin_low"], bins["bin_high"])).all()


def test_temperature_scaling_fixes_overconfidence_without_changing_labels() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    margin = np.where(rng.random(2000) < 0.8, 1, -1) * (2 * y - 1) * 4.0  # 80% right but very confident
    logits = np.stack([-margin / 2, margin / 2], axis=1)
    t = fit_temperature(logits, y)
    assert t > 1
    assert nll(logits, y, t) < nll(logits, y)
    assert np.array_equal(stress_probability(logits) > 0.5, stress_probability(logits, t) > 0.5)


def test_word_and_sentence_spans_point_into_the_text() -> None:
    text = "I can't sleep.  Rent is due!\nHelp me, please"
    for spans in (word_spans(text), sentence_spans(text)):
        assert spans and all(text[s:e] == t for t, s, e in spans)
    assert [t for t, _, _ in sentence_spans(text)] == ["I can't sleep.", "Rent is due!", "Help me, please"]
    assert "can't" in [t for t, _, _ in word_spans(text)]


def test_short_posts_are_split_into_clauses() -> None:
    assert len(sentence_spans("I cannot sleep, rent is due and I am scared.")) == 2


class _CountWords(Predictor):
    """Toy model: P(stress) = share of words that are 'panic'."""

    name = display_name = "toy"

    def __init__(self) -> None:
        self.metadata = {}

    def stress_probabilities(self, texts):
        return np.array([t.split().count("panic") / max(len(t.split()), 1) for t in texts])


def test_occlusion_credits_the_sentence_that_drives_the_prediction() -> None:
    explanation = _CountWords().explain("panic panic. calm day. nice walk.")
    assert explanation.unit == "sentence"
    scores = {s.text: s.score for s in explanation.segments}
    assert scores["panic panic."] == 1.0
    assert all(scores[k] < 0 for k in ("calm day.", "nice walk."))


@pytest.mark.skipif(not (TFIDF_DIR / "model.joblib").exists(), reason="run scripts.train_baselines first")
def test_tfidf_explanation_is_word_level_and_scaled() -> None:
    explanation = TfidfPredictor().explain("I am terrified and I cannot sleep at all.")
    assert explanation.unit == "word"
    scores = [s.score for s in explanation.segments]
    assert max(abs(s) for s in scores) == pytest.approx(1.0)
    assert {s.text for s in explanation.segments} >= {"terrified", "sleep"}
