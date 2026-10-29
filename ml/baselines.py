"""Majority-class and TF-IDF + logistic regression baselines."""

from collections.abc import Sequence

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.config import SEED
from ml.evaluation import compute_metrics
from ml.splits import validation_split

C_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def build_majority() -> Pipeline:
    """Return a classifier that always predicts the most frequent training label."""
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])


def build_tfidf(c: float = 1.0) -> Pipeline:
    """Return a TF-IDF (word 1-2 grams) + logistic regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=c, max_iter=2000, random_state=SEED)),
        ]
    )


def tune_tfidf(
    train: pd.DataFrame, grid: Sequence[float] = C_GRID
) -> tuple[float, pd.DataFrame]:
    """Pick the regularization strength C by stress-class F1 on a validation split.

    Only training data is used: it is split into train and validation parts.
    Returns the best C and a table of validation scores for every C.
    """
    fit_part, val_part = validation_split(train)
    rows = []
    for c in grid:
        model = build_tfidf(c).fit(fit_part["text"], fit_part["label"])
        scores = compute_metrics(val_part["label"], model.predict(val_part["text"]))
        rows.append({"C": c, **scores})
    table = pd.DataFrame(rows)
    best_c = float(table.loc[table["f1"].idxmax(), "C"])
    return best_c, table


def fit_majority(train: pd.DataFrame) -> Pipeline:
    """Fit the majority-class baseline."""
    return build_majority().fit(train["text"], train["label"])


def fit_tfidf(train: pd.DataFrame) -> tuple[Pipeline, float, pd.DataFrame]:
    """Tune C on validation data, then refit on all of ``train`` with the best C."""
    best_c, table = tune_tfidf(train)
    model = build_tfidf(best_c).fit(train["text"], train["label"])
    return model, best_c, table
