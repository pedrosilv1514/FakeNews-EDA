"""Factories dos modelos textuais tradicionais."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


MODEL_IDS = {
    "logistic_regression": "E01",
    "linear_svm": "E02",
    "multinomial_nb": "E03",
}


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        sublinear_tf=True,
        max_df=0.98,
        max_features=100_000,
        dtype=np.float32,
    )


def build_uncalibrated_pipeline(model_name: str, seed: int = 42) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2_000, solver="liblinear", random_state=seed
        )
    elif model_name == "linear_svm":
        classifier = LinearSVC(random_state=seed)
    elif model_name == "multinomial_nb":
        classifier = MultinomialNB()
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return Pipeline([("tfidf", _vectorizer()), ("clf", classifier)])


def parameter_grid(model_name: str) -> dict[str, list[object]]:
    common: dict[str, list[object]] = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
    }
    if model_name == "logistic_regression":
        common["clf__C"] = [0.5, 2.0]
    elif model_name == "linear_svm":
        common["clf__C"] = [0.5, 1.5]
    elif model_name == "multinomial_nb":
        common["clf__alpha"] = [0.1, 0.5]
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return common


def grouped_cv_splits(
    y: Sequence[int], groups: Sequence[str], n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    y_array = np.asarray(y)
    group_array = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(y_array), 1))
    return list(splitter.split(dummy, y_array, group_array))


def fit_estimator(
    model_name: str,
    params: dict[str, object],
    X: Sequence[str],
    y: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int = 42,
    calibration_folds: int = 5,
):
    """Ajusta modelo final; o SVM recebe calibracao sigmoid group-aware."""
    pipeline = build_uncalibrated_pipeline(model_name, seed).set_params(**params)
    if model_name != "linear_svm":
        return pipeline.fit(X, y)
    cv = grouped_cv_splits(y, groups, n_splits=calibration_folds, seed=seed + 17)
    calibrated = CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=cv,
        ensemble=True,
        n_jobs=-1,
    )
    return calibrated.fit(X, y)


def positive_scores(estimator, X: Sequence[str]) -> np.ndarray:
    """Converte mecanismos distintos no score da classe fake."""
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))[:, 1]
    scores = np.asarray(estimator.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -35, 35)))
