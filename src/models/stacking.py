"""Stacking manual para permitir OOF e calibracao respeitando grupos."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.models.classical import positive_scores


class ManualStackingClassifier:
    """Artefato de inferencia do stacking treinado fora da classe."""

    def __init__(self, base_estimators: dict[str, object], meta_estimator, base_order: list[str]):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.base_order = base_order
        self.classes_ = np.array([0, 1])

    def transform(self, X: Sequence[str]) -> np.ndarray:
        return np.column_stack(
            [positive_scores(self.base_estimators[name], X) for name in self.base_order]
        )

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        return self.meta_estimator.predict_proba(self.transform(X))

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
