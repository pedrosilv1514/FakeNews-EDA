"""Avaliacao adicional treino-passado/teste-futuro dos baselines."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import evaluate_estimator
from src.models.classical import MODEL_IDS, fit_estimator
from src.preprocessing.modeling import prepare_modeling_dataset


def evaluate_temporal() -> pd.DataFrame:
    cfg = load_config()
    data, audit = prepare_modeling_dataset()
    train = data[data["temporal_split"] == "train"]
    test = data[data["temporal_split"] == "test"]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise RuntimeError("Split temporal nao contem ambas as classes.")
    rows = []
    for name, experiment_id in MODEL_IDS.items():
        artifact = joblib.load(cfg["models"] / f"{experiment_id}_{name}.joblib")
        raw = artifact["metadata"]["best_params"]
        params = {
            key: tuple(value) if key.endswith("ngram_range") and isinstance(value, list) else value
            for key, value in raw.items()
        }
        estimator = fit_estimator(
            name, params, train["text"], train["label"], train["group_id"], seed=int(cfg["seed"])
        )
        metrics, _, _, _ = evaluate_estimator(estimator, test["text"], test["label"])
        rows.append(
            {
                "experiment": experiment_id,
                "model": name,
                "cutoff": audit["temporal_cutoff"],
                "train_rows": len(train),
                "test_rows": len(test),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    output = cfg["project_root"] / "reports/metrics/temporal_evaluation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    evaluate_temporal()
