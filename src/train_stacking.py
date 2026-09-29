"""Treina E04 por previsoes OOF group-aware e avalia ablations."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.evaluation.metrics import (
    evaluate_estimator,
    plot_model_diagnostics,
    save_error_overlap,
    save_evaluation_artifacts,
    update_comparison_figures,
)
from src.models.classical import MODEL_IDS, fit_estimator, grouped_cv_splits, positive_scores
from src.models.stacking import ManualStackingClassifier
from src.preprocessing.modeling import prepare_modeling_dataset


BASE_NAMES = list(MODEL_IDS)


def _load_best_params(models_dir: Path) -> dict[str, dict[str, object]]:
    params = {}
    for name, experiment_id in MODEL_IDS.items():
        artifact = joblib.load(models_dir / f"{experiment_id}_{name}.joblib")
        raw = artifact["metadata"]["best_params"]
        params[name] = {
            key: tuple(value) if key.endswith("ngram_range") and isinstance(value, list) else value
            for key, value in raw.items()
        }
    return params


def _oof_features(frame: pd.DataFrame, params: dict[str, dict[str, object]], seed: int):
    splits = grouped_cv_splits(frame["label"], frame["group_id"], n_splits=5, seed=seed)
    features = np.full((len(frame), len(BASE_NAMES)), np.nan, dtype=float)
    fold_rows = []
    for fold, (fit_pos, holdout_pos) in enumerate(splits):
        fit_frame, holdout = frame.iloc[fit_pos], frame.iloc[holdout_pos]
        for column, name in enumerate(BASE_NAMES):
            estimator = fit_estimator(
                name,
                params[name],
                fit_frame["text"],
                fit_frame["label"],
                fit_frame["group_id"],
                seed=seed + fold,
                calibration_folds=3,
            )
            features[holdout_pos, column] = positive_scores(estimator, holdout["text"])
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(fit_frame),
                "holdout_rows": len(holdout),
                "train_groups": fit_frame["group_id"].nunique(),
                "holdout_groups": holdout["group_id"].nunique(),
                "group_overlap": len(set(fit_frame["group_id"]) & set(holdout["group_id"])),
            }
        )
    if np.isnan(features).any():
        raise RuntimeError("OOF incompleto: ha previsoes ausentes.")
    return features, pd.DataFrame(fold_rows)


def _fit_bases(frame: pd.DataFrame, params: dict[str, dict[str, object]], seed: int):
    return {
        name: fit_estimator(
            name,
            params[name],
            frame["text"],
            frame["label"],
            frame["group_id"],
            seed=seed,
        )
        for name in BASE_NAMES
    }


def _base_matrix(estimators: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([positive_scores(estimators[name], frame["text"]) for name in BASE_NAMES])


def _upsert(path: Path, row: dict[str, object]) -> None:
    data = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not data.empty:
        data = data[data["experiment"] != row["experiment"]]
    pd.concat([data, pd.DataFrame([row])], ignore_index=True).sort_values("experiment").to_csv(
        path, index=False
    )


def train_stacking() -> dict[str, object]:
    cfg = load_config()
    data, _ = prepare_modeling_dataset()
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    development = data[data["split"].isin(["train", "validation"])].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    params = _load_best_params(cfg["models"])
    metrics_dir = cfg["project_root"] / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train_oof, selection_folds = _oof_features(train, params, int(cfg["seed"]))
    selection_bases = _fit_bases(train, params, int(cfg["seed"]))
    validation_features = _base_matrix(selection_bases, validation)
    ablations = []
    candidates = []
    for size in (2, 3):
        candidates.extend(itertools.combinations(range(len(BASE_NAMES)), size))
    for columns in candidates:
        meta = LogisticRegression(max_iter=1_000, random_state=int(cfg["seed"]))
        meta.fit(train_oof[:, columns], train["label"])
        probability = meta.predict_proba(validation_features[:, columns])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        from sklearn.metrics import f1_score, log_loss

        ablations.append(
            {
                "base_learners": "+".join(BASE_NAMES[index] for index in columns),
                "columns": list(columns),
                "n_base_learners": len(columns),
                "validation_macro_f1": f1_score(
                    validation["label"], prediction, average="macro", zero_division=0
                ),
                "validation_log_loss": log_loss(
                    validation["label"], np.c_[1 - probability, probability], labels=[0, 1]
                ),
            }
        )
    ablation_frame = pd.DataFrame(ablations).sort_values(
        ["validation_macro_f1", "validation_log_loss", "n_base_learners"],
        ascending=[False, True, True],
    )
    ablation_frame.drop(columns="columns").to_csv(metrics_dir / "E04_ablation_results.csv", index=False)
    best = ablation_frame.iloc[0]
    selected_columns = list(best["columns"])
    selected_names = [BASE_NAMES[index] for index in selected_columns]

    # Regera OOF usando todo o development; o teste segue completamente intocado.
    development_oof, final_folds = _oof_features(development, params, int(cfg["seed"]) + 101)
    meta = LogisticRegression(max_iter=1_000, random_state=int(cfg["seed"]))
    meta.fit(development_oof[:, selected_columns], development["label"])
    final_bases_all = _fit_bases(development, params, int(cfg["seed"]) + 101)
    final_bases = {name: final_bases_all[name] for name in selected_names}
    estimator = ManualStackingClassifier(final_bases, meta, selected_names)
    training_seconds = time.perf_counter() - started

    metrics, report, predictions, probabilities = evaluate_estimator(
        estimator, test["text"], test["label"]
    )
    metadata = {
        "experiment": "E04",
        "model_name": "manual_oof_stacking",
        "model_version": "1.0.0",
        "base_learners": selected_names,
        "meta_learner": "logistic_regression",
        "positive_class": "fake",
        "threshold": 0.5,
        "oof_folds": 5,
        "validation_macro_f1": float(best["validation_macro_f1"]),
        "meta_coefficients": meta.coef_.ravel().tolist(),
        "disclaimer": "Probabilidade de pertencimento a classe; nao comprova a veracidade factual.",
    }
    joblib.dump(
        {"estimator": estimator, "metadata": metadata},
        cfg["models"] / "E04_stacking.joblib",
        compress=3,
    )
    selection_folds.to_csv(metrics_dir / "E04_selection_oof_folds.csv", index=False)
    final_folds.to_csv(metrics_dir / "E04_final_oof_folds.csv", index=False)
    save_evaluation_artifacts(
        metrics_dir,
        "E04",
        test["record_id"],
        test["label"],
        predictions,
        probabilities,
        report,
    )
    plot_model_diagnostics(test["label"], probabilities, "E04", cfg["reports"]["figures"])
    row = {
        "experiment": "E04",
        "model": "manual_oof_stacking",
        **metrics,
        "cv_macro_f1": float("nan"),
        "validation_macro_f1": float(best["validation_macro_f1"]),
        "tuning_seconds": float("nan"),
        "training_seconds": training_seconds,
        "best_params": json.dumps({"base_learners": selected_names}, ensure_ascii=False),
        "status": "completed",
    }
    comparison_path = metrics_dir / "experiment_comparison.csv"
    _upsert(comparison_path, row)
    save_error_overlap(metrics_dir, ["E01", "E02", "E03", "E04"])
    update_comparison_figures(comparison_path, cfg["reports"]["figures"])
    print(f"E04 concluido ({'+'.join(selected_names)}): macro-F1={metrics['macro_f1']:.4f}")
    return row


def main() -> None:
    train_stacking()


if __name__ == "__main__":
    main()
