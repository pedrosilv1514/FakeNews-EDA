"""Ajusta E01-E03 com CV por grupos e avalia no teste isolado."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config import load_config
from src.evaluation.metrics import (
    evaluate_estimator,
    plot_model_diagnostics,
    save_error_overlap,
    save_evaluation_artifacts,
    update_comparison_figures,
)
from src.models.classical import (
    MODEL_IDS,
    build_uncalibrated_pipeline,
    fit_estimator,
    grouped_cv_splits,
    parameter_grid,
)
from src.preprocessing.modeling import prepare_modeling_dataset


MODELS = tuple(MODEL_IDS)


def _json_safe_params(params: dict[str, object]) -> dict[str, object]:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}


def _record_environment(destination: Path) -> None:
    packages = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib", "pyarrow"]
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _upsert_metrics(path: Path, row: dict[str, object]) -> None:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty:
        current = current[current["experiment"] != row["experiment"]]
    pd.concat([current, pd.DataFrame([row])], ignore_index=True).sort_values("experiment").to_csv(
        path, index=False
    )


def train_baselines(selected_models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    cfg = load_config()
    processed_path = cfg["data"]["processed"] / "modeling_dataset.parquet"
    data, _ = prepare_modeling_dataset(output_path=processed_path)
    metrics_dir = cfg["project_root"] / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _record_environment(metrics_dir / "environment.json")
    comparison_path = metrics_dir / "experiment_comparison.csv"

    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    development = data[data["split"].isin(["train", "validation"])]
    test = data[data["split"] == "test"]
    all_rows = []
    for model_name in selected_models:
        experiment_id = MODEL_IDS[model_name]
        search_cv = grouped_cv_splits(train["label"], train["group_id"], seed=int(cfg["seed"]))
        search = GridSearchCV(
            build_uncalibrated_pipeline(model_name, int(cfg["seed"])),
            parameter_grid(model_name),
            scoring={"macro_f1": "f1_macro", "roc_auc": "roc_auc", "accuracy": "accuracy"},
            refit="macro_f1",
            cv=search_cv,
            n_jobs=-1,
            return_train_score=False,
        )
        started = time.perf_counter()
        search.fit(train["text"], train["label"])
        tuning_seconds = time.perf_counter() - started
        validation_metrics, _, _, _ = evaluate_estimator(
            search.best_estimator_, validation["text"], validation["label"], repeats=1
        )

        started = time.perf_counter()
        estimator = fit_estimator(
            model_name,
            search.best_params_,
            development["text"],
            development["label"],
            development["group_id"],
            seed=int(cfg["seed"]),
        )
        final_fit_seconds = time.perf_counter() - started
        metrics, report, predictions, probabilities = evaluate_estimator(
            estimator, test["text"], test["label"]
        )
        metadata = {
            "experiment": experiment_id,
            "model_name": model_name,
            "model_version": "1.0.0",
            "positive_class": "fake",
            "negative_class": "real",
            "threshold": 0.5,
            "best_params": _json_safe_params(search.best_params_),
            "cv_best_macro_f1": search.best_score_,
            "validation_macro_f1": validation_metrics["macro_f1"],
            "training_rows": len(development),
            "disclaimer": "Probabilidade de pertencimento a classe; nao comprova a veracidade factual.",
        }
        joblib.dump(
            {"estimator": estimator, "metadata": metadata},
            cfg["models"] / f"{experiment_id}_{model_name}.joblib",
            compress=3,
        )
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(metrics_dir / f"{experiment_id}_cv_results.csv", index=False)
        save_evaluation_artifacts(
            metrics_dir,
            experiment_id,
            test["record_id"],
            test["label"],
            predictions,
            probabilities,
            report,
        )
        plot_model_diagnostics(
            test["label"], probabilities, experiment_id, cfg["reports"]["figures"]
        )
        row = {
            "experiment": experiment_id,
            "model": model_name,
            **metrics,
            "cv_macro_f1": search.best_score_,
            "validation_macro_f1": validation_metrics["macro_f1"],
            "tuning_seconds": tuning_seconds,
            "training_seconds": final_fit_seconds,
            "best_params": json.dumps(_json_safe_params(search.best_params_), ensure_ascii=False),
            "status": "completed",
        }
        _upsert_metrics(comparison_path, row)
        all_rows.append(row)
        print(f"{experiment_id} concluido: macro-F1={metrics['macro_f1']:.4f}")

    save_error_overlap(metrics_dir, ["E01", "E02", "E03"])
    update_comparison_figures(comparison_path, cfg["reports"]["figures"])
    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    train_baselines(tuple(args.models))


if __name__ == "__main__":
    main()
