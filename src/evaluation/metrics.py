"""Metricas, artefatos tabulares e figuras dos experimentos."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.classical import positive_scores


def evaluate_estimator(estimator, X, y, *, repeats: int = 3):
    y_array = np.asarray(y, dtype=int)
    start = time.perf_counter()
    for _ in range(repeats):
        probabilities = positive_scores(estimator, X)
    inference_seconds = (time.perf_counter() - start) / repeats
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_array, predictions),
        "precision": precision_score(y_array, predictions, zero_division=0),
        "recall": recall_score(y_array, predictions, zero_division=0),
        "f1": f1_score(y_array, predictions, zero_division=0),
        "macro_f1": f1_score(y_array, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_array, probabilities),
        "pr_auc": average_precision_score(y_array, probabilities),
        "brier_score": brier_score_loss(y_array, probabilities),
        "log_loss": log_loss(y_array, np.c_[1 - probabilities, probabilities], labels=[0, 1]),
        "inference_seconds": inference_seconds,
        "inference_ms_per_sample": inference_seconds * 1000 / len(y_array),
        "n_samples": len(y_array),
    }
    report = classification_report(
        y_array,
        predictions,
        labels=[0, 1],
        target_names=["real", "fake"],
        output_dict=True,
        zero_division=0,
    )
    return metrics, report, predictions, probabilities


def save_evaluation_artifacts(
    output_dir: Path,
    experiment_id: str,
    record_ids,
    y,
    predictions,
    probabilities,
    report: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "record_id": list(record_ids),
            "y_true": np.asarray(y, dtype=int),
            "y_pred": predictions,
            "probability_fake": probabilities,
        }
    )
    prediction_frame["error"] = prediction_frame["y_true"] != prediction_frame["y_pred"]
    prediction_frame.to_csv(output_dir / f"{experiment_id}_test_predictions.csv", index=False)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=["real", "fake"], columns=["real", "fake"]).to_csv(
        output_dir / f"{experiment_id}_confusion_matrix.csv"
    )
    (output_dir / f"{experiment_id}_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_model_diagnostics(y, probabilities, experiment_id: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    y_array = np.asarray(y, dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fpr, tpr, _ = roc_curve(y_array, probabilities)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_array, probabilities):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC", xlabel="Falso positivo", ylabel="Verdadeiro positivo")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_array, probabilities)
    axes[1].plot(recall, precision, label=f"AP={average_precision_score(y_array, probabilities):.3f}")
    axes[1].set(title="Precisao-recall", xlabel="Recall", ylabel="Precisao")
    axes[1].legend()
    observed, predicted = calibration_curve(y_array, probabilities, n_bins=10, strategy="quantile")
    axes[2].plot(predicted, observed, marker="o")
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    axes[2].set(title="Calibracao", xlabel="Probabilidade media", ylabel="Frequencia observada")
    fig.suptitle(experiment_id)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{experiment_id}_diagnostics.png", dpi=160)
    plt.close(fig)


def update_comparison_figures(metrics_path: Path, figures_dir: Path) -> None:
    if not metrics_path.exists():
        return
    data = pd.read_csv(metrics_path).sort_values("experiment")
    if data.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    quality = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
    axes = data.set_index("experiment")[quality].plot.bar(figsize=(12, 5), ylim=(0, 1))
    axes.set_ylabel("Score no teste isolado")
    axes.set_title("Comparacao dos modelos")
    axes.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=160)
    plt.close()

    calibration = ["brier_score", "log_loss"]
    axes = data.set_index("experiment")[calibration].plot.bar(figsize=(8, 4))
    axes.set_ylabel("Menor e melhor")
    axes.set_title("Qualidade probabilistica")
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_comparison.png", dpi=160)
    plt.close()


def save_error_overlap(metrics_dir: Path, experiment_ids: list[str]) -> None:
    frames = {}
    for experiment_id in experiment_ids:
        path = metrics_dir / f"{experiment_id}_test_predictions.csv"
        if path.exists():
            frames[experiment_id] = pd.read_csv(path).set_index("record_id")["error"].astype(bool)
    if len(frames) < 2:
        return
    errors = pd.DataFrame(frames).dropna().astype(bool)
    matrix = pd.DataFrame(index=frames, columns=frames, dtype=float)
    for left in frames:
        for right in frames:
            union = (errors[left] | errors[right]).sum()
            matrix.loc[left, right] = (errors[left] & errors[right]).sum() / union if union else 1.0
    matrix.to_csv(metrics_dir / "error_overlap_jaccard.csv")
    errors.astype(int).corr().to_csv(metrics_dir / "error_correlation.csv")
