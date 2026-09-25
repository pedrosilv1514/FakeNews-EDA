"""Graficos Matplotlib reprodutiveis para o relatorio."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_class_distribution(frame: pd.DataFrame, path: Path | str) -> Path:
    """Compara desbalanceamento de classes entre datasets."""
    pivot = frame.pivot(index="dataset", columns="label_semantic", values="count").fillna(0)
    fig, axis = plt.subplots(figsize=(10, max(4, 0.6 * len(pivot))))
    pivot.plot(kind="barh", ax=axis)
    axis.set(title="Distribuicao de classes por dataset", xlabel="Registros", ylabel="")
    axis.legend(title="Classe")
    return _save(fig, path)


def plot_text_length_by_class(frame: pd.DataFrame, path: Path | str) -> Path:
    """Mostra se o tamanho do texto separa classes (escala limitada ao p99)."""
    labels = sorted(frame["label_semantic"].dropna().unique())
    groups = [frame.loc[frame["label_semantic"].eq(label), "n_words"].dropna() for label in labels]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(groups, tick_labels=labels, showfliers=False)
    axis.set(title="O comprimento do texto difere por classe?", ylabel="Numero de palavras")
    return _save(fig, path)


def plot_punctuation_by_class(frame: pd.DataFrame, path: Path | str) -> Path:
    """Compara pontuacao potencialmente sensacionalista por classe."""
    summary = frame.groupby("label_semantic")[["n_exclamations", "n_questions"]].mean()
    fig, axis = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", ax=axis)
    axis.set(title="Pontuacao media por classe", xlabel="Classe", ylabel="Media por noticia")
    axis.tick_params(axis="x", rotation=0)
    return _save(fig, path)


def plot_top_domains(frame: pd.DataFrame, path: Path | str, top_n: int = 15) -> Path:
    """Evidencia se dominios frequentes estao associados a classes."""
    usable = frame[frame["domain"].astype("string").str.strip().ne("")]
    top = usable["domain"].value_counts().head(top_n).index
    table = pd.crosstab(usable["domain"], usable["label_semantic"], normalize="index").loc[top]
    fig, axis = plt.subplots(figsize=(10, max(5, 0.4 * len(table))))
    table.plot(kind="barh", stacked=True, ax=axis)
    axis.set(title="Composicao de classe nos dominios mais frequentes", xlabel="Proporcao", ylabel="")
    axis.legend(title="Classe", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, path)

