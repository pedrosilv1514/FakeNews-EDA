"""Metricas textuais interpretaveis para EDA."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.preprocessing.clean_text import collapse_whitespace


WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ]+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def text_features(value: object) -> dict[str, float | int]:
    """Calcula atributos superficiais sem stemming, lematizacao ou stopwords."""
    text = collapse_whitespace(value)
    letters = [character for character in text if character.isalpha()]
    uppercase = sum(character.isupper() for character in letters)
    sentence_count = len(SENTENCE_RE.findall(text))
    if text and sentence_count == 0:
        sentence_count = 1
    return {
        "n_characters": len(text),
        "n_words": len(WORD_RE.findall(text)),
        "n_sentences": sentence_count,
        "n_exclamations": text.count("!"),
        "n_questions": text.count("?"),
        "n_urls": len(URL_RE.findall(text)),
        "n_numbers": len(NUMBER_RE.findall(text)),
        "uppercase_percent": 100 * uppercase / len(letters) if letters else np.nan,
    }


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona metricas de texto e titulo a uma copia do DataFrame."""
    result = frame.copy()
    metrics = pd.DataFrame(result["text"].map(text_features).tolist(), index=result.index)
    for column in metrics:
        result[column] = metrics[column]
    result["title_length"] = result.get(
        "title", pd.Series("", index=result.index, dtype="string")
    ).map(lambda value: len(collapse_whitespace(value)))
    return result


def summarize_text_features(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Resume metricas por classe usando mediana e intervalo interquartil."""
    metrics = [
        "n_characters",
        "n_words",
        "n_sentences",
        "title_length",
        "n_exclamations",
        "n_questions",
        "n_urls",
        "n_numbers",
        "uppercase_percent",
    ]
    available = [column for column in metrics if column in frame]
    summary = (
        frame.dropna(subset=["label_semantic"]).groupby("label_semantic")[available]
        .agg(["median", lambda values: values.quantile(0.25), lambda values: values.quantile(0.75)])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "q25").replace("<lambda_1>", "q75"))
    summary = summary.reset_index()
    summary.insert(0, "dataset", dataset_name)
    return summary
