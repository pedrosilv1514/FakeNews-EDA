"""Analise de frequencia, n-gramas e TF-IDF por classe."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def top_terms_by_class(
    frame: pd.DataFrame,
    *,
    method: str = "count",
    ngram_range: tuple[int, int] = (1, 1),
    top_n: int = 25,
    min_df: int = 3,
) -> pd.DataFrame:
    """Calcula termos mais fortes por classe sem remover stopwords automaticamente."""
    vectorizer_class = CountVectorizer if method == "count" else TfidfVectorizer
    if method not in {"count", "tfidf"}:
        raise ValueError("method deve ser 'count' ou 'tfidf'.")
    rows: list[dict[str, object]] = []
    for label, group in frame.dropna(subset=["label_semantic"]).groupby("label_semantic"):
        texts = group["text"].fillna("").astype(str)
        if texts.str.strip().eq("").all():
            continue
        vectorizer = vectorizer_class(
            lowercase=True,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=50_000,
        )
        matrix = vectorizer.fit_transform(texts)
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranking = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)[:top_n]
        rows.extend(
            {
                "label": label,
                "method": method,
                "ngram": f"{ngram_range[0]}-{ngram_range[1]}",
                "term": term,
                "score": float(score),
            }
            for term, score in ranking
        )
    return pd.DataFrame(rows)


def class_distinctive_terms(frame: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Termos discriminantes por classe via media de TF-IDF, para auditar atalhos."""
    usable = frame.dropna(subset=["label_semantic"]).copy()
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2), max_features=60_000)
    matrix = vectorizer.fit_transform(usable["text"].fillna("").astype(str))
    terms = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []
    for label in sorted(usable["label_semantic"].unique()):
        mask = usable["label_semantic"].eq(label).to_numpy()
        own = matrix[mask].mean(axis=0).A1
        other = matrix[~mask].mean(axis=0).A1
        difference = own - other
        best = difference.argsort()[::-1][:top_n]
        rows.extend(
            {"label": label, "term": terms[index], "mean_tfidf_difference": float(difference[index])}
            for index in best
        )
    return pd.DataFrame(rows)

