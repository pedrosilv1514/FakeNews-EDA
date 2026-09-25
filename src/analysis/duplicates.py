"""Deteccao de duplicatas exatas e candidatos quase duplicados."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pandas as pd

from src.preprocessing.clean_text import normalize_for_matching


def fingerprint(value: object) -> str:
    """Gera assinatura SHA-256 de uma representacao textual normalizada."""
    normalized = normalize_for_matching(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def duplicate_summary(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Resume duplicatas completas e por campos sensiveis."""
    rows = [
        {
            "dataset": dataset_name,
            "scope": "full_row",
            "duplicated_rows": int(frame.duplicated().sum()),
            "duplicate_groups": int(frame[frame.duplicated(keep=False)].drop_duplicates().shape[0]),
        }
    ]
    for column in ("title", "text", "url"):
        if column not in frame:
            continue
        normalized = frame[column].map(normalize_for_matching)
        usable = normalized.ne("")
        duplicated = normalized[usable].duplicated(keep=False)
        rows.append(
            {
                "dataset": dataset_name,
                "scope": column,
                "duplicated_rows": int(duplicated.sum()),
                "duplicate_groups": int(normalized[usable][duplicated].nunique()),
            }
        )
    return pd.DataFrame(rows)


def cross_dataset_duplicates(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Encontra sobreposicao exata entre datasets por URL ou texto normalizado."""
    signatures: list[pd.DataFrame] = []
    for name, frame in datasets.items():
        comparable_label = frame["label_semantic"].replace({"not_fake": "real"})
        local = pd.DataFrame({"dataset": name, "label": comparable_label})
        local["url_key"] = frame["url"].map(normalize_for_matching)
        local["text_key"] = frame["text"].map(fingerprint)
        signatures.append(local)
    combined = pd.concat(signatures, ignore_index=True)

    rows: list[dict[str, object]] = []
    for key_type in ("url_key", "text_key"):
        valid = combined[combined[key_type].ne("")]
        for key, group in valid.groupby(key_type):
            names = sorted(group["dataset"].unique())
            if len(names) < 2:
                continue
            rows.append(
                {
                    "match_type": key_type.removesuffix("_key"),
                    "signature": key,
                    "datasets": " | ".join(names),
                    "records": len(group),
                    "labels": " | ".join(sorted(group["label"].dropna().astype(str).unique())),
                    "label_conflict": group["label"].dropna().nunique() > 1,
                }
            )
    return pd.DataFrame(rows)


def near_duplicate_candidates(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.92,
    max_features: int = 50_000,
) -> pd.DataFrame:
    """Busca candidatos proximos com TF-IDF de caracteres e vizinho mais proximo.

    A funcao e diagnostica: pares retornados precisam de inspecao humana antes de
    qualquer remocao. Textos vazios e muito curtos sao excluidos.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    normalized = frame["text"].map(normalize_for_matching)
    eligible = normalized.str.len().ge(80)
    texts = normalized[eligible]
    if len(texts) < 2:
        return pd.DataFrame(columns=["left_index", "right_index", "similarity"])
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 5), min_df=2, max_features=max_features
    ).fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1).fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    rows: list[dict[str, object]] = []
    original_indices = texts.index.to_numpy()
    seen: set[tuple[object, object]] = set()
    for position, (distance_row, index_row) in enumerate(zip(distances, indices)):
        left = original_indices[position]
        right = original_indices[index_row[1]]
        pair = tuple(sorted((left, right), key=str))
        similarity = 1 - float(distance_row[1])
        if similarity >= threshold and pair not in seen:
            seen.add(pair)
            rows.append({"left_index": left, "right_index": right, "similarity": similarity})
    return pd.DataFrame(rows)
