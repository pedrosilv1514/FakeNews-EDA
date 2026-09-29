"""Preparacao reproduzivel do corpus usado pelos experimentos de ML."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import load_config
from src.preprocessing.clean_text import collapse_whitespace, normalize_for_matching


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LABEL_TO_INT = {"real": 0, "fake": 1}


def sanitize_model_text(value: object) -> str:
    """Normaliza Unicode/espacos e remove apenas controles invalidos.

    A funcao deliberadamente preserva caixa, acentos, pontuacao e stopwords.
    """
    text = unicodedata.normalize("NFKC", collapse_whitespace(value))
    return CONTROL_RE.sub(" ", text).strip()


def compose_news_text(title: object, content: object, subtitle: object = "") -> str:
    """Monta a unidade de inferencia igual para treino e producao."""
    parts = []
    for marker, value in (("TITULO", title), ("SUBTITULO", subtitle), ("TEXTO", content)):
        cleaned = sanitize_model_text(value)
        if cleaned:
            parts.append(f"{marker}: {cleaned}")
    return "\n".join(parts)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _connect_equal_keys(frame: pd.DataFrame, dsu: _DisjointSet, column: str) -> None:
    valid = frame.loc[frame[column].ne(""), ["raw_index", column]]
    for _, group in valid.groupby(column, sort=False):
        indices = group["raw_index"].astype(int).tolist()
        for index in indices[1:]:
            dsu.union(indices[0], index)


def _add_near_duplicate_edges(
    frame: pd.DataFrame, dsu: _DisjointSet, candidates_path: Path
) -> int:
    if not candidates_path.exists():
        return 0
    candidates = pd.read_csv(candidates_path)
    candidates = candidates[
        (candidates["dataset"] == "fakerecogna/original")
        & (candidates["similarity"] >= 0.96)
    ]
    available = set(frame["raw_index"].astype(int))
    used = 0
    for row in candidates.itertuples(index=False):
        left, right = int(row.left_index), int(row.right_index)
        if left in available and right in available:
            dsu.union(left, right)
            used += 1
    return used


def _assign_grouped_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    """Cria holdouts ~1/7 cada, sem compartilhar grupos."""
    X = frame[["text"]]
    y = frame["label"].to_numpy()
    groups = frame["group_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    remaining_pos, test_pos = next(outer.split(X, y, groups))

    remaining = frame.iloc[remaining_pos]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
    train_local, validation_local = next(
        inner.split(
            remaining[["text"]],
            remaining["label"].to_numpy(),
            remaining["group_id"].to_numpy(),
        )
    )
    split = pd.Series("", index=frame.index, dtype="string")
    split.iloc[test_pos] = "test"
    split.iloc[remaining_pos[train_local]] = "train"
    split.iloc[remaining_pos[validation_local]] = "validation"
    if split.eq("").any():
        raise RuntimeError("Falha ao atribuir todos os registros a um split.")
    return split


def _assign_temporal_roles(frame: pd.DataFrame) -> tuple[pd.Series, str | None]:
    roles = pd.Series("unavailable", index=frame.index, dtype="string")
    dated = frame[frame["date"].notna()].copy()
    if dated.empty:
        return roles, None
    cutoff = dated["date"].quantile(0.80)
    test_groups = set(dated.loc[dated["date"] > cutoff, "group_id"])
    train_groups = set(dated.loc[dated["date"] <= cutoff, "group_id"]) - test_groups
    roles.loc[frame["group_id"].isin(train_groups) & frame["date"].notna()] = "train"
    roles.loc[frame["group_id"].isin(test_groups) & frame["date"].notna()] = "test"
    return roles, pd.Timestamp(cutoff).date().isoformat()


def prepare_modeling_dataset(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Materializa o corpus binario e retorna dados mais auditoria."""
    cfg = load_config()
    source = Path(input_path or cfg["data"]["interim"] / "fakerecogna_original.parquet")
    destination = Path(output_path or cfg["data"]["processed"] / "modeling_dataset.parquet")
    raw = pd.read_parquet(source).copy()
    initial_rows = len(raw)
    raw["raw_index"] = raw.index.astype(int)
    raw = raw[raw["label_semantic"].isin(LABEL_TO_INT)].copy()
    non_binary_rows = initial_rows - len(raw)
    raw["text"] = [
        compose_news_text(title, content, subtitle)
        for title, content, subtitle in zip(raw["title"], raw["text"], raw["subtitle"])
    ]
    raw["label_name"] = raw["label_semantic"].astype("string")
    raw["label"] = raw["label_name"].map(LABEL_TO_INT).astype(int)
    raw["url_key"] = raw["url"].map(normalize_for_matching)
    raw["text_key"] = raw["text"].map(normalize_for_matching).map(_digest)
    raw["body_key"] = raw["Noticia"].map(normalize_for_matching).map(_digest)
    raw["date"] = pd.to_datetime(raw["date_parsed"], errors="coerce")

    valid = raw["text"].str.len().ge(50) & raw["text"].str.contains(r"\w", regex=True)
    invalid_rows = int((~valid).sum())
    raw = raw[valid].copy()

    dsu = _DisjointSet(raw["raw_index"].astype(int).tolist())
    _connect_equal_keys(raw, dsu, "url_key")
    _connect_equal_keys(raw, dsu, "text_key")
    _connect_equal_keys(raw, dsu, "body_key")
    near_path = cfg["reports"]["tables"] / "near_duplicates.csv"
    near_edges = _add_near_duplicate_edges(raw, dsu, near_path)
    raw["component"] = raw["raw_index"].map(lambda value: dsu.find(int(value)))
    raw["group_id"] = raw["component"].map(lambda value: f"grp-{_digest(str(value))[:16]}")

    conflicts = raw.groupby("group_id")["label"].nunique()
    conflict_groups = set(conflicts[conflicts > 1].index)
    conflict_rows = int(raw["group_id"].isin(conflict_groups).sum())
    raw = raw[~raw["group_id"].isin(conflict_groups)].copy()

    # Registros textualmente identicos sao uma unica observacao; URL compartilhada
    # ou quase duplicata permanece, mas sempre no mesmo grupo/split.
    before_exact = len(raw)
    raw = raw.drop_duplicates(subset=["text_key", "label"], keep="first").copy()
    exact_removed = before_exact - len(raw)
    raw = raw.reset_index(drop=True)
    raw["split"] = _assign_grouped_splits(raw, int(cfg["seed"]))
    raw["temporal_split"], temporal_cutoff = _assign_temporal_roles(raw)
    raw["record_id"] = raw.apply(
        lambda row: f"fr-{_digest(str(row.raw_index) + row.text_key)[:20]}", axis=1
    )

    columns = [
        "record_id", "raw_index", "text", "label", "label_name", "group_id",
        "split", "temporal_split", "date", "url", "domain", "category",
    ]
    result = raw[columns].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)

    split_counts = (
        result.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict("index")
    )
    def portable_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(cfg["project_root"].resolve()))
        except ValueError:
            return str(path)

    audit: dict[str, object] = {
        "source": portable_path(source),
        "output": portable_path(destination),
        "initial_rows": initial_rows,
        "missing_or_non_binary_labels_removed": non_binary_rows,
        "invalid_text_rows_removed": invalid_rows,
        "exact_duplicate_rows_removed": exact_removed,
        "near_duplicate_edges_used": near_edges,
        "conflicting_group_rows_removed": conflict_rows,
        "final_rows": len(result),
        "groups": int(result["group_id"].nunique()),
        "split_counts": split_counts,
        "temporal_cutoff": temporal_cutoff,
        "temporal_counts": result["temporal_split"].value_counts().to_dict(),
        "normalization": "NFKC + whitespace + invalid control removal; accents, case, punctuation and stopwords retained",
        "source_body_note": "The distributed FakeRecogna body already appears linguistically preprocessed; no additional stemming/lemmatization is applied.",
    }
    audit_path = cfg["reports"]["tables"] / "modeling_data_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result, audit


def main() -> None:
    data, audit = prepare_modeling_dataset()
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    print(data.groupby(["split", "label_name"]).size())


if __name__ == "__main__":
    main()
