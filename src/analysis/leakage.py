"""Indicadores de associacao entre metadados e label."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metadata_label_association(
    frame: pd.DataFrame,
    column: str,
    dataset_name: str,
    *,
    min_count: int = 2,
) -> pd.DataFrame:
    """Calcula pureza por valor: proporcao pertencente a classe majoritaria."""
    if column not in frame:
        return pd.DataFrame()
    data = frame[[column, "label_semantic"]].copy()
    data[column] = data[column].astype("string").str.strip().replace("", pd.NA)
    data = data.dropna()
    counts = data.groupby([column, "label_semantic"]).size().rename("count").reset_index()
    totals = counts.groupby(column)["count"].sum().rename("total")
    dominant = counts.loc[counts.groupby(column)["count"].idxmax()].set_index(column)
    result = dominant.join(totals)
    result = result[result["total"] >= min_count].reset_index()
    result["purity"] = result["count"] / result["total"]
    result.insert(0, "dataset", dataset_name)
    result.insert(1, "field", column)
    return result.rename(columns={column: "value", "label_semantic": "dominant_label"})


def weighted_purity(association: pd.DataFrame) -> float:
    """Media ponderada da pureza dos grupos, entre acaso e 1."""
    if association.empty or association["total"].sum() == 0:
        return float("nan")
    return float(np.average(association["purity"], weights=association["total"]))


def leakage_risk_table(
    datasets: dict[str, pd.DataFrame],
    duplicate_cross: pd.DataFrame,
) -> pd.DataFrame:
    """Gera registro de riscos com evidencia quantitativa e recomendacao."""
    rows: list[dict[str, str]] = []
    for name, frame in datasets.items():
        for field in ("domain", "author", "category"):
            association = metadata_label_association(frame, field, name, min_count=5)
            if association.empty:
                continue
            pure_share = association.loc[association["purity"].eq(1), "total"].sum() / len(frame)
            severity = "alta" if pure_share >= 0.5 else "média" if pure_share >= 0.15 else "baixa"
            rows.append(
                {
                    "risk": f"Associacao label-{field} em {name}",
                    "evidence": f"{pure_share:.1%} dos registros estao em valores com >=5 casos e uma unica classe.",
                    "severity": severity,
                    "recommendation": f"Auditar e considerar group split por {field}; comparar baseline com e sem o campo.",
                }
            )
        dated = frame.dropna(subset=["date_parsed", "label_semantic"])
        if not dated.empty:
            ranges = dated.groupby("label_semantic")["date_parsed"].agg(["min", "max"])
            if len(ranges) > 1:
                starts = (ranges["min"].max() - ranges["min"].min()).days
                severity = "alta" if abs(starts) > 730 else "média" if abs(starts) > 180 else "baixa"
                rows.append(
                    {
                        "risk": f"Separacao temporal entre classes em {name}",
                        "evidence": f"Inicio das classes difere em ate {abs(starts)} dias; consultar tabela de periodos.",
                        "severity": severity,
                        "recommendation": "Testar split temporal e medir desempenho por janela de tempo.",
                    }
                )

    if not duplicate_cross.empty:
        conflicts = int(duplicate_cross["label_conflict"].sum())
        severity = "alta" if len(duplicate_cross) >= 100 else "média" if len(duplicate_cross) else "baixa"
        rows.append(
            {
                "risk": "Sobreposicao entre datasets",
                "evidence": f"{len(duplicate_cross)} assinaturas cruzadas; {conflicts} apresentam conflito semantico de label.",
                "severity": severity,
                "recommendation": "Agrupar por URL/texto antes do split e revisar conflitos; nunca sortear linhas isoladamente.",
            }
        )
    return pd.DataFrame(rows)

