"""Perfis estruturais e de qualidade."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def dataset_overview(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Resume dimensoes, labels, periodo e campos de cada dataset."""
    rows: list[dict[str, object]] = []
    for name, frame in datasets.items():
        dates = frame.get("date_parsed", pd.Series(dtype="datetime64[ns]"))
        valid_dates = dates.dropna()
        labels = frame.get("raw_label", pd.Series(dtype="string")).dropna().unique()
        rows.append(
            {
                "dataset": name,
                "records": len(frame),
                "columns": len(frame.columns),
                "raw_labels": ", ".join(sorted(map(str, labels))),
                "period_start": valid_dates.min().date().isoformat() if not valid_dates.empty else "",
                "period_end": valid_dates.max().date().isoformat() if not valid_dates.empty else "",
                "main_columns": ", ".join(frame.columns[:8]),
            }
        )
    return pd.DataFrame(rows)


def missingness_table(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Conta valores nulos ou strings vazias por coluna."""
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        empty_strings = series.astype("string").str.strip().eq("").fillna(False)
        missing = int(series.isna().sum() + empty_strings.sum())
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "missing_count": missing,
                "missing_percent": round(100 * missing / len(frame), 3),
            }
        )
    return pd.DataFrame(rows)


def label_distribution(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Distribuicao conjunta do label bruto e da interpretacao semantica."""
    counts = (
        frame.groupby(["raw_label", "label_semantic"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts.insert(0, "dataset", dataset_name)
    counts["percent"] = (100 * counts["count"] / len(frame)).round(3)
    return counts


def top_values(
    frame: pd.DataFrame,
    column: str,
    dataset_name: str,
    limit: int = 15,
) -> pd.DataFrame:
    """Lista valores mais frequentes, ignorando ausentes e vazios."""
    if column not in frame:
        return pd.DataFrame(columns=["dataset", "field", "value", "count", "percent"])
    series = frame[column].astype("string").str.strip().replace("", pd.NA).dropna()
    counts = series.value_counts().head(limit)
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "field": column,
            "value": counts.index.astype(str),
            "count": counts.values,
            "percent": (100 * counts.values / len(frame)).round(3),
        }
    )

