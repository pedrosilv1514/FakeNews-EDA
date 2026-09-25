"""Normalizacao explicita de labels e datas, preservando os valores originais."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import re

import pandas as pd


LABEL_MAPPINGS: dict[str, Mapping[str, str]] = {
    # FactChecks.br: 1 significa fake, -1 significa not fake, 0 significa other.
    "factchecks": {"1": "fake", "-1": "not_fake", "0": "other"},
    # FakeRecogna original: 0 significa fake e 1 significa real.
    "fakerecogna": {"0": "fake", "1": "real"},
}


def normalize_labels(values: pd.Series, dataset_family: str) -> pd.Series:
    """Mapeia labels com semantica documentada, sem sobrescrever ``raw_label``."""
    if dataset_family not in LABEL_MAPPINGS:
        raise KeyError(f"Mapeamento de labels desconhecido: {dataset_family}")
    raw = values.astype("string").str.strip()
    # O parser CSV do Hugging Face pode materializar labels inteiros como floats.
    # A representacao original segue em ``raw_label``; apenas a chave de lookup muda.
    lookup = raw.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    normalized = lookup.map(LABEL_MAPPINGS[dataset_family])
    unknown = sorted(raw[raw.notna() & normalized.isna()].unique().tolist())
    if unknown:
        raise ValueError(f"Labels nao mapeados em {dataset_family}: {unknown}")
    return normalized.astype("string")


def parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Extrai a primeira data valida de formatos heterogeneos.

    O valor bruto nunca e substituido. Textos editoriais (``Publicado em``), meses
    em portugues e sufixos acidentais sao tolerados; anos fora de 1900--ano atual
    viram ``NaT`` e permanecem auditaveis em ``date_raw``.
    """
    months = {
        "janeiro": "01",
        "fevereiro": "02",
        "março": "03",
        "marco": "03",
        "abril": "04",
        "maio": "05",
        "junho": "06",
        "julho": "07",
        "agosto": "08",
        "setembro": "09",
        "outubro": "10",
        "novembro": "11",
        "dezembro": "12",
    }

    def extract(value: object) -> object:
        if value is None or pd.isna(value):
            return pd.NA
        text = str(value).strip().casefold()
        if not text:
            return pd.NA
        for name, number in months.items():
            text = re.sub(rf"\b{name}\b", number, text)
        text = re.sub(r"\s+de\s+", "/", text)
        iso = re.search(r"(?<!\d)(\d{4}-\d{1,2}-\d{1,2})(?!\d)", text)
        if iso:
            return iso.group(1)
        dmy = re.search(r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
        return dmy.group(1) if dmy else pd.NA

    extracted = values.map(extract).astype("string")
    parsed = pd.to_datetime(extracted, errors="coerce", dayfirst=True, format="mixed")
    valid_year = parsed.dt.year.between(1900, datetime.now().year)
    return parsed.where(valid_year)
