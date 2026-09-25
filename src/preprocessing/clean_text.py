"""Limpeza minima e assinaturas para comparacao textual."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def collapse_whitespace(value: object) -> str:
    """Remove espacos redundantes sem alterar caixa, acentos ou pontuacao."""
    if value is None or pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def normalize_for_matching(value: object) -> str:
    """Normaliza texto apenas para deteccao de duplicatas, nunca para sobrescrever o original."""
    text = collapse_whitespace(value).casefold()
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    return WHITESPACE_RE.sub(" ", NON_WORD_RE.sub(" ", text)).strip()


def extract_domain(value: object) -> str:
    """Extrai hostname canonico de uma URL, retornando vazio quando ausente."""
    url = collapse_whitespace(value)
    if not url:
        return ""
    candidate = url if "://" in url else f"https://{url}"
    domain = urlparse(candidate).hostname or ""
    return domain.casefold().removeprefix("www.")

