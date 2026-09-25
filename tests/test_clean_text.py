import pandas as pd

from src.preprocessing.clean_text import collapse_whitespace, extract_domain, normalize_for_matching


def test_collapse_whitespace_preserves_content() -> None:
    assert collapse_whitespace("  Olá\n  Brasil! ") == "Olá Brasil!"


def test_normalize_for_matching_removes_accents_and_punctuation() -> None:
    assert normalize_for_matching("Notícia: FALSA!") == "noticia falsa"


def test_extract_domain_canonicalizes_www() -> None:
    assert extract_domain("https://www.Example.com/noticia") == "example.com"


def test_missing_text_becomes_empty() -> None:
    assert collapse_whitespace(pd.NA) == ""

