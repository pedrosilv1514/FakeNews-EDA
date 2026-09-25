from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_delimited, validate_columns


def test_load_delimited_preserves_expected_columns(tmp_path: Path) -> None:
    source = tmp_path / "sample.csv"
    source.write_text("Titulo,Noticia,Classe\nA,Texto,0\n", encoding="utf-8")
    frame = load_delimited(source, separator=",")
    assert frame.columns.tolist() == ["Titulo", "Noticia", "Classe"]
    assert len(frame) == 1


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_text("Titulo,Noticia,Classe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="esta vazio"):
        load_delimited(source, separator=",")


def test_missing_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_delimited(tmp_path / "missing.csv", separator=",")


def test_schema_validation_is_explicit() -> None:
    frame = pd.DataFrame({"Titulo": ["A"], "Noticia": ["B"], "Classe": ["0"]})
    expected = {"Titulo", "Noticia", "Classe"}
    assert validate_columns(frame, expected, "sample") is frame


def test_schema_validation_reports_missing_columns() -> None:
    frame = pd.DataFrame({"Titulo": ["A"]})
    with pytest.raises(ValueError, match="Colunas ausentes"):
        validate_columns(frame, {"Titulo", "Noticia", "Classe"}, "sample")
