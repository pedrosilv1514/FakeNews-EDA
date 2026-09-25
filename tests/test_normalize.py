import pandas as pd
import pytest

from src.preprocessing.normalize import normalize_labels, parse_mixed_dates


def test_factchecks_label_semantics() -> None:
    result = normalize_labels(pd.Series(["1", "-1", "0"]), "factchecks")
    assert result.tolist() == ["fake", "not_fake", "other"]


def test_fakerecogna_label_semantics() -> None:
    result = normalize_labels(pd.Series(["0", "1", "0.0", "1.0"]), "fakerecogna")
    assert result.tolist() == ["fake", "real", "fake", "real"]


def test_unknown_label_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Labels nao mapeados"):
        normalize_labels(pd.Series(["maybe"]), "fakerecogna")


def test_parse_mixed_dates_handles_portuguese_and_editorial_text() -> None:
    values = pd.Series(["26 de março de 2018", "Publicado em 20/03/21 16:57"])
    result = parse_mixed_dates(values)
    assert result.dt.strftime("%Y-%m-%d").tolist() == ["2018-03-26", "2021-03-20"]


def test_parse_mixed_dates_rejects_implausible_year() -> None:
    assert pd.isna(parse_mixed_dates(pd.Series(["03/09/0201"])).iloc[0])
