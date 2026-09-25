import pandas as pd

from src.data.make_dataset import standardize_factchecks, standardize_fakerecogna


def test_standardize_factchecks_keeps_raw_columns() -> None:
    source = pd.DataFrame(
        {
            "claim_text": ["Texto"],
            "claim_author": ["Autor"],
            "claim_url": ["https://example.com/a"],
            "claim_date": ["2020-01-01"],
            "category": ["politica"],
            "is_fake": ["1"],
        }
    )
    result = standardize_factchecks(source, "fakebr")
    assert set(source.columns).issubset(result.columns)
    assert result.loc[0, "raw_label"] == "1"
    assert result.loc[0, "label_semantic"] == "fake"


def test_standardize_fakerecogna_maps_documented_labels() -> None:
    source = pd.DataFrame(
        {
            "Titulo": ["A"],
            "Subtitulo": ["B"],
            "Noticia": ["C"],
            "Categoria": ["saude"],
            "Data": ["01/01/2020"],
            "Autor": ["D"],
            "URL": ["https://example.com"],
            "Classe": ["0"],
        }
    )
    result = standardize_fakerecogna(source)
    assert result.loc[0, "label_semantic"] == "fake"
    assert set(source.columns).issubset(result.columns)

