import pandas as pd

from src.preprocessing.modeling import compose_news_text, sanitize_model_text


def test_sanitize_preserves_portuguese_semantics() -> None:
    assert sanitize_model_text("  NÃO\x00 é boato!  ") == "NÃO  é boato!"


def test_compose_news_uses_same_public_fields_as_training() -> None:
    text = compose_news_text("Título", "Conteúdo principal", "Linha fina")
    assert text == "TITULO: Título\nSUBTITULO: Linha fina\nTEXTO: Conteúdo principal"


def test_split_groups_are_disjoint_in_materialized_data() -> None:
    from src.preprocessing.modeling import prepare_modeling_dataset

    data, _ = prepare_modeling_dataset()
    memberships = data.groupby("group_id")["split"].nunique()
    assert memberships.max() == 1
    temporal = data[data["temporal_split"].isin(["train", "test"])]
    assert temporal.groupby("group_id")["temporal_split"].nunique().max() == 1
    assert set(data["label_name"]) == {"fake", "real"}
    assert not data["text"].isna().any()
