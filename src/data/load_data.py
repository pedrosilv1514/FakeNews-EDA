"""Carregadores read-only para os datasets brutos.

Os carregadores nao escrevem em ``data/raw`` e nao pressupõem schema unico.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import load_config


FACTCHECKS_CANONICAL_FILES = (
    "fakebr.tsv",
    "FakeRecogna.tsv",
    "central_de_fatos.tsv",
    "fact_check_tweet_pt.tsv",
    "FakeNewsSet.tsv",
)

FAKERECOGNA_COLUMNS = {
    "Titulo", "Subtitulo", "Noticia", "Categoria", "Data", "Autor", "URL", "Classe"
}
FACTCHECKS_TEXT_COLUMNS = {"review_id", "review_text", "review_url", "is_fake"}
FACTCHECKS_PAIR_COLUMNS = {"review_id", "review_url", "claim_ids", "is_fake"}
FAKEBR_COLUMNS = {"claim_text", "claim_url", "claim_date", "is_fake"}


def _require_non_empty(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"O dataset {name!r} esta vazio.")
    return frame


def validate_columns(
    frame: pd.DataFrame,
    expected: set[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Falha explicitamente quando uma fonte muda o schema esperado."""
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"Colunas ausentes em {dataset_name}: {missing}")
    return frame


def load_delimited(path: Path | str, *, separator: str) -> pd.DataFrame:
    """Carrega CSV/TSV preservando strings e valores ausentes."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Arquivo bruto nao encontrado: {source}")
    frame = pd.read_csv(
        source,
        sep=separator,
        dtype="string",
        keep_default_na=True,
        encoding="utf-8-sig",
    )
    return _require_non_empty(frame, source.name)


def load_fakerecogna(
    path: Path | str | None = None,
    *,
    use_huggingface: bool = True,
) -> pd.DataFrame:
    """Carrega a copia local do FakeRecogna, preferindo ``datasets``.

    A biblioteca Hugging Face e usada sobre o CSV local, sem baixar nem alterar
    o dado bruto. Se indisponivel, o fallback explicito e o pandas.
    """
    cfg = load_config()
    source = Path(path or cfg["data"]["fakerecogna"])
    if not source.is_file():
        raise FileNotFoundError(f"FakeRecogna nao encontrado: {source}")

    if use_huggingface:
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "csv",
                data_files={"train": str(source)},
                split="train",
                cache_dir=str(cfg["project_root"] / ".cache" / "huggingface" / "datasets"),
            )
            frame = _require_non_empty(dataset.to_pandas(), "FakeRecogna")
            return validate_columns(frame, FAKERECOGNA_COLUMNS, "FakeRecogna")
        except ImportError:
            pass
    frame = load_delimited(source, separator=",")
    return validate_columns(frame, FAKERECOGNA_COLUMNS, "FakeRecogna")


def discover_factchecks_files(directory: Path | str | None = None) -> list[Path]:
    """Lista os TSVs do release sem assumir que todos tem o mesmo schema."""
    cfg = load_config()
    base = Path(directory or cfg["data"]["factchecks_release"])
    if not base.is_dir():
        raise FileNotFoundError(f"Release FactChecks.br nao encontrado: {base}")
    return sorted(base.glob("*.tsv"))


def load_factchecks(
    directory: Path | str | None = None,
    *,
    include_aliases: bool = False,
) -> dict[str, pd.DataFrame]:
    """Carrega separadamente cada benchmark do release FactChecks.br.

    ``fake_br.tsv`` e ``fakebr.tsv`` sao aliases byte-a-byte no release v0.1;
    por padrao apenas ``fakebr`` e exposto para nao contar o mesmo corpus duas vezes.
    """
    files = discover_factchecks_files(directory)
    selected: Iterable[Path] = files
    if not include_aliases:
        selected = [path for path in files if path.name != "fake_br.tsv"]
    datasets: dict[str, pd.DataFrame] = {}
    for path in selected:
        frame = load_delimited(path, separator="\t")
        if path.stem in {"fakebr", "fake_br"}:
            expected = FAKEBR_COLUMNS
        elif path.stem in {"FakeNewsSet", "fact_check_tweet_pt"}:
            expected = FACTCHECKS_PAIR_COLUMNS
        else:
            expected = FACTCHECKS_TEXT_COLUMNS
        datasets[path.stem] = validate_columns(frame, expected, path.stem)
    return datasets


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Calcula SHA-256 em streaming para rastreabilidade do dado bruto."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_file_inventory() -> pd.DataFrame:
    """Inventaria arquivos de dados brutos com tamanho e checksum."""
    cfg = load_config()
    roots = {
        Path(cfg["data"]["factchecks_repo"]): "https://github.com/fake-news-UFG/FactChecks.br",
        Path(cfg["data"]["factchecks_release"]).parents[1]: "https://github.com/fake-news-UFG/FactChecks.br/releases/tag/v0.1",
        Path(cfg["data"]["fakerecogna"]).parent: "https://huggingface.co/datasets/recogna-nlp/FakeRecogna",
    }
    rows: list[dict[str, object]] = []
    for root, source_url in roots.items():
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and ".git" not in item.parts
        ):
            rows.append(
                {
                    "path": str(path.relative_to(cfg["project_root"])),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source_url": source_url,
                }
            )
    return pd.DataFrame(rows)
