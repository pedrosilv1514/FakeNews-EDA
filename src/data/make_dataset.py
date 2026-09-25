"""Cria copias intermediarias padronizadas, sem combinar benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ensure_output_directories, load_config
from src.data.load_data import load_factchecks, load_fakerecogna
from src.preprocessing.clean_text import collapse_whitespace, extract_domain
from src.preprocessing.normalize import normalize_labels, parse_mixed_dates


def _empty_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def standardize_factchecks(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Adapta um benchmark FactChecks.br ao schema analitico, preservando colunas."""
    result = frame.copy()
    result.insert(0, "dataset", f"factchecks/{dataset_name}")
    result["raw_label"] = result["is_fake"].astype("string")
    result["label_semantic"] = normalize_labels(result["raw_label"], "factchecks")

    result["text"] = result.get("claim_text", result.get("review_text", _empty_series(result)))
    result["title"] = _empty_series(result)
    result["author"] = result.get("claim_author", result.get("review_author", _empty_series(result)))
    result["url"] = result.get("claim_url", result.get("review_url", _empty_series(result)))
    result["date_raw"] = result.get("claim_date", result.get("review_date", _empty_series(result)))
    result["date_parsed"] = parse_mixed_dates(result["date_raw"])
    result["domain"] = result.get("review_domain", result["url"].map(extract_domain))
    for column in ("text", "title", "author", "url", "domain"):
        result[column] = result[column].map(collapse_whitespace).astype("string")
    return result


def standardize_fakerecogna(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapta o FakeRecogna original sem descartar os oito campos de origem."""
    result = frame.copy()
    result.insert(0, "dataset", "fakerecogna/original")
    result["raw_label"] = result["Classe"].astype("string")
    valid_mask = result["raw_label"].notna()
    result["label_semantic"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[valid_mask, "label_semantic"] = normalize_labels(
        result.loc[valid_mask, "raw_label"], "fakerecogna"
    )
    result["title"] = result["Titulo"].map(collapse_whitespace).astype("string")
    result["subtitle"] = result["Subtitulo"].map(collapse_whitespace).astype("string")
    result["text"] = result["Noticia"].map(collapse_whitespace).astype("string")
    result["author"] = result["Autor"].map(collapse_whitespace).astype("string")
    result["url"] = result["URL"].map(collapse_whitespace).astype("string")
    result["domain"] = result["url"].map(extract_domain).astype("string")
    result["date_raw"] = result["Data"].astype("string")
    result["date_parsed"] = parse_mixed_dates(result["date_raw"])
    result["category"] = result["Categoria"].map(collapse_whitespace).astype("string")
    return result


def build_interim_datasets(output_dir: Path | str | None = None) -> dict[str, Path]:
    """Gera um Parquet por dataset; nao concatena nem remove registros."""
    cfg = load_config()
    ensure_output_directories(cfg)
    destination = Path(output_dir or cfg["data"]["interim"])
    destination.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for name, frame in load_factchecks().items():
        standardized = standardize_factchecks(frame, name)
        path = destination / f"factchecks_{name}.parquet"
        standardized.to_parquet(path, index=False)
        outputs[f"factchecks/{name}"] = path

    fakerecogna = standardize_fakerecogna(load_fakerecogna())
    path = destination / "fakerecogna_original.parquet"
    fakerecogna.to_parquet(path, index=False)
    outputs["fakerecogna/original"] = path
    return outputs


def load_interim_datasets(directory: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Carrega todos os Parquets intermediarios gerados pelo pipeline."""
    cfg = load_config()
    source = Path(directory or cfg["data"]["interim"])
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Nenhum dataset interim. Execute `python -m src.data.make_dataset`.")
    return {path.stem: pd.read_parquet(path) for path in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for name, path in build_interim_datasets(args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

