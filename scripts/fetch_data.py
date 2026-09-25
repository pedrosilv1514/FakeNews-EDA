"""Adquire as fontes oficiais somente quando os destinos raw ainda nao existem."""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from src.config import PROJECT_ROOT


FACTCHECKS_GIT = "https://github.com/fake-news-UFG/FactChecks.br.git"
FACTCHECKS_RELEASE = "https://github.com/fake-news-UFG/FactChecks.br/releases/download/v0.1/FactChecksbr.zip"
FACTCHECKS_BUILDER = "https://huggingface.co/datasets/fake-news-UFG/FactChecksbr/resolve/main/FactChecksbr.py"
FAKERECOGNA_CSV = "https://huggingface.co/datasets/recogna-nlp/FakeRecogna/resolve/main/FakeRecogna.csv"
FAKERECOGNA_README = "https://huggingface.co/datasets/recogna-nlp/FakeRecogna/resolve/main/README.md"


def _require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Destino raw ja existe e nao sera sobrescrito: {path}. "
            "Remova-o conscientemente fora deste script apenas se quiser uma nova aquisicao."
        )


def fetch() -> None:
    """Baixa fontes oficiais em destinos novos, sem sobrescrever arquivos."""
    raw = PROJECT_ROOT / "data" / "raw"
    repository = raw / "factchecks"
    factchecks_hf = raw / "factchecks_hf"
    fakerecogna = raw / "fakerecogna"
    for path in (repository, factchecks_hf, fakerecogna):
        _require_absent(path)

    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", FACTCHECKS_GIT, str(repository)],
        check=True,
    )
    factchecks_hf.mkdir()
    fakerecogna.mkdir()
    urllib.request.urlretrieve(FACTCHECKS_RELEASE, factchecks_hf / "FactChecksbr.zip")
    urllib.request.urlretrieve(FACTCHECKS_BUILDER, factchecks_hf / "FactChecksbr.py")
    urllib.request.urlretrieve(FAKERECOGNA_CSV, fakerecogna / "FakeRecogna.csv")
    urllib.request.urlretrieve(FAKERECOGNA_README, fakerecogna / "README.md")
    with zipfile.ZipFile(factchecks_hf / "FactChecksbr.zip") as archive:
        archive.extractall(factchecks_hf / "release_v0.1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-network-download", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_network_download:
        parser.error("use --acknowledge-network-download para confirmar a aquisicao")
    fetch()


if __name__ == "__main__":
    main()

