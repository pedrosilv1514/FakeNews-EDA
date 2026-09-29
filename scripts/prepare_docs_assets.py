"""Copia figuras versionadas para a arvore isolada de build do MkDocs."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "figures"
DESTINATION = ROOT / "docs" / "assets" / "generated"
FIGURES = (
    "class_distribution.png",
    "model_comparison.png",
    "calibration_comparison.png",
    "E01_diagnostics.png",
    "E02_diagnostics.png",
    "E03_diagnostics.png",
    "E04_diagnostics.png",
)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FIGURES if not (SOURCE / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Figuras ausentes; execute os experimentos antes do build: " + ", ".join(missing)
        )
    for name in FIGURES:
        shutil.copy2(SOURCE / name, DESTINATION / name)
    print(f"{len(FIGURES)} figuras copiadas para {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
