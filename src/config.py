"""Carregamento centralizado da configuracao do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Carrega a configuracao YAML e resolve caminhos contra a raiz do projeto."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    for section in ("data", "reports"):
        for key, value in config.get(section, {}).items():
            config[section][key] = PROJECT_ROOT / value
    config["models"] = PROJECT_ROOT / config["models"]
    config["project_root"] = PROJECT_ROOT
    return config


def ensure_output_directories(config: dict[str, Any] | None = None) -> None:
    """Cria apenas diretorios de saida; nunca altera arquivos em ``data/raw``."""
    cfg = config or load_config()
    for path in (
        cfg["data"]["interim"],
        cfg["data"]["processed"],
        cfg["reports"]["figures"],
        cfg["reports"]["tables"],
        cfg["models"],
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

