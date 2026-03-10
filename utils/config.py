"""Configuration helpers for YAML-based experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    """Raised when a YAML configuration is missing required values."""


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError(f"Config at {path} must be a YAML mapping.")

    return config


def require_keys(mapping: Dict[str, Any], section_name: str, keys: list[str]) -> None:
    """Ensure keys are present in a config section."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        missing_txt = ", ".join(missing)
        raise ConfigError(f"Missing keys in '{section_name}': {missing_txt}")
