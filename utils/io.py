"""I/O helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_run_dirs(
    output_root: str | Path,
    checkpoint_root: str | Path,
    problem_name: str,
    method: str,
    seed: int,
) -> Tuple[Path, Path]:
    """Create deterministic output/checkpoint directories for one run."""
    out_dir = ensure_dir(Path(output_root) / problem_name / method / f"seed_{seed}")
    ckpt_dir = ensure_dir(Path(checkpoint_root) / problem_name / method / f"seed_{seed}")
    return out_dir, ckpt_dir


def convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Save a dictionary as pretty JSON, converting numpy objects."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(convert_numpy(data), handle, indent=2)


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save model state payload as NPZ."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    np.savez(file_path, **payload)
