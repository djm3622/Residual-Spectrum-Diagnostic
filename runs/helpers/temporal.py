"""Neural-operator temporal training helper functions."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def normalize_method_name(method: str) -> str:
    return str(method).strip().lower().replace("-", "_")


def resolve_temporal_training_config(method: str, operator_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    normalized_method = normalize_method_name(method)
    if normalized_method not in {
        "fno",
        "tfno",
        "uno",
        "neuralop_fno",
        "neuralop_tfno",
        "neuralop_uno",
        "operator_fno",
        "operator_tfno",
        "operator_uno",
    }:
        return {"enabled": False, "input_steps": 1, "target_mode": "shifted"}

    if not isinstance(operator_config, Mapping):
        return {"enabled": False, "input_steps": 1, "target_mode": "shifted"}

    common = operator_config.get("common", {})
    specific_key = "fno"
    if "tfno" in normalized_method:
        specific_key = "tfno"
    elif "uno" in normalized_method:
        specific_key = "uno"

    merged: Dict[str, Any] = {}
    if isinstance(common, Mapping):
        merged.update(common)
    specific = operator_config.get(specific_key, {})
    if isinstance(specific, Mapping):
        merged.update(specific)

    temporal = merged.get("temporal", {})
    if not isinstance(temporal, Mapping):
        return {"enabled": False, "input_steps": 1, "target_mode": "shifted"}

    enabled = bool(temporal.get("enabled", False))
    input_steps = max(2, int(temporal.get("input_steps", 20))) if enabled else 1
    target_mode = str(temporal.get("target_mode", "shifted")).strip().lower().replace("-", "_")
    if target_mode not in {"shifted", "next_block"}:
        target_mode = "shifted"
    return {"enabled": enabled, "input_steps": input_steps, "target_mode": target_mode}


def window_start_indices(n_steps: int, window: int, target_mode: str) -> range:
    if target_mode == "next_block":
        max_start = n_steps - (2 * window)
    else:
        max_start = n_steps - window - 1
    if max_start < 0:
        return range(0, 0)
    return range(0, max_start + 1)


def window_target_start(start: int, window: int, target_mode: str) -> int:
    if target_mode == "next_block":
        return start + window
    return start + 1
