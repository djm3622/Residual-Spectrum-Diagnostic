"""Neural-operator temporal training helper functions."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def normalize_method_name(method: str) -> str:
    return str(method).strip().lower().replace("-", "_")


def _resolve_temporal_steps(temporal: Mapping[str, Any], default_steps: int = 20) -> int:
    """Resolve temporal window length using the standard input_steps key."""
    resolved = temporal.get("input_steps", default_steps)
    try:
        resolved = int(resolved)
    except Exception as exc:
        raise ValueError(f"temporal.input_steps must be an integer, got {resolved!r}.") from exc
    return max(2, int(resolved))


def _merge_operator_section(
    merged: Dict[str, Any],
    section: Any,
) -> None:
    """Merge section and optional selected profile."""
    if not isinstance(section, Mapping):
        return

    for key, value in section.items():
        if key in {"profiles", "profile"}:
            continue
        merged[key] = value

    profile_name = section.get("profile")
    profiles = section.get("profiles")
    if isinstance(profile_name, str) and isinstance(profiles, Mapping):
        selected = profiles.get(profile_name)
        if isinstance(selected, Mapping):
            merged.update(selected)


def _resolve_temporal_section(
    normalized_method: str,
    operator_config: Mapping[str, Any] | None,
    baseline_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if normalized_method in {
        "tfno",
        "itfno",
        "uno",
        "wno",
        "rno",
        "neuralop_tfno",
        "neuralop_itfno",
        "neuralop_uno",
        "neuralop_wno",
        "neuralop_rno",
        "operator_tfno",
        "operator_itfno",
        "operator_uno",
        "operator_wno",
        "operator_rno",
    }:
        if not isinstance(operator_config, Mapping):
            return {}
        specific_key = "tfno"
        if "itfno" in normalized_method:
            specific_key = "itfno"
        elif "uno" in normalized_method:
            specific_key = "uno"
        elif "wno" in normalized_method:
            specific_key = "wno"
        elif "rno" in normalized_method:
            specific_key = "rno"
        merged: Dict[str, Any] = {}
        _merge_operator_section(merged, operator_config.get("common", {}))
        specific = operator_config.get(specific_key, {})
        if specific_key == "itfno" and not isinstance(specific, Mapping):
            specific = operator_config.get("tfno", {})
        _merge_operator_section(merged, specific)
        temporal = merged.get("temporal", {})
        return dict(temporal) if isinstance(temporal, Mapping) else {}

    if normalized_method in {"swin", "swin_transformer", "swin_t", "attn_unet", "attention_unet", "unet_attn"}:
        if not isinstance(baseline_config, Mapping):
            return {}
        merged: Dict[str, Any] = {}
        common = baseline_config.get("common", {})
        if isinstance(common, Mapping):
            merged.update(common)
        aliases = {normalized_method}
        if normalized_method in {"swin", "swin_transformer", "swin_t"}:
            aliases.update({"swin", "swin_transformer", "swin_t"})
        else:
            aliases.update({"attn_unet", "attention_unet", "unet_attn"})
        for alias in aliases:
            section = baseline_config.get(alias, {})
            if isinstance(section, Mapping):
                merged.update(section)
        temporal = merged.get("temporal", {})
        return dict(temporal) if isinstance(temporal, Mapping) else {}
    return {}


def resolve_temporal_training_config(
    method: str,
    operator_config: Mapping[str, Any] | None,
    baseline_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_method = normalize_method_name(method)
    temporal = _resolve_temporal_section(normalized_method, operator_config, baseline_config)
    supported_block_methods = {
        "tfno",
        "itfno",
        "uno",
        "wno",
        "rno",
        "neuralop_tfno",
        "neuralop_itfno",
        "neuralop_uno",
        "neuralop_wno",
        "neuralop_rno",
        "operator_tfno",
        "operator_itfno",
        "operator_uno",
        "operator_wno",
        "operator_rno",
        "swin",
        "swin_transformer",
        "swin_t",
        "attn_unet",
        "attention_unet",
        "unet_attn",
    }
    default_enabled = normalized_method in supported_block_methods
    if normalized_method in {"rno", "neuralop_rno", "operator_rno"}:
        # RNO uses recurrent block training/eval path rather than temporal window pairing.
        default_enabled = False
    enabled = bool(temporal.get("enabled", default_enabled))
    input_steps = _resolve_temporal_steps(temporal, default_steps=20) if enabled else 1
    output_steps = max(2, int(temporal.get("output_steps", input_steps))) if enabled else 1
    if enabled and output_steps != input_steps:
        raise ValueError(
            "Current temporal training path requires output_steps == input_steps "
            f"(got input_steps={input_steps}, output_steps={output_steps})."
        )
    target_mode = str(temporal.get("target_mode", "next_block")).strip().lower().replace("-", "_")
    if target_mode not in {"shifted", "next_block"}:
        target_mode = "next_block"
    return {
        "enabled": enabled,
        "input_steps": input_steps,
        "output_steps": output_steps,
        "target_mode": target_mode,
    }


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
