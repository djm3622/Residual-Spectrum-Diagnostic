"""Common utility helpers for runner scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import load_checkpoint


def move_model_device(model: Any, device_name: str) -> None:
    """Move the underlying torch module when available (best-effort)."""
    net = getattr(model, "net", None)
    if net is None:
        return
    try:
        net.to(device_name)
    except Exception:
        return


def load_best_checkpoint_for_eval(
    model: Any,
    checkpoint_dir: Path | None,
    model_tag: str,
) -> None:
    """Best-effort restore of the model weights saved at best validation epoch."""
    if checkpoint_dir is None:
        return

    best_path = Path(checkpoint_dir) / f"model_{str(model_tag)}_best.npz"
    if not best_path.exists():
        return

    payload = load_checkpoint(best_path)
    state_dict: dict[str, Any] | None = None

    if isinstance(payload, dict):
        training_state = payload.get("training_state")
        if isinstance(training_state, dict):
            best_state = training_state.get("best_model_state")
            if isinstance(best_state, dict):
                state_dict = best_state
            else:
                model_state = training_state.get("model_state")
                if isinstance(model_state, dict):
                    state_dict = model_state
        if state_dict is None:
            best_state = payload.get("best_model_state")
            if isinstance(best_state, dict):
                state_dict = best_state
            else:
                model_state = payload.get("model_state")
                if isinstance(model_state, dict):
                    state_dict = model_state

    if not isinstance(state_dict, dict):
        return

    net = getattr(model, "net", None)
    if net is not None and hasattr(net, "load_state_dict"):
        try:
            net.load_state_dict(state_dict)
            return
        except Exception:
            pass

    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(state_dict)
        except Exception:
            return
