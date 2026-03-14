"""Reaction-diffusion-specific evaluation helpers."""

from __future__ import annotations

import numpy as np


def extract_panel_frames(
    input_u_window: np.ndarray,
    input_v_window: np.ndarray,
    target_u_window: np.ndarray,
    target_v_window: np.ndarray,
    pred_u_window: np.ndarray,
    pred_v_window: np.ndarray,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_mode == "next_block":
        return (
            np.asarray(input_u_window[-1], dtype=np.float32),
            np.asarray(input_v_window[-1], dtype=np.float32),
            np.asarray(target_u_window[0], dtype=np.float32),
            np.asarray(target_v_window[0], dtype=np.float32),
            np.asarray(pred_u_window[0], dtype=np.float32),
            np.asarray(pred_v_window[0], dtype=np.float32),
        )
    return (
        np.asarray(input_u_window[-1], dtype=np.float32),
        np.asarray(input_v_window[-1], dtype=np.float32),
        np.asarray(target_u_window[-1], dtype=np.float32),
        np.asarray(target_v_window[-1], dtype=np.float32),
        np.asarray(pred_u_window[-1], dtype=np.float32),
        np.asarray(pred_v_window[-1], dtype=np.float32),
    )


def extract_target_frame(target_window: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "next_block":
        return np.asarray(target_window[0], dtype=np.float32)
    return np.asarray(target_window[-1], dtype=np.float32)
