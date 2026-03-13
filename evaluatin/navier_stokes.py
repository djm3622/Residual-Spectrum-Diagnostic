"""Navier-Stokes-specific evaluation helpers."""

from __future__ import annotations

from typing import List

import numpy as np

from evaluatin.metrics import safe_mean


def block_future_step_indices(
    n_snapshots: int,
    block_size: int,
    max_points: int = 6,
) -> List[int]:
    """Choose evenly interpretable future checkpoints for block-prediction models."""
    n_steps = max(1, int(n_snapshots))
    stride = max(1, int(block_size))
    if n_steps == 1:
        return [0]

    steps = [0]
    step = stride
    while step < n_steps and len(steps) < max_points:
        steps.append(step)
        step += stride
    if len(steps) < max_points and steps[-1] != n_steps - 1:
        steps.append(n_steps - 1)
    return sorted(set(int(np.clip(idx, 0, n_steps - 1)) for idx in steps))


def future_block_rel_l2(
    pred: np.ndarray,
    target: np.ndarray,
    horizon: int,
) -> float:
    """Average relative L2 at fixed future horizons (e.g., every 20 steps)."""
    pred_arr = np.asarray(pred, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    n_steps = min(pred_arr.shape[0], target_arr.shape[0])
    if n_steps <= 1:
        return float("nan")

    stride = max(1, int(horizon))
    indices = list(range(stride, n_steps, stride))
    if not indices:
        indices = [n_steps - 1]

    rel_errors: List[float] = []
    for idx in indices:
        numerator = float(np.linalg.norm(pred_arr[idx] - target_arr[idx]))
        denominator = float(np.linalg.norm(target_arr[idx]) + 1e-12)
        rel_errors.append(numerator / denominator)
    return safe_mean(rel_errors)


def extract_panel_frames(
    input_window: np.ndarray,
    target_window: np.ndarray,
    pred_window: np.ndarray,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del target_mode
    return (
        np.asarray(input_window[-1], dtype=np.float32),
        np.asarray(target_window[-1], dtype=np.float32),
        np.asarray(pred_window[-1], dtype=np.float32),
    )
