"""Navier-Stokes runner helpers for noisy references and supervised pair building."""

from __future__ import annotations

from typing import List

import numpy as np

from data.navier_stokes import NSConfig
from utils.noise import add_hf_noise_2d
from utils.progress import progress_iter

from runs.helpers.temporal import window_start_indices, window_target_start


def noisy_reference_field(
    field: np.ndarray,
    config: NSConfig,
    rng_seed: int,
) -> np.ndarray:
    prev_state = np.random.get_state()
    np.random.seed(int(rng_seed))
    try:
        noisy = add_hf_noise_2d(
            np.asarray(field, dtype=np.float32),
            config.noise_level,
            config.nx,
            config.ny,
            Lx=config.Lx,
            Ly=config.Ly,
        )
    finally:
        np.random.set_state(prev_state)
    return np.asarray(noisy, dtype=np.float32)


def noisy_reference_trajectory(
    trajectory: np.ndarray,
    config: NSConfig,
    rng_seed: int,
) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.float32)
    noisy_traj = np.empty_like(traj, dtype=np.float32)

    prev_state = np.random.get_state()
    np.random.seed(int(rng_seed))
    try:
        for step in range(traj.shape[0]):
            noisy_traj[step] = np.asarray(
                add_hf_noise_2d(
                    traj[step],
                    config.noise_level,
                    config.nx,
                    config.ny,
                    Lx=config.Lx,
                    Ly=config.Ly,
                ),
                dtype=np.float32,
            )
    finally:
        np.random.set_state(prev_state)

    return noisy_traj


def build_supervised_pairs(
    trajectories: List[np.ndarray],
    config: NSConfig,
    temporal_enabled: bool,
    temporal_window: int,
    temporal_target_mode: str,
    *,
    noisy: bool,
    show_progress: bool,
    progress_desc: str,
    return_trajectories: bool = False,
) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray] | None]:
    """Build one-step/windowed train pairs from trajectories (optionally noisy)."""
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    stored_trajectories: List[np.ndarray] | None = [] if return_trajectories else None

    for trajectory in progress_iter(
        trajectories,
        enabled=show_progress,
        desc=progress_desc,
        total=len(trajectories),
    ):
        traj = np.asarray(trajectory, dtype=np.float32)
        if noisy:
            source_traj = np.empty_like(traj, dtype=np.float32)
            for step in range(len(traj)):
                source_traj[step] = np.asarray(
                    add_hf_noise_2d(
                        traj[step],
                        config.noise_level,
                        config.nx,
                        config.ny,
                        Lx=config.Lx,
                        Ly=config.Ly,
                    ),
                    dtype=np.float32,
                )
        else:
            source_traj = traj

        if stored_trajectories is not None:
            stored_trajectories.append(source_traj)

        if temporal_enabled:
            for start in window_start_indices(len(source_traj), temporal_window, temporal_target_mode):
                target_start = window_target_start(start, temporal_window, temporal_target_mode)
                inputs.append(np.asarray(source_traj[start : start + temporal_window], dtype=np.float32))
                targets.append(
                    np.asarray(source_traj[target_start : target_start + temporal_window], dtype=np.float32)
                )
        else:
            for step in range(len(source_traj) - 1):
                inputs.append(source_traj[step])
                targets.append(source_traj[step + 1])

    return inputs, targets, stored_trajectories
