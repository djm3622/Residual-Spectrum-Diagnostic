"""Reaction-diffusion runner helpers for pair building and noisy references."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from data.reaction_diffusion import GrayScottConfig, GrayScottSolver
from utils.noise import add_hf_noise_coupled
from utils.progress import progress_iter

from runs.helpers.temporal import window_start_indices, window_target_start


def sample_initial_condition(solver: GrayScottSolver, config: GrayScottConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw one configured initial condition for Gray-Scott trajectories."""
    mode = str(config.initial_condition).strip().lower().replace("-", "_")

    if mode in {"random_seeds", "random", "seeds"}:
        return solver.initial_condition_random_seeds(
            n_seeds=max(1, int(config.initial_n_seeds)),
            seed=seed,
        )
    if mode in {"center_square", "square", "center"}:
        return solver.initial_condition_center_square(
            size=max(2, int(config.initial_square_size)),
            noise_amplitude=max(0.0, float(config.initial_noise_amplitude)),
            seed=seed,
        )

    raise ValueError(
        f"Unsupported data.initial_condition '{config.initial_condition}'. "
        "Use one of: random_seeds, center_square."
    )


def noisy_reference_frame_coupled(
    u_field: np.ndarray,
    v_field: np.ndarray,
    config: GrayScottConfig,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    prev_state = np.random.get_state()
    np.random.seed(int(rng_seed))
    try:
        u_noisy, v_noisy = add_hf_noise_coupled(
            np.asarray(u_field, dtype=np.float32),
            np.asarray(v_field, dtype=np.float32),
            config.noise_level,
            config.nx,
            config.ny,
            Lx=config.Lx,
            Ly=config.Ly,
        )
    finally:
        np.random.set_state(prev_state)
    return np.asarray(u_noisy, dtype=np.float32), np.asarray(v_noisy, dtype=np.float32)


def noisy_reference_trajectory_coupled(
    u_traj: np.ndarray,
    v_traj: np.ndarray,
    config: GrayScottConfig,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    u_arr = np.asarray(u_traj, dtype=np.float32)
    v_arr = np.asarray(v_traj, dtype=np.float32)
    u_noisy = np.empty_like(u_arr, dtype=np.float32)
    v_noisy = np.empty_like(v_arr, dtype=np.float32)

    prev_state = np.random.get_state()
    np.random.seed(int(rng_seed))
    try:
        for step in range(u_arr.shape[0]):
            u_step_noisy, v_step_noisy = add_hf_noise_coupled(
                u_arr[step],
                v_arr[step],
                config.noise_level,
                config.nx,
                config.ny,
                Lx=config.Lx,
                Ly=config.Ly,
            )
            u_noisy[step] = np.asarray(u_step_noisy, dtype=np.float32)
            v_noisy[step] = np.asarray(v_step_noisy, dtype=np.float32)
    finally:
        np.random.set_state(prev_state)

    return u_noisy, v_noisy


def build_supervised_pairs_coupled(
    trajectories: List[Dict[str, np.ndarray]],
    config: GrayScottConfig,
    temporal_enabled: bool,
    temporal_window: int,
    temporal_target_mode: str,
    *,
    noisy: bool,
    show_progress: bool,
    progress_desc: str,
    return_pair_steps: bool = False,
    return_trajectories: bool = False,
) -> tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[int] | None,
    List[np.ndarray] | None,
    List[np.ndarray] | None,
]:
    """Build coupled train/val pairs from trajectories (optionally noisy)."""
    inputs_u: List[np.ndarray] = []
    inputs_v: List[np.ndarray] = []
    targets_u: List[np.ndarray] = []
    targets_v: List[np.ndarray] = []
    pair_steps: List[int] | None = [] if return_pair_steps else None
    trajectory_u: List[np.ndarray] | None = [] if return_trajectories else None
    trajectory_v: List[np.ndarray] | None = [] if return_trajectories else None

    for data in progress_iter(
        trajectories,
        enabled=show_progress,
        desc=progress_desc,
        total=len(trajectories),
    ):
        u_traj = np.asarray(data["u"], dtype=np.float32)
        v_traj = np.asarray(data["v"], dtype=np.float32)

        if noisy:
            source_u = np.empty_like(u_traj, dtype=np.float32)
            source_v = np.empty_like(v_traj, dtype=np.float32)
            for step in range(len(u_traj)):
                u_noisy_step, v_noisy_step = add_hf_noise_coupled(
                    u_traj[step],
                    v_traj[step],
                    config.noise_level,
                    config.nx,
                    config.ny,
                    Lx=config.Lx,
                    Ly=config.Ly,
                )
                source_u[step] = np.asarray(u_noisy_step, dtype=np.float32)
                source_v[step] = np.asarray(v_noisy_step, dtype=np.float32)
        else:
            source_u = u_traj
            source_v = v_traj

        if trajectory_u is not None and trajectory_v is not None:
            trajectory_u.append(source_u)
            trajectory_v.append(source_v)

        if temporal_enabled:
            for start in window_start_indices(len(source_u), temporal_window, temporal_target_mode):
                target_start = window_target_start(start, temporal_window, temporal_target_mode)
                inputs_u.append(np.asarray(source_u[start : start + temporal_window], dtype=np.float32))
                inputs_v.append(np.asarray(source_v[start : start + temporal_window], dtype=np.float32))
                targets_u.append(np.asarray(source_u[target_start : target_start + temporal_window], dtype=np.float32))
                targets_v.append(np.asarray(source_v[target_start : target_start + temporal_window], dtype=np.float32))
                if pair_steps is not None:
                    pair_steps.append(start)
        else:
            for step in range(len(source_u) - 1):
                inputs_u.append(source_u[step])
                inputs_v.append(source_v[step])
                targets_u.append(source_u[step + 1])
                targets_v.append(source_v[step + 1])
                if pair_steps is not None:
                    pair_steps.append(step)

    return inputs_u, inputs_v, targets_u, targets_v, pair_steps, trajectory_u, trajectory_v


def build_noisy_trajectories_coupled(
    trajectories: List[Dict[str, np.ndarray]],
    config: GrayScottConfig,
    *,
    show_progress: bool,
    progress_desc: str,
) -> List[Dict[str, np.ndarray]]:
    """Build noisy coupled trajectories while preserving time-step alignment."""
    noisy_data: List[Dict[str, np.ndarray]] = []

    for data in progress_iter(
        trajectories,
        enabled=show_progress,
        desc=progress_desc,
        total=len(trajectories),
    ):
        u_traj = np.asarray(data["u"], dtype=np.float32)
        v_traj = np.asarray(data["v"], dtype=np.float32)

        source_u = np.empty_like(u_traj, dtype=np.float32)
        source_v = np.empty_like(v_traj, dtype=np.float32)
        for step in range(len(u_traj)):
            u_noisy_step, v_noisy_step = add_hf_noise_coupled(
                u_traj[step],
                v_traj[step],
                config.noise_level,
                config.nx,
                config.ny,
                Lx=config.Lx,
                Ly=config.Ly,
            )
            source_u[step] = np.asarray(u_noisy_step, dtype=np.float32)
            source_v[step] = np.asarray(v_noisy_step, dtype=np.float32)

        noisy_data.append({"u": source_u, "v": source_v})

    return noisy_data
