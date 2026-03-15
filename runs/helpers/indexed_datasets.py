"""Indexed torch Dataset helpers for PDE surrogate training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from runs.helpers.temporal import window_start_indices, window_target_start


@dataclass(frozen=True)
class PairIndex:
    """One supervised pair index into a trajectory collection."""

    trajectory_index: int
    input_start: int
    target_start: int


def resolve_dataloader_num_workers(configured_workers: int) -> int:
    """Resolve dataloader workers from config with auto fallback.

    A negative value means "auto". Auto uses multiple CPU cores while capping
    worker count to avoid excessive process overhead.
    """

    requested = int(configured_workers)
    if requested >= 0:
        return requested

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 0
    if cpu_count <= 2:
        return 1
    # Use multiple cores by default while staying conservative for large datasets.
    return max(2, min(4, cpu_count // 4))


class NavierStokesIndexedPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Index-based NS supervised pairs without materializing all windows."""

    def __init__(
        self,
        trajectories: Sequence[np.ndarray],
        temporal_enabled: bool,
        temporal_window: int,
        temporal_target_mode: str,
    ):
        self.trajectories = [np.asarray(traj, dtype=np.float32) for traj in trajectories]
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_window = max(1, int(temporal_window))
        self.temporal_target_mode = str(temporal_target_mode)
        self.pairs: List[PairIndex] = []

        for traj_idx, trajectory in enumerate(self.trajectories):
            n_steps = int(trajectory.shape[0])
            if self.temporal_enabled:
                starts = window_start_indices(n_steps, self.temporal_window, self.temporal_target_mode)
                for start in starts:
                    target_start = window_target_start(start, self.temporal_window, self.temporal_target_mode)
                    self.pairs.append(
                        PairIndex(
                            trajectory_index=traj_idx,
                            input_start=int(start),
                            target_start=int(target_start),
                        )
                    )
            else:
                for step in range(max(0, n_steps - 1)):
                    self.pairs.append(
                        PairIndex(
                            trajectory_index=traj_idx,
                            input_start=int(step),
                            target_start=int(step + 1),
                        )
                    )

    def __len__(self) -> int:
        return len(self.pairs)

    def get_pair_arrays(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        pair = self.pairs[int(index)]
        trajectory = self.trajectories[pair.trajectory_index]
        if self.temporal_enabled:
            x = np.asarray(
                trajectory[pair.input_start : pair.input_start + self.temporal_window],
                dtype=np.float32,
            )
            y = np.asarray(
                trajectory[pair.target_start : pair.target_start + self.temporal_window],
                dtype=np.float32,
            )
        else:
            x = np.asarray(trajectory[pair.input_start], dtype=np.float32)
            y = np.asarray(trajectory[pair.target_start], dtype=np.float32)
        return x, y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.get_pair_arrays(index)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class ReactionDiffusionIndexedPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Index-based coupled RD supervised pairs without materializing all windows."""

    def __init__(
        self,
        trajectories: Sequence[Dict[str, np.ndarray]],
        temporal_enabled: bool,
        temporal_window: int,
        temporal_target_mode: str,
    ):
        self.trajectories = [
            {
                "u": np.asarray(item["u"], dtype=np.float32),
                "v": np.asarray(item["v"], dtype=np.float32),
            }
            for item in trajectories
        ]
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_window = max(1, int(temporal_window))
        self.temporal_target_mode = str(temporal_target_mode)
        self.pairs: List[PairIndex] = []

        for traj_idx, trajectory in enumerate(self.trajectories):
            n_steps = int(trajectory["u"].shape[0])
            if self.temporal_enabled:
                starts = window_start_indices(n_steps, self.temporal_window, self.temporal_target_mode)
                for start in starts:
                    target_start = window_target_start(start, self.temporal_window, self.temporal_target_mode)
                    self.pairs.append(
                        PairIndex(
                            trajectory_index=traj_idx,
                            input_start=int(start),
                            target_start=int(target_start),
                        )
                    )
            else:
                for step in range(max(0, n_steps - 1)):
                    self.pairs.append(
                        PairIndex(
                            trajectory_index=traj_idx,
                            input_start=int(step),
                            target_start=int(step + 1),
                        )
                    )

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def pair_steps(self) -> List[int]:
        return [pair.input_start for pair in self.pairs]

    def get_pair_arrays(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pair = self.pairs[int(index)]
        trajectory = self.trajectories[pair.trajectory_index]
        u_traj = trajectory["u"]
        v_traj = trajectory["v"]

        if self.temporal_enabled:
            x_u = np.asarray(
                u_traj[pair.input_start : pair.input_start + self.temporal_window],
                dtype=np.float32,
            )
            x_v = np.asarray(
                v_traj[pair.input_start : pair.input_start + self.temporal_window],
                dtype=np.float32,
            )
            y_u = np.asarray(
                u_traj[pair.target_start : pair.target_start + self.temporal_window],
                dtype=np.float32,
            )
            y_v = np.asarray(
                v_traj[pair.target_start : pair.target_start + self.temporal_window],
                dtype=np.float32,
            )
        else:
            x_u = np.asarray(u_traj[pair.input_start], dtype=np.float32)
            x_v = np.asarray(v_traj[pair.input_start], dtype=np.float32)
            y_u = np.asarray(u_traj[pair.target_start], dtype=np.float32)
            y_v = np.asarray(v_traj[pair.target_start], dtype=np.float32)

        return x_u, x_v, y_u, y_v

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_u, x_v, y_u, y_v = self.get_pair_arrays(index)
        x = np.stack([x_u, x_v], axis=0).astype(np.float32, copy=False)
        y = np.stack([y_u, y_v], axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
