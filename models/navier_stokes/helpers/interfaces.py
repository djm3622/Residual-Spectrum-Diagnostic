"""Protocol interfaces for Navier-Stokes one-step models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset


class OneStepModel(Protocol):
    """Minimal model protocol used by run scripts."""

    def forward(self, omega: np.ndarray) -> np.ndarray:
        ...

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        lr: float,
        n_iter: int,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,
        use_one_cycle_lr: bool = False,
        one_cycle_pct_start: float = 0.3,
        one_cycle_div_factor: float = 25.0,
        one_cycle_final_div_factor: float = 10000.0,
        trajectory: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
        val_inputs: List[np.ndarray] | None = None,
        val_targets: List[np.ndarray] | None = None,
        checkpoint_callback: (
            Callable[[int, float], None]
            | Callable[[int, float, Dict[str, Any]], None]
            | None
        ) = None,
        early_stopping_patience: int | None = None,
        resume_state: Dict[str, Any] | None = None,
        train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = None,
        val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = None,
        dataloader_num_workers: int = 0,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        ...

    def state_dict(self) -> Dict[str, np.ndarray]:
        ...
