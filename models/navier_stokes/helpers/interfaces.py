"""Protocol interfaces for Navier-Stokes one-step models."""

from __future__ import annotations

from typing import Dict, List, Protocol

import numpy as np


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
        trajectory: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
        val_inputs: List[np.ndarray] | None = None,
        val_targets: List[np.ndarray] | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        ...

    def state_dict(self) -> Dict[str, np.ndarray]:
        ...
