"""Physics-consistent reaction-diffusion surrogate model."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from data.reaction_diffusion import GrayScottConfig, GrayScottSolver

from ..helpers.sanitization import sanitize_species


class PhysicsConsistentSurrogate2DCoupled:
    """Gray-Scott rollout model using the same operator-split physics as data generation."""

    def __init__(
        self,
        config: GrayScottConfig,
        snapshot_dt: float,
        device: str = "cpu",
    ):
        self.config = config
        self.snapshot_dt = max(float(snapshot_dt), 0.0)
        self.internal_dt = max(float(config.dt), 1e-8)
        # Kept for run metadata parity with trainable models.
        self.device = str(device)
        self.solver = GrayScottSolver(config)
        self.loss_name = "physics"

    def _advance_snapshot(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_next = np.asarray(u, dtype=np.float64).copy()
        v_next = np.asarray(v, dtype=np.float64).copy()

        remaining = self.snapshot_dt
        while remaining > 1e-12:
            dt_step = min(self.internal_dt, remaining)
            u_next, v_next = self.solver.step(u_next, v_next, dt=dt_step)
            remaining -= dt_step

        return sanitize_species(u_next.astype(np.float32, copy=False)), sanitize_species(
            v_next.astype(np.float32, copy=False)
        )

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._advance_snapshot(u, v)

    def train(
        self,
        inputs_u: List[np.ndarray],
        inputs_v: List[np.ndarray],
        targets_u: List[np.ndarray],
        targets_v: List[np.ndarray],
        lr: float = 0.001,
        n_iter: int = 100,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,
        use_one_cycle_lr: bool = False,
        one_cycle_pct_start: float = 0.3,
        one_cycle_div_factor: float = 25.0,
        one_cycle_final_div_factor: float = 10000.0,
        trajectory_u: List[np.ndarray] | None = None,
        trajectory_v: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
        val_inputs_u: List[np.ndarray] | None = None,
        val_inputs_v: List[np.ndarray] | None = None,
        val_targets_u: List[np.ndarray] | None = None,
        val_targets_v: List[np.ndarray] | None = None,
        pair_steps: List[int] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
        u_weight: float = 1.0,
        v_weight: float = 1.0,
        channel_balance_cap: float = 3.0,
        dynamics_weight: float = 0.0,
        early_step_bias: float = 0.0,
        early_step_decay: float = 24.0,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        # Physics-only model: no trainable parameters for this case study.
        return

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "model_type": np.asarray("physics_consistent"),
            "snapshot_dt": np.asarray(self.snapshot_dt, dtype=np.float32),
            "internal_dt": np.asarray(self.internal_dt, dtype=np.float32),
            "Du": np.asarray(self.config.Du, dtype=np.float32),
            "Dv": np.asarray(self.config.Dv, dtype=np.float32),
            "F": np.asarray(self.config.F, dtype=np.float32),
            "k": np.asarray(self.config.k, dtype=np.float32),
        }
