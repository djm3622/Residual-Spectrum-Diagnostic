"""Neuraloperator-backed coupled reaction-diffusion surrogate model."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from models.losses import ObjectiveLoss
from utils.progress import progress_range
from utils.torch_runtime import (
    build_adam_optimizer,
    build_grad_scaler,
    configure_torch_backend,
    resolve_torch_device,
    maybe_disable_grad_scaler_for_complex_params,
    train_autocast,
)

from ..helpers.neural_operator import build_fno_like_model, resolve_operator_config
from ..helpers.sanitization import sanitize_species


class NeuralOperatorSurrogate2DCoupled:
    """Neuraloperator-backed one-step map for coupled Gray-Scott dynamics."""

    def __init__(
        self,
        nx: int,
        ny: int,
        operator: str,
        seed: int | None = None,
        device: str = "auto",
        loss: str = "combined",
        operator_config: Optional[Mapping[str, Any]] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.operator = operator
        self.device = resolve_torch_device(device)
        configure_torch_backend(self.device)
        self.grad_scaler = build_grad_scaler(self.device)
        self.objective_loss = ObjectiveLoss(nx=nx, ny=ny, device=self.device, loss=loss)
        self.loss_name = self.objective_loss.loss_name

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        merged_config = resolve_operator_config(operator, operator_config)
        self.net = build_fno_like_model(
            operator=operator,
            in_channels=2,
            out_channels=2,
            nx=nx,
            ny=ny,
            config=merged_config,
        ).to(self.device)
        self.grad_scaler = maybe_disable_grad_scaler_for_complex_params(self.grad_scaler, self.net)
        self.net.eval()

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.stack([u, v], axis=0).astype(np.float32, copy=False)[np.newaxis, ...]
        xb = torch.from_numpy(x).to(self.device)
        with torch.inference_mode():
            pred = self.net(xb)[0].cpu().numpy()

        u_next = sanitize_species(pred[0])
        v_next = sanitize_species(pred[1])
        return u_next, v_next

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
        trajectory_u: List[np.ndarray] | None = None,
        trajectory_v: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
        val_inputs_u: List[np.ndarray] | None = None,
        val_inputs_v: List[np.ndarray] | None = None,
        val_targets_u: List[np.ndarray] | None = None,
        val_targets_v: List[np.ndarray] | None = None,
        pair_steps: List[int] | None = None,
        u_weight: float = 1.0,
        v_weight: float = 1.0,
        channel_balance_cap: float = 3.0,
        dynamics_weight: float = 0.0,
        early_step_bias: float = 0.0,
        early_step_decay: float = 24.0,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        if not inputs_u:
            raise ValueError("Training inputs are empty.")

        x_u = np.asarray(inputs_u, dtype=np.float32)
        x_v = np.asarray(inputs_v, dtype=np.float32)
        y_u = np.asarray(targets_u, dtype=np.float32)
        y_v = np.asarray(targets_v, dtype=np.float32)

        x_train = torch.from_numpy(np.stack([x_u, x_v], axis=1)).to(self.device)
        y_train = torch.from_numpy(np.stack([y_u, y_v], axis=1)).to(self.device)
        has_val = (
            val_inputs_u is not None
            and val_inputs_v is not None
            and val_targets_u is not None
            and val_targets_v is not None
            and len(val_inputs_u) > 0
            and len(val_inputs_u) == len(val_inputs_v) == len(val_targets_u) == len(val_targets_v)
        )
        if has_val:
            x_val = torch.from_numpy(
                np.stack(
                    [
                        np.asarray(val_inputs_u, dtype=np.float32),
                        np.asarray(val_inputs_v, dtype=np.float32),
                    ],
                    axis=1,
                )
            ).to(self.device)
            y_val = torch.from_numpy(
                np.stack(
                    [
                        np.asarray(val_targets_u, dtype=np.float32),
                        np.asarray(val_targets_v, dtype=np.float32),
                    ],
                    axis=1,
                )
            ).to(self.device)
        else:
            x_val = None
            y_val = None
        channel_rms = torch.sqrt(torch.mean(y_train * y_train, dim=(0, 2, 3), keepdim=True))
        channel_rms = torch.clamp(channel_rms, min=1e-6)
        balance_cap = max(1.0, float(channel_balance_cap))
        reference_scale = torch.max(channel_rms)
        min_scale = torch.clamp(reference_scale / balance_cap, min=1e-6)
        channel_scale = torch.clamp(channel_rms, min=min_scale)

        channel_weight = torch.tensor(
            [max(float(u_weight), 1e-6), max(float(v_weight), 1e-6)],
            dtype=torch.float32,
            device=self.device,
        )
        channel_weight = channel_weight / torch.mean(channel_weight)
        channel_weight_sqrt = torch.sqrt(channel_weight).view(1, 2, 1, 1)

        def _scaled_state(state: torch.Tensor) -> torch.Tensor:
            return (state / channel_scale) * channel_weight_sqrt

        n_samples = x_train.shape[0]
        batch = max(1, min(int(batch_size), n_samples))
        clip = max(float(grad_clip), 1e-8)
        horizon = max(1, int(rollout_horizon))
        rollout_w = max(0.0, float(rollout_weight))
        dynamics_w = max(0.0, float(dynamics_weight))

        early_bias = max(0.0, float(early_step_bias))
        early_decay = max(1.0, float(early_step_decay))
        sample_weights: torch.Tensor | None = None
        if pair_steps is not None and len(pair_steps) == n_samples and early_bias > 0.0:
            step_arr = np.asarray(pair_steps, dtype=np.float32)
            step_weights = 1.0 + early_bias * np.exp(-step_arr / early_decay)
            sample_weights = torch.from_numpy(step_weights).to(self.device)
            sample_weights = sample_weights / torch.sum(sample_weights)

        use_rollout = (
            rollout_w > 0.0
            and horizon > 1
            and trajectory_u is not None
            and trajectory_v is not None
            and len(trajectory_u) > 0
        )
        if use_rollout:
            seq_u = torch.from_numpy(np.asarray(trajectory_u, dtype=np.float32)).to(self.device)
            seq_v = torch.from_numpy(np.asarray(trajectory_v, dtype=np.float32)).to(self.device)
            n_traj, n_steps, _, _ = seq_u.shape
            max_start = n_steps - horizon - 1
            if max_start < 0:
                use_rollout = False
            else:
                rollout_batch = max(1, min(batch // 2, n_traj))
                if early_bias > 0.0:
                    start_steps = torch.arange(max_start + 1, device=self.device, dtype=torch.float32)
                    rollout_start_weights = 1.0 + early_bias * torch.exp(-start_steps / early_decay)
                    rollout_start_weights = rollout_start_weights / torch.sum(rollout_start_weights)
                else:
                    rollout_start_weights = None
        else:
            seq_u = None
            seq_v = None
            rollout_batch = 0
            max_start = -1
            rollout_start_weights = None

        optimizer = build_adam_optimizer(self.net.parameters(), lr=float(lr), device=self.device)

        def _compute_validation_loss() -> float:
            if not has_val or x_val is None or y_val is None:
                return float("nan")

            was_training = self.net.training
            self.net.eval()
            total = 0.0
            count = 0
            with torch.inference_mode():
                for start in range(0, x_val.shape[0], batch):
                    xb_val = x_val[start : start + batch]
                    yb_val = y_val[start : start + batch]
                    with train_autocast(self.device):
                        pred_val = self.net(xb_val)
                        one_step_val = self.objective_loss(
                            _scaled_state(pred_val),
                            _scaled_state(yb_val),
                        )
                        dynamics_val = torch.zeros((), device=self.device, dtype=one_step_val.dtype)
                        if dynamics_w > 0.0:
                            dynamics_val = self.objective_loss(
                                _scaled_state(pred_val - xb_val),
                                _scaled_state(yb_val - xb_val),
                            )
                        loss_val = one_step_val + dynamics_w * dynamics_val
                    if torch.isfinite(loss_val):
                        bs = int(xb_val.shape[0])
                        total += float(loss_val.item()) * bs
                        count += bs
            if was_training:
                self.net.train()

            if count == 0:
                return float("nan")
            return total / float(count)

        self.net.train()
        total_iter = max(1, int(n_iter))
        iter_desc = progress_desc or "Training iterations"
        iter_progress = progress_range(total_iter, enabled=show_progress, desc=iter_desc)
        for _ in iter_progress:
            train_loss_sum = 0.0
            train_loss_count = 0
            if sample_weights is not None:
                perm = torch.multinomial(sample_weights, n_samples, replacement=True)
            else:
                perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch):
                idx = perm[start : start + batch]
                xb = x_train.index_select(0, idx)
                yb = y_train.index_select(0, idx)

                with train_autocast(self.device):
                    pred = self.net(xb)
                    pred_scaled = _scaled_state(pred)
                    target_scaled = _scaled_state(yb)
                    one_step_loss = self.objective_loss(pred_scaled, target_scaled)

                    rollout_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                    if use_rollout and seq_u is not None and seq_v is not None and rollout_batch > 0:
                        traj_idx = torch.randint(0, seq_u.shape[0], (rollout_batch,), device=self.device)
                        if rollout_start_weights is not None:
                            start_idx = torch.multinomial(rollout_start_weights, rollout_batch, replacement=True)
                        else:
                            start_idx = torch.randint(0, max_start + 1, (rollout_batch,), device=self.device)

                        cur_u = seq_u[traj_idx, start_idx]
                        cur_v = seq_v[traj_idx, start_idx]
                        state_roll = torch.stack([cur_u, cur_v], dim=1)

                        for offset in range(1, horizon + 1):
                            pred_roll = self.net(state_roll)
                            tgt_u = seq_u[traj_idx, start_idx + offset]
                            tgt_v = seq_v[traj_idx, start_idx + offset]
                            tgt = torch.stack([tgt_u, tgt_v], dim=1)
                            rollout_loss = rollout_loss + self.objective_loss(
                                _scaled_state(pred_roll),
                                _scaled_state(tgt),
                            )
                            state_roll = pred_roll

                        rollout_loss = rollout_loss / float(horizon)

                    dynamics_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                    if dynamics_w > 0.0:
                        dynamics_loss = self.objective_loss(
                            _scaled_state(pred - xb),
                            _scaled_state(yb - xb),
                        )

                    loss = one_step_loss + rollout_w * rollout_loss + dynamics_w * dynamics_loss

                if not torch.isfinite(loss):
                    continue

                train_loss_sum += float(loss.detach().item())
                train_loss_count += 1

                optimizer.zero_grad(set_to_none=True)
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=clip)
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=clip)
                    optimizer.step()

            if show_progress and hasattr(iter_progress, "set_postfix"):
                train_loss_value = (
                    train_loss_sum / float(train_loss_count) if train_loss_count > 0 else float("nan")
                )
                val_loss_value = _compute_validation_loss()
                postfix = {
                    "train_loss": f"{train_loss_value:.3e}" if np.isfinite(train_loss_value) else "nan",
                    "val_loss": f"{val_loss_value:.3e}" if np.isfinite(val_loss_value) else "n/a",
                }
                iter_progress.set_postfix(postfix, refresh=False)
        self.net.eval()

    def state_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        for key, value in self.net.state_dict().items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                payload[key] = np.asarray(str(value))
            else:
                payload[key] = np.asarray(value)
        payload["_operator"] = np.asarray(self.operator)
        return payload
