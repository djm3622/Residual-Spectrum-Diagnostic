"""Neuraloperator-backed Navier-Stokes surrogate model."""

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
    maybe_disable_grad_scaler_for_complex_params,
    resolve_torch_device,
    train_autocast,
)

from ..helpers.neural_operator import build_fno_like_model, resolve_operator_config
from ..helpers.sanitization import sanitize_field


class NeuralOperatorSurrogate2D:
    """Neuraloperator-backed one-step map for Navier-Stokes."""

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
        temporal_cfg = merged_config.get("temporal")
        self.temporal_enabled = bool(
            isinstance(temporal_cfg, Mapping) and temporal_cfg.get("enabled", False)
        )
        self.temporal_window = 1
        self.temporal_target_mode = "shifted"
        self.temporal_modes: Tuple[int, int, int] | None = None
        n_modes_override = None
        if self.temporal_enabled:
            if operator == "uno":
                raise ValueError("Temporal window mode is supported for FNO/TFNO, not UNO.")
            self.temporal_window = max(2, int(temporal_cfg.get("input_steps", 20)))
            output_steps = max(2, int(temporal_cfg.get("output_steps", self.temporal_window)))
            if output_steps != self.temporal_window:
                raise ValueError(
                    "Current temporal TFNO/FNO path requires output_steps == input_steps "
                    f"(got input_steps={self.temporal_window}, output_steps={output_steps})."
                )
            target_mode = str(temporal_cfg.get("target_mode", "shifted")).strip().lower().replace("-", "_")
            if target_mode not in {"shifted", "next_block"}:
                raise ValueError("temporal.target_mode must be 'shifted' or 'next_block'.")
            self.temporal_target_mode = target_mode

            raw_modes = merged_config.get("n_modes")
            max_t = max(2, self.temporal_window // 2)
            max_x = max(2, int(self.nx) // 2)
            max_y = max(2, int(self.ny) // 2)
            if isinstance(raw_modes, (list, tuple)) and len(raw_modes) >= 3:
                mt = int(raw_modes[0])
                mx = int(raw_modes[1])
                my = int(raw_modes[2])
            elif isinstance(raw_modes, (list, tuple)) and len(raw_modes) >= 2:
                mt = int(temporal_cfg.get("n_modes_time", min(8, max_t)))
                mx = int(raw_modes[0])
                my = int(raw_modes[1])
            elif isinstance(raw_modes, (int, float)):
                mt = int(temporal_cfg.get("n_modes_time", min(8, max_t)))
                mx = int(raw_modes)
                my = int(raw_modes)
            else:
                mt = int(temporal_cfg.get("n_modes_time", min(8, max_t)))
                mx = int(temporal_cfg.get("n_modes_x", 20))
                my = int(temporal_cfg.get("n_modes_y", 20))

            mt = int(np.clip(mt, 2, max_t))
            mx = int(np.clip(mx, 2, max_x))
            my = int(np.clip(my, 2, max_y))
            self.temporal_modes = (mt, mx, my)
            n_modes_override = self.temporal_modes

        self.net = build_fno_like_model(
            operator=operator,
            in_channels=1,
            out_channels=1,
            nx=nx,
            ny=ny,
            config=merged_config,
            n_modes_override=n_modes_override,
        ).to(self.device)
        self.grad_scaler = maybe_disable_grad_scaler_for_complex_params(self.grad_scaler, self.net)
        self.input_mean = torch.zeros(1, 1, 1, 1, device=self.device)
        self.input_std = torch.ones(1, 1, 1, 1, device=self.device)
        self.target_mean = torch.zeros(1, 1, 1, 1, device=self.device)
        self.target_std = torch.ones(1, 1, 1, 1, device=self.device)
        self.net.eval()

    def _prepare_window(self, omega_window: np.ndarray) -> np.ndarray:
        arr = np.asarray(omega_window, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected window with shape [T,H,W], got {arr.shape}.")
        if arr.shape[1:] != (self.nx, self.ny):
            raise ValueError(
                f"Window spatial shape mismatch: expected {(self.nx, self.ny)}, got {arr.shape[1:]}."
            )
        if arr.shape[0] < self.temporal_window:
            pad = np.repeat(arr[-1][np.newaxis, ...], self.temporal_window - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > self.temporal_window:
            arr = arr[-self.temporal_window :]
        return np.asarray(arr, dtype=np.float32)

    def _fit_normalizers(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        reduce_dims = tuple(idx for idx in range(x_train.ndim) if idx != 1)
        self.input_mean = torch.mean(x_train, dim=reduce_dims, keepdim=True).to(torch.float32)
        self.input_std = torch.std(x_train, dim=reduce_dims, keepdim=True, unbiased=False).to(torch.float32)
        self.input_std = torch.clamp(self.input_std, min=1e-6)

        target_reduce_dims = tuple(idx for idx in range(y_train.ndim) if idx != 1)
        self.target_mean = torch.mean(y_train, dim=target_reduce_dims, keepdim=True).to(torch.float32)
        self.target_std = torch.std(y_train, dim=target_reduce_dims, keepdim=True, unbiased=False).to(torch.float32)
        self.target_std = torch.clamp(self.target_std, min=1e-6)

    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / self.input_std
        pred_norm = self.net(x_norm)
        return pred_norm * self.target_std + self.target_mean

    def _scaled_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.target_mean) / self.target_std

    def predict_window(self, omega_window: np.ndarray) -> np.ndarray:
        if not self.temporal_enabled:
            raise ValueError("predict_window is only available when temporal mode is enabled.")
        window = self._prepare_window(omega_window)
        x = torch.from_numpy(window[np.newaxis, np.newaxis, ...]).to(self.device)
        with torch.inference_mode():
            pred = self._predict_batch(x)[0, 0].detach().cpu().numpy()
        return np.asarray([sanitize_field(frame) for frame in pred], dtype=np.float32)

    def rollout(
        self,
        omega0: np.ndarray,
        n_steps: int,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        total_steps = max(1, int(n_steps))
        omega0_f32 = sanitize_field(np.asarray(omega0, dtype=np.float32))

        trajectory = np.zeros((total_steps, self.nx, self.ny), dtype=np.float32)
        trajectory[0] = omega0_f32
        if total_steps == 1:
            return trajectory

        if not self.temporal_enabled:
            omega = omega0_f32
            for step in range(1, total_steps):
                omega = sanitize_field(self.forward(omega))
                trajectory[step] = omega
            return trajectory

        if context is not None:
            context_arr = np.asarray(context, dtype=np.float32)
            if context_arr.ndim == 3 and context_arr.shape[0] > 0:
                window = self._prepare_window(context_arr[: self.temporal_window])
            else:
                window = np.repeat(omega0_f32[np.newaxis, ...], self.temporal_window, axis=0)
        else:
            window = np.repeat(omega0_f32[np.newaxis, ...], self.temporal_window, axis=0)

        known_steps = min(total_steps, self.temporal_window)
        trajectory[:known_steps] = window[:known_steps]
        cursor = known_steps

        while cursor < total_steps:
            pred_window = self.predict_window(window)
            if self.temporal_target_mode == "next_block":
                take = min(pred_window.shape[0], total_steps - cursor)
                trajectory[cursor : cursor + take] = pred_window[:take]
                cursor += take
                window = pred_window
            else:
                next_frame = sanitize_field(pred_window[-1])
                trajectory[cursor] = next_frame
                cursor += 1
                window = np.concatenate([window[1:], next_frame[np.newaxis, ...]], axis=0)

        return trajectory

    def forward(self, omega: np.ndarray) -> np.ndarray:
        omega_arr = np.asarray(omega, dtype=np.float32)
        if self.temporal_enabled:
            if omega_arr.ndim == 2:
                omega_window = np.repeat(omega_arr[np.newaxis, ...], self.temporal_window, axis=0)
            else:
                omega_window = self._prepare_window(omega_arr)
            pred_window = self.predict_window(omega_window)
            frame = pred_window[0] if self.temporal_target_mode == "next_block" else pred_window[-1]
            return sanitize_field(frame)

        inp = omega_arr[np.newaxis, np.newaxis, ...]
        x = torch.from_numpy(inp).to(self.device)
        with torch.inference_mode():
            pred = self._predict_batch(x)[0, 0].detach().cpu().numpy()
        return sanitize_field(pred)

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        lr: float = 0.001,
        n_iter: int = 100,
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
        if not inputs:
            raise ValueError("Training inputs are empty.")

        x_arr = np.asarray(inputs, dtype=np.float32)
        y_arr = np.asarray(targets, dtype=np.float32)
        if self.temporal_enabled:
            if x_arr.ndim != 4 or y_arr.ndim != 4:
                raise ValueError(
                    "Temporal mode expects windowed tensors with shape [N,T,H,W] "
                    f"(got x={x_arr.shape}, y={y_arr.shape})."
                )
        else:
            if x_arr.ndim != 3 or y_arr.ndim != 3:
                raise ValueError(
                    "One-step mode expects tensors with shape [N,H,W] "
                    f"(got x={x_arr.shape}, y={y_arr.shape})."
                )

        x_train = torch.from_numpy(x_arr[:, np.newaxis, ...]).to(self.device)
        y_train = torch.from_numpy(y_arr[:, np.newaxis, ...]).to(self.device)
        self._fit_normalizers(x_train, y_train)
        has_val = (
            val_inputs is not None
            and val_targets is not None
            and len(val_inputs) > 0
            and len(val_inputs) == len(val_targets)
        )
        if has_val:
            x_val_arr = np.asarray(val_inputs, dtype=np.float32)
            y_val_arr = np.asarray(val_targets, dtype=np.float32)
            x_val = torch.from_numpy(x_val_arr[:, np.newaxis, ...]).to(self.device)
            y_val = torch.from_numpy(y_val_arr[:, np.newaxis, ...]).to(self.device)
        else:
            x_val = None
            y_val = None

        n_samples = x_train.shape[0]
        batch = max(1, min(int(batch_size), n_samples))
        clip = max(float(grad_clip), 1e-8)
        horizon = max(1, int(rollout_horizon))
        rollout_w = max(0.0, float(rollout_weight))

        use_rollout = (
            (not self.temporal_enabled)
            and rollout_w > 0.0
            and horizon > 1
            and trajectory is not None
            and len(trajectory) > 0
        )
        if use_rollout:
            seq = torch.from_numpy(np.asarray(trajectory, dtype=np.float32)).to(self.device)
            n_traj, n_steps, _, _ = seq.shape
            max_start = n_steps - horizon - 1
            if max_start < 0:
                use_rollout = False
            else:
                rollout_batch = max(1, min(batch // 2, n_traj))
        else:
            seq = None
            rollout_batch = 0
            max_start = -1

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
                        pred_val = self._predict_batch(xb_val)
                        loss_val = self.objective_loss(
                            self._scaled_state(pred_val),
                            self._scaled_state(yb_val),
                        )
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
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch):
                idx = perm[start : start + batch]
                xb = x_train.index_select(0, idx)
                yb = y_train.index_select(0, idx)

                with train_autocast(self.device):
                    pred = self._predict_batch(xb)
                    one_step_loss = self.objective_loss(
                        self._scaled_state(pred),
                        self._scaled_state(yb),
                    )

                    rollout_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                    if use_rollout and seq is not None and rollout_batch > 0:
                        traj_idx = torch.randint(0, seq.shape[0], (rollout_batch,), device=self.device)
                        start_idx = torch.randint(0, max_start + 1, (rollout_batch,), device=self.device)

                        state_roll = seq[traj_idx, start_idx].unsqueeze(1)
                        for offset in range(1, horizon + 1):
                            pred_roll = self._predict_batch(state_roll)
                            target_roll = seq[traj_idx, start_idx + offset].unsqueeze(1)
                            rollout_loss = rollout_loss + self.objective_loss(
                                self._scaled_state(pred_roll),
                                self._scaled_state(target_roll),
                            )
                            state_roll = pred_roll
                        rollout_loss = rollout_loss / float(horizon)

                    loss = one_step_loss + rollout_w * rollout_loss

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
        payload["_input_mean"] = self.input_mean.detach().cpu().numpy()
        payload["_input_std"] = self.input_std.detach().cpu().numpy()
        payload["_target_mean"] = self.target_mean.detach().cpu().numpy()
        payload["_target_std"] = self.target_std.detach().cpu().numpy()
        payload["_temporal_enabled"] = np.asarray(int(self.temporal_enabled), dtype=np.int64)
        payload["_temporal_window"] = np.asarray(int(self.temporal_window), dtype=np.int64)
        payload["_temporal_target_mode"] = np.asarray(self.temporal_target_mode)
        if self.temporal_modes is not None:
            payload["_temporal_modes"] = np.asarray(self.temporal_modes, dtype=np.int64)
        return payload
