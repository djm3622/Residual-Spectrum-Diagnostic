"""Convolutional Navier-Stokes surrogate model."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from models.losses import ObjectiveLoss
from utils.progress import progress_range
from utils.torch_runtime import (
    build_adam_optimizer,
    build_grad_scaler,
    configure_torch_backend,
    resolve_torch_device,
    train_autocast,
)

from ..helpers.sanitization import sanitize_field
from ..layers import NSNonlinearOneStepNet


class ConvolutionalSurrogate2D:
    """Nonlinear one-step periodic residual integrator for Navier-Stokes."""

    def __init__(
        self,
        nx: int,
        ny: int,
        seed: int | None = None,
        device: str = "auto",
        loss: str = "combined",
        model_width: int = 64,
        model_depth: int = 5,
    ):
        self.nx = nx
        self.ny = ny
        self.device = resolve_torch_device(device)
        configure_torch_backend(self.device)
        self.grad_scaler = build_grad_scaler(self.device)
        self.objective_loss = ObjectiveLoss(nx=nx, ny=ny, device=self.device, loss=loss)
        self.loss_name = self.objective_loss.loss_name

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        width = max(8, int(model_width))
        depth = max(1, int(model_depth))
        self.net = NSNonlinearOneStepNet(width=width, depth=depth).to(self.device)
        self.net.eval()

    def forward(self, omega: np.ndarray) -> np.ndarray:
        inp = np.asarray(omega, dtype=np.float32)[np.newaxis, ...]
        x = torch.from_numpy(inp).to(self.device)
        with torch.inference_mode():
            pred = self.net(x)[0].cpu().numpy()
        return sanitize_field(pred)

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        lr: float = 0.001,
        n_iter: int = 100,
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
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        if not inputs:
            raise ValueError("Training inputs are empty.")

        x_train = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(self.device)
        y_train = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(self.device)
        has_val = (
            val_inputs is not None
            and val_targets is not None
            and len(val_inputs) > 0
            and len(val_inputs) == len(val_targets)
        )
        if has_val:
            x_val = torch.from_numpy(np.asarray(val_inputs, dtype=np.float32)).to(self.device)
            y_val = torch.from_numpy(np.asarray(val_targets, dtype=np.float32)).to(self.device)
        else:
            x_val = None
            y_val = None

        n_samples = x_train.shape[0]
        batch = max(1, min(int(batch_size), n_samples))
        clip = max(float(grad_clip), 1e-8)
        wd = max(0.0, float(weight_decay))
        one_cycle_enabled = bool(use_one_cycle_lr)
        pct_start = float(np.clip(one_cycle_pct_start, 1e-4, 0.9999))
        div_factor = max(float(one_cycle_div_factor), 1.0)
        final_div_factor = max(float(one_cycle_final_div_factor), 1.0)
        horizon = max(1, int(rollout_horizon))
        rollout_w = max(0.0, float(rollout_weight))

        use_rollout = rollout_w > 0.0 and horizon > 1 and trajectory is not None and len(trajectory) > 0
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

        optimizer = build_adam_optimizer(
            self.net.parameters(),
            lr=float(lr),
            device=self.device,
            weight_decay=wd,
        )
        total_iter = max(1, int(n_iter))
        steps_per_epoch = max(1, int(np.ceil(n_samples / batch)))
        total_steps = max(1, total_iter * steps_per_epoch)
        scheduler = None
        if one_cycle_enabled:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(lr),
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy="cos",
                cycle_momentum=False,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )

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
                        loss_val = self.objective_loss(pred_val, yb_val)
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
                    pred = self.net(xb)
                    one_step_loss = self.objective_loss(pred, yb)

                    rollout_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                    if use_rollout and seq is not None and rollout_batch > 0:
                        traj_idx = torch.randint(0, seq.shape[0], (rollout_batch,), device=self.device)
                        start_idx = torch.randint(0, max_start + 1, (rollout_batch,), device=self.device)

                        state_roll = seq[traj_idx, start_idx]
                        for offset in range(1, horizon + 1):
                            pred_roll = self.net(state_roll)
                            target_roll = seq[traj_idx, start_idx + offset]
                            rollout_loss = rollout_loss + self.objective_loss(pred_roll, target_roll)
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
                if scheduler is not None:
                    scheduler.step()

            if show_progress and hasattr(iter_progress, "set_postfix"):
                train_loss_value = (
                    train_loss_sum / float(train_loss_count) if train_loss_count > 0 else float("nan")
                )
                val_loss_value = _compute_validation_loss()
                lr_value = float(optimizer.param_groups[0]["lr"])
                postfix = {
                    "train_loss": f"{train_loss_value:.3e}" if np.isfinite(train_loss_value) else "nan",
                    "val_loss": f"{val_loss_value:.3e}" if np.isfinite(val_loss_value) else "n/a",
                    "lr": f"{lr_value:.2e}",
                }
                iter_progress.set_postfix(postfix, refresh=False)
        self.net.eval()

    def state_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        for key, tensor in self.net.state_dict().items():
            payload[key] = tensor.detach().cpu().numpy()
        return payload
