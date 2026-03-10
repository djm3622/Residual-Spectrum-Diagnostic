"""Surrogate models for 2D Navier-Stokes trajectories."""

from __future__ import annotations

from typing import Dict, List, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import LOSS_CHOICES, ObjectiveLoss, normalize_loss_name
from utils.torch_runtime import (
    build_adam_optimizer,
    build_grad_scaler,
    configure_torch_backend,
    resolve_torch_device,
    train_autocast,
)
from utils.progress import progress_range


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
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        ...

    def state_dict(self) -> Dict[str, np.ndarray]:
        ...


def _sanitize_field(field: np.ndarray) -> np.ndarray:
    """Convert non-finite outputs to bounded finite values."""
    clean = np.nan_to_num(field, nan=0.0, posinf=1e6, neginf=-1e6)
    np.clip(clean, -1e6, 1e6, out=clean)
    return clean


class _PeriodicConvBlock(nn.Module):
    """Residual periodic convolution block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.gelu(x + y)


class _NSNonlinearOneStepNet(nn.Module):
    """One-step nonlinear residual map: omega_{t+1} = omega_t + Delta(omega_t)."""

    def __init__(self, width: int = 64, depth: int = 5):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList([_PeriodicConvBlock(width) for _ in range(depth)])
        self.out_proj = nn.Conv2d(width, 1, kernel_size=1)
        # Learn residual scale from near-identity initialization for stable rollout.
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        with torch.no_grad():
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        # omega: [batch, nx, ny]
        x = omega.unsqueeze(1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        delta = self.out_proj(h).squeeze(1)
        return omega + self.residual_scale * delta


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
        self.net = _NSNonlinearOneStepNet(width=width, depth=depth).to(self.device)
        self.net.eval()

    def forward(self, omega: np.ndarray) -> np.ndarray:
        inp = np.asarray(omega, dtype=np.float32)[np.newaxis, ...]
        x = torch.from_numpy(inp).to(self.device)
        with torch.inference_mode():
            pred = self.net(x)[0].cpu().numpy()
        return _sanitize_field(pred)

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
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        if not inputs:
            raise ValueError("Training inputs are empty.")

        x_train = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(self.device)
        y_train = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(self.device)

        n_samples = x_train.shape[0]
        batch = max(1, min(int(batch_size), n_samples))
        clip = max(float(grad_clip), 1e-8)
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

        optimizer = build_adam_optimizer(self.net.parameters(), lr=float(lr), device=self.device)

        self.net.train()
        total_iter = max(1, int(n_iter))
        iter_desc = progress_desc or "Training iterations"
        for _ in progress_range(total_iter, enabled=show_progress, desc=iter_desc):
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
        self.net.eval()

    def state_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        for key, tensor in self.net.state_dict().items():
            payload[key] = tensor.detach().cpu().numpy()
        return payload


def rollout_2d(model: OneStepModel, omega0: np.ndarray, n_steps: int) -> np.ndarray:
    """Autoregressive rollout for one-step models."""
    nx, ny = omega0.shape
    trajectory = np.zeros((n_steps, nx, ny), dtype=np.float32)
    trajectory[0] = np.asarray(omega0, dtype=np.float32)

    omega = np.asarray(omega0, dtype=np.float32)
    for step in range(1, n_steps):
        omega = model.forward(omega)
        omega = _sanitize_field(omega)
        trajectory[step] = omega

    return trajectory


def build_model(
    method: str,
    nx: int,
    ny: int,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    model_width: int = 64,
    model_depth: int = 5,
) -> OneStepModel:
    """Factory for NS surrogate models selected by CLI method arg."""
    normalized = method.strip().lower()

    if normalized in {"conv", "convolutional", "spectral", "nonlinear"}:
        return ConvolutionalSurrogate2D(
            nx,
            ny,
            seed=seed,
            device=device,
            loss=loss,
            model_width=model_width,
            model_depth=model_depth,
        )

    raise ValueError(
        f"Unsupported method '{method}'. Use one of: conv, convolutional, spectral, nonlinear"
    )
