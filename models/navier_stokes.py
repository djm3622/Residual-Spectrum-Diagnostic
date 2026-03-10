"""Surrogate models for 2D Navier-Stokes trajectories."""

from __future__ import annotations

from typing import Dict, List, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, width: int = 32, depth: int = 3):
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
    ):
        self.nx = nx
        self.ny = ny
        self.device = resolve_torch_device(device)
        configure_torch_backend(self.device)
        self.grad_scaler = build_grad_scaler(self.device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.net = _NSNonlinearOneStepNet().to(self.device)
        self.net.eval()

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hat = torch.fft.rfft2(pred)
        target_hat = torch.fft.rfft2(target)
        return F.mse_loss(pred_hat.real, target_hat.real) + F.mse_loss(pred_hat.imag, target_hat.imag)

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred - torch.roll(pred, shifts=1, dims=-2)
        pred_dy = pred - torch.roll(pred, shifts=1, dims=-1)
        target_dx = target - torch.roll(target, shifts=1, dims=-2)
        target_dy = target - torch.roll(target, shifts=1, dims=-1)
        return F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)

    def _combined_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        spec = self._spectral_loss(pred, target)
        grad = self._gradient_loss(pred, target)
        return mse + 0.2 * spec + 0.1 * grad

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
                    loss = self._combined_loss(pred, yb)

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
) -> OneStepModel:
    """Factory for NS surrogate models selected by CLI method arg."""
    normalized = method.strip().lower()

    if normalized in {"conv", "convolutional", "spectral", "nonlinear"}:
        return ConvolutionalSurrogate2D(nx, ny, seed=seed, device=device)

    raise ValueError(
        f"Unsupported method '{method}'. Use one of: conv, convolutional, spectral, nonlinear"
    )
