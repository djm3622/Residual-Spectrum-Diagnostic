"""Surrogate models for coupled 2D Gray-Scott trajectories."""

from __future__ import annotations

from typing import Dict, List, Protocol, Tuple

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


class CoupledOneStepModel(Protocol):
    """Minimal protocol for coupled one-step models."""

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def train(
        self,
        inputs_u: List[np.ndarray],
        inputs_v: List[np.ndarray],
        targets_u: List[np.ndarray],
        targets_v: List[np.ndarray],
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


def _sanitize_species(field: np.ndarray) -> np.ndarray:
    """Keep species values finite and within physical bounds."""
    clean = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(clean, 0.0, 1.0, out=clean)
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


class _RDNonlinearOneStepNet(nn.Module):
    """One-step nonlinear residual map for coupled (u, v) fields."""

    def __init__(self, width: int = 32, depth: int = 2):
        super().__init__()
        # Features: [u, v, lap_u, lap_v, u*v, u^2, v^2, u*v^2]
        in_channels = 8
        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([_PeriodicConvBlock(width) for _ in range(depth)])
        self.spatial_out = nn.Conv2d(width, 2, kernel_size=1)

        self.reaction = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 2, kernel_size=1),
        )

        # Positive diffusion coefficients per channel, learned from data.
        self.diff_raw = nn.Parameter(torch.tensor([0.05, 0.05], dtype=torch.float32))
        self.step_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        with torch.no_grad():
            nn.init.zeros_(self.spatial_out.weight)
            nn.init.zeros_(self.spatial_out.bias)

    @staticmethod
    def _laplacian_periodic(field: torch.Tensor) -> torch.Tensor:
        return (
            torch.roll(field, shifts=1, dims=-2)
            + torch.roll(field, shifts=-1, dims=-2)
            + torch.roll(field, shifts=1, dims=-1)
            + torch.roll(field, shifts=-1, dims=-1)
            - 4.0 * field
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: [batch, 2, nx, ny]
        u = state[:, 0:1]
        v = state[:, 1:2]
        lap_u = self._laplacian_periodic(u)
        lap_v = self._laplacian_periodic(v)
        uv = u * v
        uv2 = uv * v

        features = torch.cat([u, v, lap_u, lap_v, uv, u * u, v * v, uv2], dim=1)

        h = self.in_proj(features)
        for block in self.blocks:
            h = block(h)
        spatial_delta = self.spatial_out(h)
        reaction_delta = self.reaction(features)

        diff = F.softplus(self.diff_raw).view(1, 2, 1, 1) * torch.cat([lap_u, lap_v], dim=1)
        delta = reaction_delta + diff + 0.25 * spatial_delta
        return state + self.step_scale * delta


class ConvolutionalSurrogate2DCoupled:
    """Nonlinear one-step periodic residual integrator for Gray-Scott dynamics."""

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

        self.net = _RDNonlinearOneStepNet().to(self.device)
        self.net.eval()

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hat = torch.fft.rfft2(pred, dim=(-2, -1))
        target_hat = torch.fft.rfft2(target, dim=(-2, -1))
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

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.stack([u, v], axis=0).astype(np.float32, copy=False)[np.newaxis, ...]
        xb = torch.from_numpy(x).to(self.device)
        with torch.inference_mode():
            pred = self.net(xb)[0].cpu().numpy()

        u_next = _sanitize_species(pred[0])
        v_next = _sanitize_species(pred[1])
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


def rollout_coupled(
    model: CoupledOneStepModel,
    u0: np.ndarray,
    v0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout for coupled one-step models."""
    nx, ny = u0.shape
    u_traj = np.zeros((n_steps, nx, ny), dtype=np.float32)
    v_traj = np.zeros((n_steps, nx, ny), dtype=np.float32)

    u_traj[0] = np.asarray(u0, dtype=np.float32)
    v_traj[0] = np.asarray(v0, dtype=np.float32)

    u = np.asarray(u0, dtype=np.float32)
    v = np.asarray(v0, dtype=np.float32)
    for step in range(1, n_steps):
        u, v = model.forward(u, v)
        u = _sanitize_species(u)
        v = _sanitize_species(v)
        u_traj[step] = u
        v_traj[step] = v

    return u_traj, v_traj


def build_model(
    method: str,
    nx: int,
    ny: int,
    seed: int,
    device: str = "auto",
) -> CoupledOneStepModel:
    """Factory for coupled RD surrogate models selected by CLI method arg."""
    normalized = method.strip().lower()

    if normalized in {"conv", "convolutional", "spectral", "nonlinear"}:
        return ConvolutionalSurrogate2DCoupled(nx, ny, seed=seed, device=device)

    raise ValueError(
        f"Unsupported method '{method}'. Use one of: conv, convolutional, spectral, nonlinear"
    )
