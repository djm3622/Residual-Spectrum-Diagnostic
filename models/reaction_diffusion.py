"""Surrogate models for coupled 2D Gray-Scott trajectories."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.reaction_diffusion import GrayScottConfig, GrayScottSolver
from models.losses import LOSS_CHOICES, ObjectiveLoss, normalize_loss_name
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
        trajectory_u: List[np.ndarray] | None = None,
        trajectory_v: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
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


class _UNetResBlock(nn.Module):
    """Residual block with periodic convolutions and wider kernels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode="circular",
        )
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.gelu(residual + y)


class _RDUNetOneStepNet(nn.Module):
    """U-Net style one-step map with physics features and bounded update scale."""

    def __init__(self, width: int = 48):
        super().__init__()
        in_channels = 8
        w1 = int(width)
        w2 = int(width * 2)
        w3 = int(width * 4)

        self.enc0 = _UNetResBlock(in_channels, w1, kernel_size=5)
        self.enc1 = _UNetResBlock(w1, w2, kernel_size=5)
        self.enc2 = _UNetResBlock(w2, w3, kernel_size=5)
        self.bottleneck = nn.Sequential(
            _UNetResBlock(w3, w3, kernel_size=5),
            _UNetResBlock(w3, w3, kernel_size=5),
        )
        self.dec1 = _UNetResBlock(w3 + w2, w2, kernel_size=5)
        self.dec0 = _UNetResBlock(w2 + w1, w1, kernel_size=5)
        self.spatial_out = nn.Conv2d(w1, 2, kernel_size=1)

        self.reaction = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(w1, w1, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(w1, 2, kernel_size=1),
        )

        self.diff_raw = nn.Parameter(torch.tensor([0.06, 0.06], dtype=torch.float32))
        self.step_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.step_max = 0.35

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

    @staticmethod
    def _downsample(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    @staticmethod
    def _upsample(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0:1]
        v = state[:, 1:2]
        lap_u = self._laplacian_periodic(u)
        lap_v = self._laplacian_periodic(v)
        uv = u * v
        uv2 = uv * v
        features = torch.cat([u, v, lap_u, lap_v, uv, u * u, v * v, uv2], dim=1)

        e0 = self.enc0(features)
        e1 = self.enc1(self._downsample(e0))
        e2 = self.enc2(self._downsample(e1))
        b = self.bottleneck(e2)

        d1 = self._upsample(b, out_hw=(e1.shape[-2], e1.shape[-1]))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self._upsample(d1, out_hw=(e0.shape[-2], e0.shape[-1]))
        d0 = self.dec0(torch.cat([d0, e0], dim=1))

        spatial_delta = self.spatial_out(d0)
        reaction_delta = self.reaction(features)
        diff_delta = F.softplus(self.diff_raw).view(1, 2, 1, 1) * torch.cat([lap_u, lap_v], dim=1)

        step = self.step_max * torch.sigmoid(self.step_raw)
        delta = reaction_delta + diff_delta + 0.5 * spatial_delta
        next_state = state + step * delta
        return torch.clamp(next_state, 0.0, 1.0)


class ConvolutionalSurrogate2DCoupled:
    """Nonlinear one-step periodic residual integrator for Gray-Scott dynamics."""

    def __init__(
        self,
        nx: int,
        ny: int,
        seed: int | None = None,
        device: str = "auto",
        loss: str = "combined",
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

        self.net = _RDUNetOneStepNet().to(self.device)
        self.net.eval()

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
        trajectory_u: List[np.ndarray] | None = None,
        trajectory_v: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
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
        # Balance u/v contributions in the objective: u is O(1) while v is often O(1e-2).
        channel_scale = torch.sqrt(torch.mean(y_train * y_train, dim=(0, 2, 3), keepdim=True))
        channel_scale = torch.clamp(channel_scale, min=1e-6)

        n_samples = x_train.shape[0]
        batch = max(1, min(int(batch_size), n_samples))
        clip = max(float(grad_clip), 1e-8)
        horizon = max(1, int(rollout_horizon))
        rollout_w = max(0.0, float(rollout_weight))

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
        else:
            seq_u = None
            seq_v = None
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
                    pred_scaled = pred / channel_scale
                    target_scaled = yb / channel_scale
                    one_step_loss = self.objective_loss(pred_scaled, target_scaled)

                    rollout_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                    if use_rollout and seq_u is not None and seq_v is not None and rollout_batch > 0:
                        traj_idx = torch.randint(0, seq_u.shape[0], (rollout_batch,), device=self.device)
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
                                pred_roll / channel_scale,
                                tgt / channel_scale,
                            )
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

        return _sanitize_species(u_next.astype(np.float32, copy=False)), _sanitize_species(
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
        trajectory_u: List[np.ndarray] | None = None,
        trajectory_v: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
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
    loss: str = "combined",
    config: Optional[GrayScottConfig] = None,
    snapshot_dt: Optional[float] = None,
) -> CoupledOneStepModel:
    """Factory for coupled RD surrogate models selected by CLI method arg."""
    normalized = method.strip().lower()

    if normalized in {"conv_nn", "conv_legacy", "convolutional_legacy", "nn"}:
        return ConvolutionalSurrogate2DCoupled(nx, ny, seed=seed, device=device, loss=loss)
    if normalized in {"physics", "gray_scott", "grayscott", "rd_physics"}:
        if config is None or snapshot_dt is None:
            raise ValueError("Physics RD model requires GrayScottConfig and snapshot_dt.")
        return PhysicsConsistentSurrogate2DCoupled(config=config, snapshot_dt=snapshot_dt, device=device)
    if normalized in {"conv", "convolutional", "spectral", "nonlinear"}:
        return ConvolutionalSurrogate2DCoupled(nx, ny, seed=seed, device=device, loss=loss)

    raise ValueError(
        "Unsupported method "
        f"'{method}'. Use one of: conv, convolutional, spectral, nonlinear, conv_nn, physics"
    )
