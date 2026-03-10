"""Objective losses shared across surrogate model families."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LOSS_CHOICES = ("combined", "l2", "l1", "spectral_decay", "energy")


def normalize_loss_name(loss: str) -> str:
    """Normalize objective name and map aliases."""
    alias_map = {
        "combined": "combined",
        "combo": "combined",
        "default": "combined",
        "l2": "l2",
        "mse": "l2",
        "l1": "l1",
        "mae": "l1",
        "spectral_decay": "spectral_decay",
        "spectral-decay": "spectral_decay",
        "spectral": "spectral_decay",
        "energy": "energy",
    }
    normalized = str(loss).strip().lower().replace("-", "_")
    if normalized not in alias_map:
        supported = ", ".join(LOSS_CHOICES)
        raise ValueError(f"Unsupported loss '{loss}'. Use one of: {supported}")
    return alias_map[normalized]


class ObjectiveLoss:
    """Configurable objective loss combining spatial and spectral terms."""

    def __init__(self, nx: int, ny: int, device: torch.device, loss: str = "combined"):
        self.loss_name = normalize_loss_name(loss)
        self._spectral_bin_matrix, self._spectral_bin_counts = self._build_spectral_averager(nx, ny, device)

    @staticmethod
    def _build_spectral_averager(
        nx: int,
        ny: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute linear operators for radial spectral averaging."""
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.rfftfreq(ny) * ny
        k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        bins = np.floor(k_mag).astype(np.int64, copy=False)
        flat_bins = bins.reshape(-1)
        n_bins = int(flat_bins.max()) + 1

        n_coeff = int(flat_bins.size)
        bin_matrix = np.zeros((n_bins, n_coeff), dtype=np.float32)
        bin_matrix[flat_bins, np.arange(n_coeff)] = 1.0
        bin_counts = np.maximum(np.sum(bin_matrix, axis=1), 1.0).astype(np.float32, copy=False)

        matrix = torch.from_numpy(bin_matrix).to(device)
        counts = torch.from_numpy(bin_counts).to(device)
        return matrix, counts

    @staticmethod
    def _spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hat = torch.fft.rfft2(pred, dim=(-2, -1))
        target_hat = torch.fft.rfft2(target, dim=(-2, -1))
        return F.mse_loss(pred_hat.real, target_hat.real) + F.mse_loss(pred_hat.imag, target_hat.imag)

    @staticmethod
    def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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

    def _spectral_decay_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hat = torch.fft.rfft2(pred, dim=(-2, -1))
        target_hat = torch.fft.rfft2(target, dim=(-2, -1))
        reduce_dims = tuple(range(pred_hat.ndim - 2))
        pred_power = torch.mean(torch.abs(pred_hat) ** 2, dim=reduce_dims).to(torch.float32)
        target_power = torch.mean(torch.abs(target_hat) ** 2, dim=reduce_dims).to(torch.float32)

        pred_flat = pred_power.reshape(-1)
        target_flat = target_power.reshape(-1)

        pred_decay = (self._spectral_bin_matrix @ pred_flat) / self._spectral_bin_counts
        target_decay = (self._spectral_bin_matrix @ target_flat) / self._spectral_bin_counts
        return F.mse_loss(torch.log1p(pred_decay), torch.log1p(target_decay))

    @staticmethod
    def _energy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred - torch.roll(pred, shifts=1, dims=-2)
        pred_dy = pred - torch.roll(pred, shifts=1, dims=-1)
        target_dx = target - torch.roll(target, shifts=1, dims=-2)
        target_dy = target - torch.roll(target, shifts=1, dims=-1)

        pred_energy = torch.mean(pred * pred, dim=(-2, -1))
        target_energy = torch.mean(target * target, dim=(-2, -1))
        pred_grad_energy = torch.mean(pred_dx * pred_dx + pred_dy * pred_dy, dim=(-2, -1))
        target_grad_energy = torch.mean(target_dx * target_dx + target_dy * target_dy, dim=(-2, -1))

        return F.mse_loss(pred_energy, target_energy) + 0.5 * F.mse_loss(pred_grad_energy, target_grad_energy)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l2 = F.mse_loss(pred, target)

        if self.loss_name == "combined":
            return self._combined_loss(pred, target)
        if self.loss_name == "l2":
            return l2
        if self.loss_name == "l1":
            return F.l1_loss(pred, target)
        if self.loss_name == "spectral_decay":
            return l2 + 0.2 * self._spectral_decay_loss(pred, target)
        if self.loss_name == "energy":
            return l2 + 0.2 * self._energy_loss(pred, target)

        raise RuntimeError(f"Unknown loss objective '{self.loss_name}'")
