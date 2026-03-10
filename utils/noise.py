"""High-frequency spectral noise injection utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2


def add_hf_noise_2d(
    field: np.ndarray,
    noise_level: float,
    nx: int,
    ny: int,
    Lx: float | None = None,
    Ly: float | None = None,
    hf_fraction: float = 0.25,
) -> np.ndarray:
    """Inject Gaussian random noise only into the high-frequency Fourier band."""
    if Lx is None:
        Lx = 1.0
    if Ly is None:
        Ly = 1.0

    dx = float(Lx) / float(nx)
    dy = float(Ly) / float(ny)
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ky = fftfreq(ny, d=dy) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_nyq = min(np.pi / dx, np.pi / dy)
    k_min = hf_fraction * k_nyq
    k_max = k_nyq
    hf_mask = (k_mag >= k_min) & (k_mag <= k_max)

    # Build noise in physical space first; spectral masking then preserves Hermitian symmetry.
    base_noise = np.random.randn(nx, ny)
    base_noise_hat = fft2(base_noise)
    noise_hat = np.zeros((nx, ny), dtype=complex)
    noise_hat[hf_mask] = base_noise_hat[hf_mask]
    noise = np.real(ifft2(noise_hat))

    std = np.std(noise)
    if std > 1e-10:
        noise = noise / std * noise_level * np.std(field)

    return field + noise


def add_hf_noise_coupled(
    u: np.ndarray,
    v: np.ndarray,
    noise_level: float,
    nx: int,
    ny: int,
    Lx: float | None = None,
    Ly: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject independent high-frequency noise into both fields of a coupled system."""
    u_noisy = add_hf_noise_2d(u, noise_level, nx, ny, Lx=Lx, Ly=Ly)
    v_noisy = add_hf_noise_2d(v, noise_level, nx, ny, Lx=Lx, Ly=Ly)
    return np.clip(u_noisy, 0, 1), np.clip(v_noisy, 0, 1)
