"""High-frequency spectral noise injection utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fft import fftfreq, ifft2


def add_hf_noise_2d(
    field: np.ndarray,
    noise_level: float,
    nx: int,
    ny: int,
    hf_fraction: float = 0.25,
) -> np.ndarray:
    """Inject Gaussian random noise only into the high-frequency Fourier band."""
    kx = fftfreq(nx, d=1.0 / nx)
    ky = fftfreq(ny, d=1.0 / ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_min = hf_fraction * min(nx, ny) / 2
    k_max = min(nx, ny) / 2
    hf_mask = (k_mag >= k_min) & (k_mag <= k_max)

    noise_hat = np.zeros((nx, ny), dtype=complex)
    phases = np.random.uniform(0, 2 * np.pi, (nx, ny))
    amplitudes = np.random.randn(nx, ny)
    noise_hat[hf_mask] = amplitudes[hf_mask] * np.exp(1j * phases[hf_mask])

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject independent high-frequency noise into both fields of a coupled system."""
    u_noisy = add_hf_noise_2d(u, noise_level, nx, ny)
    v_noisy = add_hf_noise_2d(v, noise_level, nx, ny)
    return np.clip(u_noisy, 0, 1), np.clip(v_noisy, 0, 1)
