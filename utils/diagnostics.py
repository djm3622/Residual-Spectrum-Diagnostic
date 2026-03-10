"""Residual-spectrum diagnostics for Navier-Stokes and Gray-Scott runs."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.fft import dctn, fft2, fftfreq, ifft2

from data.navier_stokes import NSConfig, NavierStokes2D
from data.reaction_diffusion import GrayScottConfig

BASIS_CHOICES = ("fourier", "laplace", "wavelet", "svd")


def normalize_basis_name(basis: str) -> str:
    """Normalize basis name and map aliases."""
    alias_map = {
        "fourier": "fourier",
        "fft": "fourier",
        "laplace": "laplace",
        "laplacian": "laplace",
        "laplace_eigenbasis": "laplace",
        "laplace-eigenbasis": "laplace",
        "wavelet": "wavelet",
        "haar": "wavelet",
        "svd": "svd",
    }
    normalized = str(basis).strip().lower().replace("-", "_")
    if normalized not in alias_map:
        supported = ", ".join(BASIS_CHOICES)
        raise ValueError(f"Unsupported basis '{basis}'. Use one of: {supported}")
    return alias_map[normalized]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + 1e-10))


def _as_batch(fields: np.ndarray) -> np.ndarray:
    arr = np.asarray(fields, dtype=np.float64)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected residual field to be 2D or 3D, got shape {arr.shape}")


def _max_haar_levels(nx: int, ny: int) -> int:
    levels = 0
    cur_x = int(nx)
    cur_y = int(ny)
    while cur_x >= 2 and cur_y >= 2 and cur_x % 2 == 0 and cur_y % 2 == 0:
        levels += 1
        cur_x //= 2
        cur_y //= 2
    return levels


def _haar2d_coefficients(field: np.ndarray, max_levels: int) -> np.ndarray:
    """Orthonormal 2D Haar decomposition packed into one coefficient array."""
    coeffs = np.asarray(field, dtype=np.float64).copy()
    cur_x, cur_y = coeffs.shape

    for _ in range(max_levels):
        if cur_x < 2 or cur_y < 2 or cur_x % 2 != 0 or cur_y % 2 != 0:
            break

        block = coeffs[:cur_x, :cur_y].copy()
        low_rows = (block[0::2, :] + block[1::2, :]) / np.sqrt(2.0)
        high_rows = (block[0::2, :] - block[1::2, :]) / np.sqrt(2.0)

        ll = (low_rows[:, 0::2] + low_rows[:, 1::2]) / np.sqrt(2.0)
        lh = (low_rows[:, 0::2] - low_rows[:, 1::2]) / np.sqrt(2.0)
        hl = (high_rows[:, 0::2] + high_rows[:, 1::2]) / np.sqrt(2.0)
        hh = (high_rows[:, 0::2] - high_rows[:, 1::2]) / np.sqrt(2.0)

        half_x = cur_x // 2
        half_y = cur_y // 2

        coeffs[:half_x, :half_y] = ll
        coeffs[:half_x, half_y:cur_y] = lh
        coeffs[half_x:cur_x, :half_y] = hl
        coeffs[half_x:cur_x, half_y:cur_y] = hh

        cur_x = half_x
        cur_y = half_y

    return coeffs


def _build_wavelet_scores(nx: int, ny: int, max_levels: int) -> np.ndarray:
    """Assign frequency-like scales to packed Haar coefficients."""
    scores = np.ones((nx, ny), dtype=np.float64)
    cur_x = int(nx)
    cur_y = int(ny)

    for level in range(max_levels):
        if cur_x < 2 or cur_y < 2 or cur_x % 2 != 0 or cur_y % 2 != 0:
            break

        half_x = cur_x // 2
        half_y = cur_y // 2
        # Finest details get largest score; coarser levels get smaller score.
        detail_scale = float(2 ** (max_levels - level))

        scores[:half_x, half_y:cur_y] = detail_scale
        scores[half_x:cur_x, :half_y] = detail_scale
        scores[half_x:cur_x, half_y:cur_y] = detail_scale

        cur_x = half_x
        cur_y = half_y

    scores[:cur_x, :cur_y] = 1.0
    return scores


class _BasisProjector2D:
    """Residual basis projection and HF/LF band accounting."""

    def __init__(
        self,
        basis: str,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        omega_1_frac: float,
        omega_2_frac: float,
        fourier_k_mag: np.ndarray,
        fourier_k_max: float,
    ):
        self.basis = normalize_basis_name(basis)
        self.nx = int(nx)
        self.ny = int(ny)
        self.omega_1_frac = float(omega_1_frac)
        self.omega_2_frac = float(omega_2_frac)

        if self.basis == "fourier":
            k_low = max(1.0, fourier_k_max * self.omega_1_frac)
            k_high = fourier_k_max * self.omega_2_frac
            self._low_mask = (fourier_k_mag >= 1.0) & (fourier_k_mag <= k_low)
            self._high_mask = (fourier_k_mag > k_high) & (fourier_k_mag <= fourier_k_max)
            self._total_mask = (fourier_k_mag >= 1.0) & (fourier_k_mag <= fourier_k_max)
            self._wavelet_levels = 0
            self._wavelet_scores = None
            self._svd_scores = None
        elif self.basis == "laplace":
            mode_x = np.arange(self.nx, dtype=np.float64)
            mode_y = np.arange(self.ny, dtype=np.float64)
            lambda_x = (2.0 - 2.0 * np.cos(np.pi * mode_x / self.nx)) / (dx * dx)
            lambda_y = (2.0 - 2.0 * np.cos(np.pi * mode_y / self.ny)) / (dy * dy)
            laplace_scale = np.sqrt(lambda_x[:, None] + lambda_y[None, :])

            positive = laplace_scale > 0.0
            nonzero = laplace_scale[positive]
            scale_max = float(np.max(nonzero)) if nonzero.size else 1.0
            low_cut = max(float(np.min(nonzero)) if nonzero.size else 0.0, scale_max * self.omega_1_frac)
            high_cut = scale_max * self.omega_2_frac

            self._low_mask = positive & (laplace_scale <= low_cut)
            self._high_mask = positive & (laplace_scale > high_cut)
            self._total_mask = positive
            self._wavelet_levels = 0
            self._wavelet_scores = None
            self._svd_scores = None
        elif self.basis == "wavelet":
            levels = _max_haar_levels(self.nx, self.ny)
            scores = _build_wavelet_scores(self.nx, self.ny, levels)
            positive = scores > 0.0
            nonzero = scores[positive]
            score_max = float(np.max(nonzero)) if nonzero.size else 1.0
            low_cut = max(float(np.min(nonzero)) if nonzero.size else 0.0, score_max * self.omega_1_frac)
            high_cut = score_max * self.omega_2_frac

            self._low_mask = positive & (scores <= low_cut)
            self._high_mask = positive & (scores > high_cut)
            self._total_mask = positive
            self._wavelet_levels = levels
            self._wavelet_scores = scores
            self._svd_scores = None
        elif self.basis == "svd":
            rank = min(self.nx, self.ny)
            scores = np.arange(1, rank + 1, dtype=np.float64)
            score_max = float(rank)
            low_cut = max(1.0, score_max * self.omega_1_frac)
            high_cut = score_max * self.omega_2_frac

            self._low_mask = scores <= low_cut
            self._high_mask = scores > high_cut
            self._total_mask = scores > 0.0
            self._wavelet_levels = 0
            self._wavelet_scores = None
            self._svd_scores = scores
        else:
            raise RuntimeError(f"Unhandled basis '{self.basis}'")

    def power(self, residuals: np.ndarray) -> np.ndarray:
        samples = _as_batch(residuals)

        if self.basis == "fourier":
            return np.mean(np.abs(fft2(samples, axes=(-2, -1))) ** 2, axis=0)

        if self.basis == "laplace":
            coeff = dctn(samples, type=2, norm="ortho", axes=(-2, -1))
            return np.mean(np.abs(coeff) ** 2, axis=0)

        if self.basis == "wavelet":
            power = np.zeros((self.nx, self.ny), dtype=np.float64)
            for sample in samples:
                coeff = _haar2d_coefficients(sample, self._wavelet_levels)
                power += coeff * coeff
            power /= max(samples.shape[0], 1)
            return power

        if self.basis == "svd":
            rank = int(self._svd_scores.size)
            power = np.zeros(rank, dtype=np.float64)
            for sample in samples:
                singular_values = np.linalg.svd(sample, full_matrices=False, compute_uv=False)
                power += singular_values * singular_values
            power /= max(samples.shape[0], 1)
            return power

        raise RuntimeError(f"Unhandled basis '{self.basis}'")

    def compute_hfv(self, residuals: np.ndarray) -> float:
        power = self.power(residuals)
        return self.compute_hfv_from_power(power)

    def compute_lfv(self, residuals: np.ndarray) -> float:
        power = self.power(residuals)
        return self.compute_lfv_from_power(power)

    def compute_hfv_from_power(self, power: np.ndarray) -> float:
        hf_energy = float(np.sum(power[self._high_mask]))
        total_energy = float(np.sum(power[self._total_mask]))
        return _safe_ratio(hf_energy, total_energy)

    def compute_lfv_from_power(self, power: np.ndarray) -> float:
        lf_energy = float(np.sum(power[self._low_mask]))
        total_energy = float(np.sum(power[self._total_mask]))
        return _safe_ratio(lf_energy, total_energy)


class NavierStokesRSDAnalyzer:
    """RSD metrics for 2D incompressible Navier-Stokes trajectories."""

    def __init__(self, config: NSConfig, basis: str = "fourier"):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.nu = config.nu
        self.basis = normalize_basis_name(basis)

        self.kx = fftfreq(self.nx, d=config.Lx / config.nx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=config.Ly / config.ny) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)

        dx = config.Lx / config.nx
        dy = config.Ly / config.ny
        k_nyq = min(np.pi / dx, np.pi / dy)

        self.projector = _BasisProjector2D(
            basis=self.basis,
            nx=self.nx,
            ny=self.ny,
            dx=dx,
            dy=dy,
            omega_1_frac=config.omega_1_frac,
            omega_2_frac=config.omega_2_frac,
            fourier_k_mag=self.K_mag,
            fourier_k_max=k_nyq,
        )

        self.ns_solver = NavierStokes2D(config)

    def compute_residual(self, omega_traj: np.ndarray, dt: float) -> np.ndarray:
        """Residual: r = ∂ω/∂t + (u·∇)ω - ν∇²ω - f."""
        residuals = []
        K2 = self.KX**2 + self.KY**2

        for t in range(1, omega_traj.shape[0] - 1):
            omega = omega_traj[t]
            omega_hat = fft2(omega)

            d_omega_dt = (omega_traj[t + 1] - omega_traj[t - 1]) / (2 * dt)
            advection = self.ns_solver.compute_nonlinear_term(omega)
            diffusion = np.real(ifft2(-self.nu * K2 * omega_hat))
            forcing = self.ns_solver.compute_forcing(float(t) * float(dt))

            residuals.append(d_omega_dt + advection - diffusion - forcing)

        return np.array(residuals)

    def compute_hfv(self, residuals: np.ndarray) -> float:
        """High-frequency violation ratio in residual spectral power."""
        return self.projector.compute_hfv(residuals)

    def compute_lfv(self, residuals: np.ndarray) -> float:
        """Low-frequency violation ratio in residual spectral power."""
        return self.projector.compute_lfv(residuals)

    def compute_metrics(self, omega_pred: np.ndarray, omega_true: np.ndarray, dt: float) -> Dict[str, float]:
        """Return L2, HFV, LFV, and average residual magnitude."""
        l2_error = float(np.linalg.norm(omega_pred - omega_true) / np.linalg.norm(omega_true))
        residuals = self.compute_residual(omega_pred, dt)
        hfv = self.compute_hfv(residuals)
        lfv = self.compute_lfv(residuals)
        residual_mag = float(np.mean(np.abs(residuals)))

        return {
            "l2_error": l2_error,
            "hfv": hfv,
            "lfv": lfv,
            "residual_mag": residual_mag,
        }


class ReactionDiffusionRSDAnalyzer:
    """RSD metrics for coupled Gray-Scott trajectories."""

    def __init__(self, config: GrayScottConfig, basis: str = "fourier"):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.basis = normalize_basis_name(basis)

        self.Du = config.Du
        self.Dv = config.Dv
        self.F = config.F
        self.k = config.k

        dx = config.Lx / config.nx
        dy = config.Ly / config.ny
        self.kx = fftfreq(self.nx, d=dx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)

        k_nyq = min(np.pi / dx, np.pi / dy)
        self.projector = _BasisProjector2D(
            basis=self.basis,
            nx=self.nx,
            ny=self.ny,
            dx=dx,
            dy=dy,
            omega_1_frac=config.omega_1_frac,
            omega_2_frac=config.omega_2_frac,
            fourier_k_mag=self.K_mag,
            fourier_k_max=k_nyq,
        )

    def compute_residual(self, u_traj: np.ndarray, v_traj: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Residuals for coupled Gray-Scott PDE."""
        r_u_list = []
        r_v_list = []

        for t in range(1, u_traj.shape[0] - 1):
            u = u_traj[t]
            v = v_traj[t]

            du_dt = (u_traj[t + 1] - u_traj[t - 1]) / (2 * dt)
            dv_dt = (v_traj[t + 1] - v_traj[t - 1]) / (2 * dt)

            u_hat = fft2(u)
            v_hat = fft2(v)
            lap_u = np.real(ifft2(-self.K2 * u_hat))
            lap_v = np.real(ifft2(-self.K2 * v_hat))

            uv2 = u * v * v
            react_u = -uv2 + self.F * (1 - u)
            react_v = uv2 - (self.F + self.k) * v

            r_u = du_dt - self.Du * lap_u - react_u
            r_v = dv_dt - self.Dv * lap_v - react_v

            r_u_list.append(r_u)
            r_v_list.append(r_v)

        return np.array(r_u_list), np.array(r_v_list)

    def _combined_power(self, r_u: np.ndarray, r_v: np.ndarray) -> np.ndarray:
        return self.projector.power(r_u) + self.projector.power(r_v)

    def compute_hfv(self, r_u: np.ndarray, r_v: np.ndarray) -> float:
        """High-frequency violation ratio in combined coupled residual spectrum."""
        power = self._combined_power(r_u, r_v)
        return self.projector.compute_hfv_from_power(power)

    def compute_lfv(self, r_u: np.ndarray, r_v: np.ndarray) -> float:
        """Low-frequency violation ratio in combined coupled residual spectrum."""
        power = self._combined_power(r_u, r_v)
        return self.projector.compute_lfv_from_power(power)

    def compute_metrics(
        self,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        u_true: np.ndarray,
        v_true: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        """Return coupled L2 and RSD metrics."""
        l2_u = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        l2_v = np.linalg.norm(v_pred - v_true) / np.linalg.norm(v_true)
        l2 = float(np.sqrt(l2_u**2 + l2_v**2))

        r_u, r_v = self.compute_residual(u_pred, v_pred, dt)
        hfv = self.compute_hfv(r_u, r_v)
        lfv = self.compute_lfv(r_u, r_v)

        return {
            "l2_error": l2,
            "l2_u": float(l2_u),
            "l2_v": float(l2_v),
            "hfv": hfv,
            "lfv": lfv,
        }
