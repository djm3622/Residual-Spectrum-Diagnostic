"""Residual-spectrum diagnostics for Navier-Stokes and Gray-Scott runs."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2

from data.navier_stokes import NSConfig, NavierStokes2D
from data.reaction_diffusion import GrayScottConfig


class NavierStokesRSDAnalyzer:
    """RSD metrics for 2D incompressible Navier-Stokes trajectories."""

    def __init__(self, config: NSConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.nu = config.nu

        self.kx = fftfreq(self.nx, d=config.Lx / config.nx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=config.Ly / config.ny) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)

        dx = config.Lx / config.nx
        dy = config.Ly / config.ny
        k_nyq = min(np.pi / dx, np.pi / dy)
        self.k_low = max(1, k_nyq * config.omega_1_frac)
        self.k_high = k_nyq * config.omega_2_frac
        self.k_max = k_nyq

        self.low_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_low)
        self.high_mask = (self.K_mag > self.k_high) & (self.K_mag <= self.k_max)
        self.total_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_max)

        self.ns_solver = NavierStokes2D(config)

    def compute_residual(self, omega_traj: np.ndarray, dt: float) -> np.ndarray:
        """Residual: r = ∂ω/∂t + (u·∇)ω - ν∇²ω."""
        residuals = []
        K2 = self.KX**2 + self.KY**2

        for t in range(1, omega_traj.shape[0] - 1):
            omega = omega_traj[t]
            omega_hat = fft2(omega)

            d_omega_dt = (omega_traj[t + 1] - omega_traj[t - 1]) / (2 * dt)
            advection = self.ns_solver.compute_nonlinear_term(omega)
            diffusion = np.real(ifft2(-self.nu * K2 * omega_hat))

            residuals.append(d_omega_dt + advection - diffusion)

        return np.array(residuals)

    def compute_hfv(self, residuals: np.ndarray) -> float:
        """High-frequency violation ratio in residual spectral power."""
        if residuals.ndim == 2:
            residuals = residuals[np.newaxis, :, :]

        power = np.mean(np.abs(fft2(residuals, axes=(-2, -1))) ** 2, axis=0)
        hf_energy = np.sum(power[self.high_mask])
        total_energy = np.sum(power[self.total_mask])
        return float(hf_energy / (total_energy + 1e-10))

    def compute_lfv(self, residuals: np.ndarray) -> float:
        """Low-frequency violation ratio in residual spectral power."""
        if residuals.ndim == 2:
            residuals = residuals[np.newaxis, :, :]

        power = np.mean(np.abs(fft2(residuals, axes=(-2, -1))) ** 2, axis=0)
        lf_energy = np.sum(power[self.low_mask])
        total_energy = np.sum(power[self.total_mask])
        return float(lf_energy / (total_energy + 1e-10))

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

    def __init__(self, config: GrayScottConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny

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
        self.k_low = max(1, k_nyq * config.omega_1_frac)
        self.k_high = k_nyq * config.omega_2_frac
        self.k_max = k_nyq

        self.low_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_low)
        self.high_mask = (self.K_mag > self.k_high) & (self.K_mag <= self.k_max)
        self.total_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_max)

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
        if r_u.ndim == 2:
            r_u = r_u[np.newaxis, :, :]
            r_v = r_v[np.newaxis, :, :]

        power_u = np.mean(np.abs(fft2(r_u, axes=(-2, -1))) ** 2, axis=0)
        power_v = np.mean(np.abs(fft2(r_v, axes=(-2, -1))) ** 2, axis=0)
        return power_u + power_v

    def compute_hfv(self, r_u: np.ndarray, r_v: np.ndarray) -> float:
        """High-frequency violation ratio in combined coupled residual spectrum."""
        power = self._combined_power(r_u, r_v)
        hf_energy = np.sum(power[self.high_mask])
        total_energy = np.sum(power[self.total_mask])
        return float(hf_energy / (total_energy + 1e-10))

    def compute_lfv(self, r_u: np.ndarray, r_v: np.ndarray) -> float:
        """Low-frequency violation ratio in combined coupled residual spectrum."""
        power = self._combined_power(r_u, r_v)
        lf_energy = np.sum(power[self.low_mask])
        total_energy = np.sum(power[self.total_mask])
        return float(lf_energy / (total_energy + 1e-10))

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
