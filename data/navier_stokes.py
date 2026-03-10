"""2D incompressible Navier-Stokes solver utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2

from utils.config import require_keys


@dataclass
class NSConfig:
    """Runtime configuration for Navier-Stokes experiments."""

    nx: int = 64
    ny: int = 64
    Lx: float = float(2 * np.pi)
    Ly: float = float(2 * np.pi)

    nu: float = 0.001

    t_final: float = 1.0
    n_snapshots: int = 20
    dt_max: float = 0.01

    n_train_trajectories: int = 8
    n_test_trajectories: int = 6

    noise_level: float = 0.04
    train_lr: float = 0.1
    train_iterations: int = 100
    train_batch_size: int = 32
    train_grad_clip: float = 1.0

    omega_1_frac: float = 1 / 16
    omega_2_frac: float = 1 / 6

    @classmethod
    def from_yaml(cls, config: Dict) -> "NSConfig":
        """Build NSConfig from YAML mapping."""
        grid = config.get("grid", {})
        physics = config.get("physics", {})
        time = config.get("time", {})
        data = config.get("data", {})
        training = config.get("training", {})
        rsd = config.get("rsd", {})

        require_keys(grid, "grid", ["nx", "ny", "Lx", "Ly"])
        require_keys(physics, "physics", ["nu"])
        require_keys(time, "time", ["t_final", "n_snapshots", "dt_max"])
        require_keys(data, "data", ["n_train_trajectories", "n_test_trajectories"])
        require_keys(training, "training", ["noise_level", "lr", "n_iter"])
        require_keys(rsd, "rsd", ["omega_1_frac", "omega_2_frac"])

        return cls(
            nx=int(grid["nx"]),
            ny=int(grid["ny"]),
            Lx=float(grid["Lx"]),
            Ly=float(grid["Ly"]),
            nu=float(physics["nu"]),
            t_final=float(time["t_final"]),
            n_snapshots=int(time["n_snapshots"]),
            dt_max=float(time["dt_max"]),
            n_train_trajectories=int(data["n_train_trajectories"]),
            n_test_trajectories=int(data["n_test_trajectories"]),
            noise_level=float(training["noise_level"]),
            train_lr=float(training["lr"]),
            train_iterations=int(training["n_iter"]),
            train_batch_size=int(training.get("batch_size", 32)),
            train_grad_clip=float(training.get("grad_clip", 1.0)),
            omega_1_frac=float(rsd["omega_1_frac"]),
            omega_2_frac=float(rsd["omega_2_frac"]),
        )


class NavierStokes2D:
    """Pseudo-spectral solver for 2D incompressible Navier-Stokes (vorticity form)."""

    def __init__(self, config: NSConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.nu = config.nu

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.kx = fftfreq(self.nx, d=self.dx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0

        kx_cut = self.nx // 3
        ky_cut = self.ny // 3
        self.dealias_mask = (np.abs(self.KX) < kx_cut * 2 * np.pi / self.Lx) & (
            np.abs(self.KY) < ky_cut * 2 * np.pi / self.Ly
        )

        self.inv_laplacian = -1.0 / self.K2
        self.inv_laplacian[0, 0] = 0.0

    def vorticity_to_velocity(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity field from vorticity via streamfunction inversion."""
        omega_hat = fft2(omega)
        psi_hat = self.inv_laplacian * omega_hat

        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat

        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        return u, v

    def compute_nonlinear_term(self, omega: np.ndarray) -> np.ndarray:
        """Compute advection term (u·∇)ω using pseudo-spectral gradients."""
        u, v = self.vorticity_to_velocity(omega)

        omega_hat = fft2(omega)
        domega_dx = np.real(ifft2(1j * self.KX * omega_hat))
        domega_dy = np.real(ifft2(1j * self.KY * omega_hat))

        advection = u * domega_dx + v * domega_dy

        advection_hat = fft2(advection)
        advection_hat *= self.dealias_mask
        return np.real(ifft2(advection_hat))

    def compute_rhs(self, omega: np.ndarray) -> np.ndarray:
        """RHS: - (u·∇)ω + ν∇²ω."""
        omega_hat = fft2(omega)
        diffusion = np.real(ifft2(-self.nu * self.K2 * omega_hat))
        advection = self.compute_nonlinear_term(omega)
        return -advection + diffusion

    def rk4_step(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """Advance one RK4 step."""
        k1 = self.compute_rhs(omega)
        k2 = self.compute_rhs(omega + 0.5 * dt * k1)
        k3 = self.compute_rhs(omega + 0.5 * dt * k2)
        k4 = self.compute_rhs(omega + dt * k3)
        return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(self, omega0: np.ndarray, t_final: float, n_snapshots: int) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate from omega0 to t_final and save n_snapshots states."""
        t_save = np.linspace(0, t_final, n_snapshots)
        omega_traj = np.zeros((n_snapshots, self.nx, self.ny))
        omega_traj[0] = omega0.copy()

        omega = omega0.copy()
        t = 0.0
        save_idx = 1

        dt = min(self.config.dt_max, 0.1 * self.dx / (np.max(np.abs(omega)) + 1e-10))

        while save_idx < n_snapshots:
            if t + dt > t_save[save_idx]:
                dt = t_save[save_idx] - t

            omega = self.rk4_step(omega, dt)
            t += dt

            if t >= t_save[save_idx] - 1e-10:
                omega_traj[save_idx] = omega.copy()
                save_idx += 1

            u, v = self.vorticity_to_velocity(omega)
            max_vel = max(np.max(np.abs(u)), np.max(np.abs(v))) + 1e-10
            dt = min(self.config.dt_max, 0.25 * self.dx / max_vel)

        return t_save, omega_traj

    def random_initial_condition(
        self,
        seed: Optional[int] = None,
        n_modes: int = 4,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Random smooth low-wavenumber initial vorticity."""
        if seed is not None:
            np.random.seed(seed)

        omega = np.zeros((self.nx, self.ny))
        for _ in range(n_modes):
            kx = np.random.randint(1, 4)
            ky = np.random.randint(1, 4)
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            amp = amplitude * np.random.randn() / np.sqrt(kx**2 + ky**2)
            omega += amp * np.sin(kx * self.X + phase_x) * np.sin(ky * self.Y + phase_y)
        return omega

    def taylor_green_vortex(self, amplitude: float = 1.0) -> np.ndarray:
        """Taylor-Green vortex initial condition."""
        return 2 * amplitude * np.cos(self.X) * np.cos(self.Y)

    def double_shear_layer(self, delta: float = 0.05, amplitude: float = 0.05) -> np.ndarray:
        """Double shear layer initial condition."""
        y_norm = self.Y / self.Ly
        u = np.where(
            y_norm < 0.5,
            np.tanh((y_norm - 0.25) / delta),
            np.tanh((0.75 - y_norm) / delta),
        )
        v = amplitude * np.sin(2 * np.pi * self.X / self.Lx)

        u_hat = fft2(u)
        v_hat = fft2(v)
        omega_hat = 1j * self.KX * v_hat - 1j * self.KY * u_hat
        return np.real(ifft2(omega_hat))
