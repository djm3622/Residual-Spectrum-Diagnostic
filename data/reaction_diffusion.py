"""2D Gray-Scott reaction-diffusion solver utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2

from utils.config import require_keys


@dataclass
class GrayScottConfig:
    """Runtime configuration for Gray-Scott experiments."""

    nx: int = 64
    ny: int = 64
    Lx: float = 2.5
    Ly: float = 2.5

    Du: float = 0.16
    Dv: float = 0.08
    F: float = 0.035
    k: float = 0.065

    t_final: float = 1000.0
    n_snapshots: int = 15
    dt: float = 1.0

    n_train_trajectories: int = 5
    n_test_trajectories: int = 3

    noise_level: float = 0.03
    train_lr: float = 0.05
    train_iterations: int = 100
    train_batch_size: int = 32
    train_grad_clip: float = 1.0

    omega_1_frac: float = 1 / 16
    omega_2_frac: float = 1 / 6

    @classmethod
    def from_yaml(cls, config: Dict) -> "GrayScottConfig":
        """Build GrayScottConfig from YAML mapping."""
        grid = config.get("grid", {})
        physics = config.get("physics", {})
        time = config.get("time", {})
        data = config.get("data", {})
        training = config.get("training", {})
        rsd = config.get("rsd", {})

        require_keys(grid, "grid", ["nx", "ny", "Lx", "Ly"])
        require_keys(physics, "physics", ["Du", "Dv", "F", "k"])
        require_keys(time, "time", ["t_final", "n_snapshots", "dt"])
        require_keys(data, "data", ["n_train_trajectories", "n_test_trajectories"])
        require_keys(training, "training", ["noise_level", "lr", "n_iter"])
        require_keys(rsd, "rsd", ["omega_1_frac", "omega_2_frac"])

        return cls(
            nx=int(grid["nx"]),
            ny=int(grid["ny"]),
            Lx=float(grid["Lx"]),
            Ly=float(grid["Ly"]),
            Du=float(physics["Du"]),
            Dv=float(physics["Dv"]),
            F=float(physics["F"]),
            k=float(physics["k"]),
            t_final=float(time["t_final"]),
            n_snapshots=int(time["n_snapshots"]),
            dt=float(time["dt"]),
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


class GrayScottSolver:
    """Spectral Gray-Scott solver with operator splitting."""

    def __init__(self, config: GrayScottConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.Lx = config.Lx
        self.Ly = config.Ly

        self.Du = config.Du
        self.Dv = config.Dv
        self.F = config.F
        self.k = config.k
        self.dt = config.dt

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.kx = fftfreq(self.nx, d=self.dx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2

        self.diffusion_u = np.exp(-self.Du * self.K2 * self.dt)
        self.diffusion_v = np.exp(-self.Dv * self.K2 * self.dt)

    def reaction_terms(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reaction terms for Gray-Scott model."""
        uv2 = u * v * v
        du = -uv2 + self.F * (1 - u)
        dv = uv2 - (self.F + self.k) * v
        return du, dv

    def step(self, u: np.ndarray, v: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Single operator-split timestep."""
        if dt is None:
            dt = self.dt

        diffusion_u = np.exp(-self.Du * self.K2 * dt)
        diffusion_v = np.exp(-self.Dv * self.K2 * dt)

        u_hat = fft2(u)
        v_hat = fft2(v)

        u = np.real(ifft2(u_hat * diffusion_u))
        v = np.real(ifft2(v_hat * diffusion_v))

        du, dv = self.reaction_terms(u, v)
        u = np.clip(u + dt * du, 0, 1)
        v = np.clip(v + dt * dv, 0, 1)

        return u, v

    def solve(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        t_final: float,
        n_snapshots: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate from (u0, v0) and return snapshot trajectories."""
        t_save = np.linspace(0.0, t_final, n_snapshots)
        u_traj = np.zeros((n_snapshots, self.nx, self.ny))
        v_traj = np.zeros((n_snapshots, self.nx, self.ny))

        u, v = u0.copy(), v0.copy()
        u_traj[0] = u
        v_traj[0] = v
        t = 0.0
        save_idx = 1

        while save_idx < n_snapshots:
            dt_step = min(self.dt, t_save[save_idx] - t)
            u, v = self.step(u, v, dt=dt_step)
            t += dt_step

            if t >= t_save[save_idx] - 1e-12:
                u_traj[save_idx] = u
                v_traj[save_idx] = v
                save_idx += 1

        return t_save, u_traj, v_traj

    def initial_condition_random_seeds(
        self,
        n_seeds: int = 20,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Background with random circular perturbation seeds."""
        if seed is not None:
            np.random.seed(seed)

        u = np.ones((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))

        for _ in range(n_seeds):
            cx = np.random.randint(self.nx // 4, 3 * self.nx // 4)
            cy = np.random.randint(self.ny // 4, 3 * self.ny // 4)
            radius = np.random.randint(2, 5)

            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i * i + j * j <= radius * radius:
                        xi = (cx + i) % self.nx
                        yj = (cy + j) % self.ny
                        u[xi, yj] = 0.5 + np.random.uniform(-0.1, 0.1)
                        v[xi, yj] = 0.25 + np.random.uniform(-0.1, 0.1)

        return u, v

    def initial_condition_center_square(self, size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Centered square perturbation initial condition."""
        u = np.ones((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))

        cx, cy = self.nx // 2, self.ny // 2
        half = size // 2

        u[cx - half : cx + half, cy - half : cy + half] = 0.5
        v[cx - half : cx + half, cy - half : cy + half] = 0.25

        u += np.random.uniform(-0.01, 0.01, (self.nx, self.ny))
        v += np.random.uniform(-0.01, 0.01, (self.nx, self.ny))

        return np.clip(u, 0, 1), np.clip(v, 0, 1)
