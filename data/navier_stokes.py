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
    forcing_type: str = "none"
    forcing_amplitude: float = 0.0
    forcing_kx: int = 1
    forcing_ky: int = 1
    forcing_omega: float = 0.0
    forcing_phase: float = 0.0

    t_final: float = 1.0
    n_snapshots: int = 20
    dt_max: float = 0.01

    n_train_trajectories: int = 8
    n_test_trajectories: int = 6
    initial_condition: str = "random"
    initial_target_rms: float = 0.0
    initial_n_modes: int = 4
    initial_kmax: int = 3
    line_axis: str = "diagonal_pos"
    line_amplitude: float = 1.0
    template_seed: int = 111
    template_n_modes: int = 4
    template_amplitude: float = 1.0
    template_shift_step_x: int = 5
    template_shift_step_y: int = 10
    template_shift_offset: int = 0

    noise_level: float = 0.04
    train_lr: float = 0.1
    train_iterations: int = 100
    train_batch_size: int = 32
    train_grad_clip: float = 1.0
    train_weight_decay: float = 0.0
    train_use_one_cycle_lr: bool = False
    train_one_cycle_pct_start: float = 0.3
    train_one_cycle_div_factor: float = 25.0
    train_one_cycle_final_div_factor: float = 10000.0
    train_rollout_horizon: int = 4
    train_rollout_weight: float = 0.5
    train_validation_fraction: float = 0.1
    train_checkpoint_every_epochs: int = 20
    train_model_width: int = 64
    train_model_depth: int = 5

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

        forcing = physics.get("forcing", {})
        forcing_type = str(forcing.get("type", "none")).strip().lower()

        return cls(
            nx=int(grid["nx"]),
            ny=int(grid["ny"]),
            Lx=float(grid["Lx"]),
            Ly=float(grid["Ly"]),
            nu=float(physics["nu"]),
            forcing_type=forcing_type,
            forcing_amplitude=float(forcing.get("amplitude", 0.0)),
            forcing_kx=int(forcing.get("kx", 1)),
            forcing_ky=int(forcing.get("ky", 1)),
            forcing_omega=float(forcing.get("omega", 0.0)),
            forcing_phase=float(forcing.get("phase", 0.0)),
            t_final=float(time["t_final"]),
            n_snapshots=int(time["n_snapshots"]),
            dt_max=float(time["dt_max"]),
            n_train_trajectories=int(data["n_train_trajectories"]),
            n_test_trajectories=int(data["n_test_trajectories"]),
            initial_condition=str(data.get("initial_condition", "random")),
            initial_target_rms=float(data.get("initial_target_rms", 0.0)),
            initial_n_modes=int(data.get("initial_n_modes", 4)),
            initial_kmax=int(data.get("initial_kmax", 3)),
            line_axis=str(data.get("line_axis", "y")),
            line_amplitude=float(data.get("line_amplitude", 1.0)),
            template_seed=int(data.get("template_seed", 111)),
            template_n_modes=int(data.get("template_n_modes", 4)),
            template_amplitude=float(data.get("template_amplitude", 1.0)),
            template_shift_step_x=int(data.get("template_shift_step_x", 5)),
            template_shift_step_y=int(data.get("template_shift_step_y", 10)),
            template_shift_offset=int(data.get("template_shift_offset", 0)),
            noise_level=float(training["noise_level"]),
            train_lr=float(training["lr"]),
            train_iterations=int(training["n_iter"]),
            train_batch_size=int(training.get("batch_size", 32)),
            train_grad_clip=float(training.get("grad_clip", 1.0)),
            train_weight_decay=float(training.get("weight_decay", 0.0)),
            train_use_one_cycle_lr=bool(training.get("use_one_cycle_lr", False)),
            train_one_cycle_pct_start=float(training.get("one_cycle_pct_start", 0.3)),
            train_one_cycle_div_factor=float(training.get("one_cycle_div_factor", 25.0)),
            train_one_cycle_final_div_factor=float(training.get("one_cycle_final_div_factor", 10000.0)),
            train_rollout_horizon=int(training.get("rollout_horizon", 4)),
            train_rollout_weight=float(training.get("rollout_weight", 0.5)),
            train_validation_fraction=float(training.get("validation_fraction", 0.1)),
            train_checkpoint_every_epochs=int(training.get("checkpoint_every_epochs", 20)),
            train_model_width=int(training.get("model_width", 64)),
            train_model_depth=int(training.get("model_depth", 5)),
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
        self.forcing_type = str(config.forcing_type).strip().lower()
        self.forcing_amplitude = float(config.forcing_amplitude)
        self.forcing_kx = int(config.forcing_kx)
        self.forcing_ky = int(config.forcing_ky)
        self.forcing_omega = float(config.forcing_omega)
        self.forcing_phase = float(config.forcing_phase)
        self.initial_condition = str(config.initial_condition).strip().lower().replace("-", "_")
        self.initial_target_rms = float(config.initial_target_rms)
        self.initial_n_modes = int(config.initial_n_modes)
        self.initial_kmax = int(config.initial_kmax)
        self.line_axis = str(config.line_axis).strip().lower()
        self.line_amplitude = float(config.line_amplitude)
        self.template_seed = int(config.template_seed)
        self.template_n_modes = int(config.template_n_modes)
        self.template_amplitude = float(config.template_amplitude)
        self.template_shift_step_x = int(config.template_shift_step_x)
        self.template_shift_step_y = int(config.template_shift_step_y)
        self.template_shift_offset = int(config.template_shift_offset)
        self._template_omega0_cache: Optional[np.ndarray] = None

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

    def compute_forcing(self, t: float) -> np.ndarray:
        """Configured vorticity forcing term f(x, y, t)."""
        if self.forcing_type in {"none", "off", ""} or self.forcing_amplitude == 0.0:
            return np.zeros((self.nx, self.ny), dtype=float)

        if self.forcing_type in {"sinusoidal", "kolmogorov"}:
            phase = (
                self.forcing_kx * self.X
                + self.forcing_ky * self.Y
                + self.forcing_omega * t
                + self.forcing_phase
            )
            return self.forcing_amplitude * np.sin(phase)

        raise ValueError(f"Unsupported forcing_type '{self.forcing_type}'")

    def compute_rhs(self, omega: np.ndarray, t: float = 0.0) -> np.ndarray:
        """RHS: - (u·∇)ω + ν∇²ω + f."""
        omega_hat = fft2(omega)
        diffusion = np.real(ifft2(-self.nu * self.K2 * omega_hat))
        advection = self.compute_nonlinear_term(omega)
        forcing = self.compute_forcing(t)
        return -advection + diffusion + forcing

    def rk4_step(self, omega: np.ndarray, dt: float, t: float) -> np.ndarray:
        """Advance one RK4 step."""
        k1 = self.compute_rhs(omega, t=t)
        k2 = self.compute_rhs(omega + 0.5 * dt * k1, t=t + 0.5 * dt)
        k3 = self.compute_rhs(omega + 0.5 * dt * k2, t=t + 0.5 * dt)
        k4 = self.compute_rhs(omega + dt * k3, t=t + dt)
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

            omega = self.rk4_step(omega, dt, t)
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
        kmax: int = 3,
    ) -> np.ndarray:
        """Random smooth low-wavenumber initial vorticity."""
        if seed is not None:
            np.random.seed(seed)

        omega = np.zeros((self.nx, self.ny))
        k_hi = max(1, int(kmax))
        for _ in range(n_modes):
            kx = np.random.randint(1, k_hi + 1)
            ky = np.random.randint(1, k_hi + 1)
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            amp = amplitude * np.random.randn() / np.sqrt(kx**2 + ky**2)
            omega += amp * np.sin(kx * self.X + phase_x) * np.sin(ky * self.Y + phase_y)
        return omega

    def _periodic_deterministic_initial_condition(
        self,
        index: int,
        modes: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (2, 2)),
    ) -> np.ndarray:
        """Deterministic periodic initial condition family indexed by trajectory id."""
        omega = np.zeros((self.nx, self.ny), dtype=float)
        for mode_idx, (kx, ky) in enumerate(modes):
            phase_x = 2.0 * np.pi * ((index + 1) * (mode_idx + 2) * 0.137) % (2.0 * np.pi)
            phase_y = 2.0 * np.pi * ((index + 1) * (mode_idx + 3) * 0.173) % (2.0 * np.pi)
            amp = (1.0 / np.sqrt(kx * kx + ky * ky)) * (1.0 + 0.18 * np.sin((index + 1) * (mode_idx + 1)))
            omega += amp * np.sin(kx * self.X + phase_x) * np.cos(ky * self.Y + phase_y)
        return omega - np.mean(omega)

    def _line_wave_initial_condition(self, index: int) -> np.ndarray:
        """1D sinusoidal shear-like ICs (no dot/checkerboard structures)."""
        phase = 2.0 * np.pi * ((index + 1) * 0.173)
        amp = float(self.line_amplitude)
        axis = self.line_axis
        if axis == "x":
            omega0 = amp * np.sin(self.X + phase)
        elif axis in {"diag", "diagonal_pos", "xy"}:
            omega0 = amp * np.sin(self.X + self.Y + phase)
        elif axis in {"diagonal_neg", "x_minus_y"}:
            omega0 = amp * np.sin(self.X - self.Y + phase)
        else:
            omega0 = amp * np.sin(self.Y + phase)
        return omega0 - np.mean(omega0)

    def _load_template_initial_condition(self) -> np.ndarray:
        """Build a deterministic template omega0 directly from hardcoded settings."""
        if self._template_omega0_cache is not None:
            return self._template_omega0_cache

        self._template_omega0_cache = self.random_initial_condition(
            seed=self.template_seed,
            n_modes=max(1, self.template_n_modes),
            amplitude=float(self.template_amplitude),
        )
        return self._template_omega0_cache

    def sample_initial_condition(self, seed: Optional[int], index: int = 0) -> np.ndarray:
        """Generate initial condition according to config-driven family."""
        mode = self.initial_condition
        if mode == "random":
            omega0 = self.random_initial_condition(
                seed=seed,
                n_modes=max(1, self.initial_n_modes),
                kmax=max(1, self.initial_kmax),
            )
        elif mode == "taylor_green":
            omega0 = self.taylor_green_vortex()
        elif mode == "double_shear":
            omega0 = self.double_shear_layer()
        elif mode == "periodic_deterministic":
            omega0 = self._periodic_deterministic_initial_condition(index=index)
        elif mode == "line_wave":
            omega0 = self._line_wave_initial_condition(index=index)
        elif mode == "template_shifted":
            template = self._load_template_initial_condition()
            sx = (self.template_shift_offset + index * self.template_shift_step_x) % self.nx
            sy = (self.template_shift_offset + index * self.template_shift_step_y) % self.ny
            omega0 = np.roll(np.roll(template, sx, axis=0), sy, axis=1)
        else:
            raise ValueError(
                "Unsupported data.initial_condition for Navier-Stokes. "
                "Use one of: random, taylor_green, double_shear, periodic_deterministic, line_wave, template_shifted."
            )

        if self.initial_target_rms > 0.0:
            rms0 = float(np.sqrt(np.mean(omega0**2)))
            omega0 = omega0 * (self.initial_target_rms / max(rms0, 1e-12))
        return omega0

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
