#!/usr/bin/env python3
"""
RSD Case Study: 2D Reaction-Diffusion (Gray-Scott Model)
=========================================================

PDE System (Gray-Scott):

    ∂u/∂t = Du ∇²u - uv² + F(1-u)
    ∂v/∂t = Dv ∇²v + uv² - (F+k)v

where:
    - u, v: Chemical concentrations (two species)
    - Du, Dv: Diffusion coefficients
    - F: Feed rate (replenishes u)
    - k: Kill rate (removes v)

Physics:
    - Produces Turing patterns: spots, stripes, waves, spirals
    - Reaction-diffusion instability creates spatial structure
    - Different (F, k) values give different pattern regimes

This demonstrates RSD on:
    - Multi-component systems (2 coupled PDEs)
    - Pattern-forming dynamics
    - Reaction kinetics (not just transport/diffusion)
    - Biologically/chemically relevant models

Author: A. Baheri
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import laplace
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GrayScottConfig:
    """Configuration for Gray-Scott reaction-diffusion experiments."""
    # Grid
    nx: int = 128                   # Grid points in x
    ny: int = 128                   # Grid points in y
    Lx: float = 2.5                 # Domain size in x
    Ly: float = 2.5                 # Domain size in y
    
    # Physics parameters (Gray-Scott)
    Du: float = 0.16                # Diffusion coefficient for u
    Dv: float = 0.08                # Diffusion coefficient for v
    F: float = 0.035                # Feed rate
    k: float = 0.065                # Kill rate
    
    # Time integration
    t_final: float = 2000.0         # Final time (needs long integration for patterns)
    n_snapshots: int = 20           # Number of snapshots to save
    dt: float = 1.0                 # Timestep
    
    # Training
    n_train_trajectories: int = 6   # Number of training trajectories
    n_test_trajectories: int = 4    # Number of test trajectories
    noise_level: float = 0.03       # HF noise amplitude
    
    # RSD parameters
    omega_1_frac: float = 1/16      # Low-frequency boundary
    omega_2_frac: float = 1/6       # High-frequency boundary


# ============================================================================
# Pattern Regimes for Gray-Scott
# ============================================================================

PATTERN_REGIMES = {
    'spots': {'F': 0.035, 'k': 0.065},      # Mitosis-like spots
    'stripes': {'F': 0.025, 'k': 0.055},    # Labyrinthine stripes
    'waves': {'F': 0.014, 'k': 0.045},      # Moving waves
    'coral': {'F': 0.062, 'k': 0.063},      # Coral-like growth
    'chaos': {'F': 0.026, 'k': 0.051},      # Chaotic patterns
}


# ============================================================================
# 2D Gray-Scott Solver
# ============================================================================

class GrayScottSolver:
    """
    Solver for 2D Gray-Scott reaction-diffusion system.
    
    Uses spectral method for diffusion and explicit Euler for reaction.
    (Could use more sophisticated integrators for stiff systems)
    """
    
    def __init__(self, config: GrayScottConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.Lx = config.Lx
        self.Ly = config.Ly
        
        # Parameters
        self.Du = config.Du
        self.Dv = config.Dv
        self.F = config.F
        self.k = config.k
        self.dt = config.dt
        
        # Physical grid
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Wavenumbers for spectral diffusion
        self.kx = fftfreq(self.nx, d=self.dx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        
        # Precompute diffusion operators (for implicit diffusion)
        self.diffusion_u = np.exp(-self.Du * self.K2 * self.dt)
        self.diffusion_v = np.exp(-self.Dv * self.K2 * self.dt)
    
    def reaction_terms(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reaction terms.
        
        du/dt_reaction = -uv² + F(1-u)
        dv/dt_reaction = +uv² - (F+k)v
        """
        uv2 = u * v * v
        
        du = -uv2 + self.F * (1 - u)
        dv = uv2 - (self.F + self.k) * v
        
        return du, dv
    
    def step(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single timestep using operator splitting:
        1. Diffusion (spectral, exact)
        2. Reaction (explicit Euler)
        """
        # Diffusion step (spectral)
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        u = np.real(ifft2(u_hat * self.diffusion_u))
        v = np.real(ifft2(v_hat * self.diffusion_v))
        
        # Reaction step (explicit Euler)
        du, dv = self.reaction_terms(u, v)
        u = u + self.dt * du
        v = v + self.dt * dv
        
        # Clamp to valid range [0, 1]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        
        return u, v
    
    def solve(self, u0: np.ndarray, v0: np.ndarray, 
              t_final: float, n_snapshots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate from (u0, v0) to t_final, saving n_snapshots.
        
        Returns:
            t: Array of snapshot times
            u_traj: Array of shape (n_snapshots, nx, ny)
            v_traj: Array of shape (n_snapshots, nx, ny)
        """
        n_steps = int(t_final / self.dt)
        save_interval = max(1, n_steps // (n_snapshots - 1))
        
        t_save = []
        u_traj = []
        v_traj = []
        
        u, v = u0.copy(), v0.copy()
        
        for step in range(n_steps + 1):
            if step % save_interval == 0 and len(t_save) < n_snapshots:
                t_save.append(step * self.dt)
                u_traj.append(u.copy())
                v_traj.append(v.copy())
            
            if step < n_steps:
                u, v = self.step(u, v)
        
        return np.array(t_save), np.array(u_traj), np.array(v_traj)
    
    def initial_condition_random_seeds(self, n_seeds: int = 20, 
                                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize with random localized perturbations.
        
        Background: u=1, v=0 (stable state)
        Seeds: Small regions with u=0.5, v=0.25 (triggers pattern)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Uniform background
        u = np.ones((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        
        # Random seed locations
        for _ in range(n_seeds):
            cx = np.random.randint(self.nx // 4, 3 * self.nx // 4)
            cy = np.random.randint(self.ny // 4, 3 * self.ny // 4)
            radius = np.random.randint(2, 5)
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i*i + j*j <= radius*radius:
                        xi = (cx + i) % self.nx
                        yj = (cy + j) % self.ny
                        u[xi, yj] = 0.5 + np.random.uniform(-0.1, 0.1)
                        v[xi, yj] = 0.25 + np.random.uniform(-0.1, 0.1)
        
        return u, v
    
    def initial_condition_center_square(self, size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize with a square perturbation in the center.
        
        Classic setup for observing pattern formation.
        """
        u = np.ones((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        
        cx, cy = self.nx // 2, self.ny // 2
        half = size // 2
        
        u[cx-half:cx+half, cy-half:cy+half] = 0.5
        v[cx-half:cx+half, cy-half:cy+half] = 0.25
        
        # Add small noise to break symmetry
        u += np.random.uniform(-0.01, 0.01, (self.nx, self.ny))
        v += np.random.uniform(-0.01, 0.01, (self.nx, self.ny))
        
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        
        return u, v
    
    def initial_condition_stripes(self, n_stripes: int = 4, 
                                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize with stripe perturbations (good for stripe patterns).
        """
        if seed is not None:
            np.random.seed(seed)
        
        u = np.ones((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        
        # Vertical stripes
        for i in range(n_stripes):
            x_start = int((i + 0.3) * self.nx / n_stripes)
            x_end = int((i + 0.7) * self.nx / n_stripes)
            u[x_start:x_end, :] = 0.5
            v[x_start:x_end, :] = 0.25
        
        # Add noise
        u += np.random.uniform(-0.05, 0.05, (self.nx, self.ny))
        v += np.random.uniform(-0.05, 0.05, (self.nx, self.ny))
        
        return np.clip(u, 0, 1), np.clip(v, 0, 1)


# ============================================================================
# High-Frequency Noise Injection (2D, for coupled fields)
# ============================================================================

def add_hf_noise_2d(field: np.ndarray, noise_level: float, 
                    nx: int, ny: int, hf_fraction: float = 0.25) -> np.ndarray:
    """
    Add high-frequency noise to a 2D field.
    
    Injects noise in the high-frequency band of the 2D spectrum.
    """
    kx = fftfreq(nx, d=1.0/nx)
    ky = fftfreq(ny, d=1.0/ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)
    
    # High-frequency mask
    k_min = hf_fraction * min(nx, ny) / 2
    k_max = min(nx, ny) / 2
    hf_mask = (K_mag >= k_min) & (K_mag <= k_max)
    
    # Create noise in spectral space
    noise_hat = np.zeros((nx, ny), dtype=complex)
    phases = np.random.uniform(0, 2*np.pi, (nx, ny))
    amplitudes = np.random.randn(nx, ny)
    noise_hat[hf_mask] = amplitudes[hf_mask] * np.exp(1j * phases[hf_mask])
    
    # Transform to physical space
    noise = np.real(ifft2(noise_hat))
    
    # Normalize
    if np.std(noise) > 1e-10:
        noise = noise / np.std(noise) * noise_level * np.std(field)
    
    return field + noise


def add_hf_noise_coupled(u: np.ndarray, v: np.ndarray, 
                          noise_level: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Add HF noise to both fields in the coupled system."""
    u_noisy = add_hf_noise_2d(u, noise_level, nx, ny)
    v_noisy = add_hf_noise_2d(v, noise_level, nx, ny)
    
    # Keep in valid range
    u_noisy = np.clip(u_noisy, 0, 1)
    v_noisy = np.clip(v_noisy, 0, 1)
    
    return u_noisy, v_noisy


# ============================================================================
# Surrogate Models for Coupled System
# ============================================================================

class ConvolutionalSurrogate2D_Coupled:
    """
    Spectral filter model for coupled (u, v) system.
    
    Learns transfer functions:
        û(t+1) = Huu ⊙ û(t) + Huv ⊙ v̂(t)
        v̂(t+1) = Hvu ⊙ û(t) + Hvv ⊙ v̂(t)
    """
    
    def __init__(self, nx: int, ny: int, seed: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        
        if seed is not None:
            np.random.seed(seed)
        
        # Transfer functions (4 for coupled system)
        self.Huu = np.ones((nx, ny), dtype=complex) * 0.9
        self.Huv = np.zeros((nx, ny), dtype=complex)
        self.Hvu = np.zeros((nx, ny), dtype=complex)
        self.Hvv = np.ones((nx, ny), dtype=complex) * 0.9
        
        # Add small random perturbations
        for H in [self.Huu, self.Huv, self.Hvu, self.Hvv]:
            H += (np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)) * 0.01
    
    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next timestep."""
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        u_next_hat = self.Huu * u_hat + self.Huv * v_hat
        v_next_hat = self.Hvu * u_hat + self.Hvv * v_hat
        
        u_next = np.real(ifft2(u_next_hat))
        v_next = np.real(ifft2(v_next_hat))
        
        return np.clip(u_next, 0, 1), np.clip(v_next, 0, 1)
    
    def train(self, inputs_u: List[np.ndarray], inputs_v: List[np.ndarray],
              targets_u: List[np.ndarray], targets_v: List[np.ndarray],
              lr: float = 0.05, n_iter: int = 100):
        """Train the spectral filters."""
        for iteration in range(n_iter):
            grad_Huu = np.zeros_like(self.Huu)
            grad_Huv = np.zeros_like(self.Huv)
            grad_Hvu = np.zeros_like(self.Hvu)
            grad_Hvv = np.zeros_like(self.Hvv)
            
            for u_in, v_in, u_tgt, v_tgt in zip(inputs_u, inputs_v, targets_u, targets_v):
                u_hat = fft2(u_in)
                v_hat = fft2(v_in)
                u_tgt_hat = fft2(u_tgt)
                v_tgt_hat = fft2(v_tgt)
                
                # Forward
                u_pred_hat = self.Huu * u_hat + self.Huv * v_hat
                v_pred_hat = self.Hvu * u_hat + self.Hvv * v_hat
                
                # Errors
                err_u = u_pred_hat - u_tgt_hat
                err_v = v_pred_hat - v_tgt_hat
                
                # Gradients
                grad_Huu += np.conj(u_hat) * err_u
                grad_Huv += np.conj(v_hat) * err_u
                grad_Hvu += np.conj(u_hat) * err_v
                grad_Hvv += np.conj(v_hat) * err_v
            
            n = len(inputs_u)
            self.Huu -= lr * grad_Huu / n
            self.Huv -= lr * grad_Huv / n
            self.Hvu -= lr * grad_Hvu / n
            self.Hvv -= lr * grad_Hvv / n


class LinearSurrogate2D_Coupled:
    """
    Simple linear model for coupled system (baseline).
    
    Flattens both fields and learns a single large linear map.
    """
    
    def __init__(self, nx: int, ny: int, seed: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        self.n_features = 2 * nx * ny  # Both u and v
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize near identity
        self.W = np.eye(self.n_features) * 0.95
        self.W += np.random.randn(self.n_features, self.n_features) * 0.001
        self.b = np.zeros(self.n_features)
    
    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next timestep."""
        x = np.concatenate([u.flatten(), v.flatten()])
        y = self.W @ x + self.b
        
        n = self.nx * self.ny
        u_next = y[:n].reshape(self.nx, self.ny)
        v_next = y[n:].reshape(self.nx, self.ny)
        
        return np.clip(u_next, 0, 1), np.clip(v_next, 0, 1)
    
    def train(self, inputs_u: List[np.ndarray], inputs_v: List[np.ndarray],
              targets_u: List[np.ndarray], targets_v: List[np.ndarray],
              lr: float = 0.01, n_iter: int = 100):
        """Train with gradient descent."""
        X = np.array([np.concatenate([u.flatten(), v.flatten()]) 
                      for u, v in zip(inputs_u, inputs_v)])
        Y = np.array([np.concatenate([u.flatten(), v.flatten()]) 
                      for u, v in zip(targets_u, targets_v)])
        
        for _ in range(n_iter):
            pred = X @ self.W.T + self.b
            error = pred - Y
            
            self.W -= lr * (error.T @ X) / len(X)
            self.b -= lr * np.mean(error, axis=0)


# ============================================================================
# Rollout for Coupled System
# ============================================================================

def rollout_coupled(model, u0: np.ndarray, v0: np.ndarray, 
                    n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory by autoregressive rollout for coupled system.
    
    Returns:
        u_traj: (n_steps, nx, ny)
        v_traj: (n_steps, nx, ny)
    """
    nx, ny = u0.shape
    u_traj = np.zeros((n_steps, nx, ny))
    v_traj = np.zeros((n_steps, nx, ny))
    
    u_traj[0] = u0.copy()
    v_traj[0] = v0.copy()
    
    u, v = u0.copy(), v0.copy()
    for t in range(1, n_steps):
        u, v = model.forward(u, v)
        u_traj[t] = u
        v_traj[t] = v
    
    return u_traj, v_traj


# ============================================================================
# RSD Analysis for Coupled Reaction-Diffusion
# ============================================================================

class RSDAnalyzer2D_ReactionDiffusion:
    """
    RSD for Gray-Scott reaction-diffusion system.
    
    Physics residuals:
        r_u = ∂u/∂t - Du∇²u + uv² - F(1-u)
        r_v = ∂v/∂t - Dv∇²v - uv² + (F+k)v
    
    Computes HFV/LFV from combined residual spectrum.
    """
    
    def __init__(self, config: GrayScottConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.Du = config.Du
        self.Dv = config.Dv
        self.F = config.F
        self.k = config.k
        
        # Wavenumbers
        dx = config.Lx / config.nx
        dy = config.Ly / config.ny
        self.kx = fftfreq(self.nx, d=dx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)
        
        # Frequency bands
        k_nyq = min(self.nx, self.ny) / 2
        self.k_low = max(1, k_nyq * config.omega_1_frac)
        self.k_high = k_nyq * config.omega_2_frac
        self.k_max = k_nyq
        
        # Band masks
        self.low_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_low)
        self.high_mask = (self.K_mag > self.k_high) & (self.K_mag <= self.k_max)
        self.total_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_max)
    
    def compute_residual(self, u_traj: np.ndarray, v_traj: np.ndarray, 
                         dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute physics residuals for both species.
        
        Returns:
            r_u: (n_steps-2, nx, ny) residuals for u
            r_v: (n_steps-2, nx, ny) residuals for v
        """
        n_steps = u_traj.shape[0]
        r_u_list = []
        r_v_list = []
        
        for t in range(1, n_steps - 1):
            u = u_traj[t]
            v = v_traj[t]
            
            # Time derivatives (central difference)
            du_dt = (u_traj[t+1] - u_traj[t-1]) / (2 * dt)
            dv_dt = (v_traj[t+1] - v_traj[t-1]) / (2 * dt)
            
            # Laplacians (spectral)
            u_hat = fft2(u)
            v_hat = fft2(v)
            lap_u = np.real(ifft2(-self.K2 * u_hat))
            lap_v = np.real(ifft2(-self.K2 * v_hat))
            
            # Reaction terms
            uv2 = u * v * v
            react_u = -uv2 + self.F * (1 - u)
            react_v = uv2 - (self.F + self.k) * v
            
            # Residuals: should be zero for true solution
            # r = ∂u/∂t - Du∇²u - reaction
            r_u = du_dt - self.Du * lap_u - react_u
            r_v = dv_dt - self.Dv * lap_v - react_v
            
            r_u_list.append(r_u)
            r_v_list.append(r_v)
        
        return np.array(r_u_list), np.array(r_v_list)
    
    def compute_combined_power_spectrum(self, r_u: np.ndarray, 
                                         r_v: np.ndarray) -> np.ndarray:
        """
        Compute combined 2D power spectrum from both residuals.
        
        P_combined = |FFT(r_u)|² + |FFT(r_v)|²
        """
        if r_u.ndim == 2:
            r_u = r_u[np.newaxis, :, :]
            r_v = r_v[np.newaxis, :, :]
        
        power_u = np.mean(np.abs(fft2(r_u, axes=(-2, -1)))**2, axis=0)
        power_v = np.mean(np.abs(fft2(r_v, axes=(-2, -1)))**2, axis=0)
        
        return power_u + power_v
    
    def compute_hfv(self, r_u: np.ndarray, r_v: np.ndarray) -> float:
        """Compute High-Frequency Violation from combined residuals."""
        power = self.compute_combined_power_spectrum(r_u, r_v)
        
        hf_energy = np.sum(power[self.high_mask])
        total_energy = np.sum(power[self.total_mask])
        
        return hf_energy / (total_energy + 1e-10)
    
    def compute_lfv(self, r_u: np.ndarray, r_v: np.ndarray) -> float:
        """Compute Low-Frequency Violation from combined residuals."""
        power = self.compute_combined_power_spectrum(r_u, r_v)
        
        lf_energy = np.sum(power[self.low_mask])
        total_energy = np.sum(power[self.total_mask])
        
        return lf_energy / (total_energy + 1e-10)
    
    def compute_all_metrics(self, u_traj: np.ndarray, v_traj: np.ndarray,
                           u_true: np.ndarray, v_true: np.ndarray,
                           dt: float) -> Dict[str, float]:
        """Compute all RSD metrics."""
        # L2 errors (combined)
        l2_u = np.linalg.norm(u_traj - u_true) / np.linalg.norm(u_true)
        l2_v = np.linalg.norm(v_traj - v_true) / np.linalg.norm(v_true)
        l2_combined = np.sqrt(l2_u**2 + l2_v**2)
        
        # Physics residuals
        r_u, r_v = self.compute_residual(u_traj, v_traj, dt)
        
        # RSD metrics
        hfv = self.compute_hfv(r_u, r_v)
        lfv = self.compute_lfv(r_u, r_v)
        
        return {
            'l2_error': l2_combined,
            'l2_u': l2_u,
            'l2_v': l2_v,
            'hfv': hfv,
            'lfv': lfv,
        }


# ============================================================================
# Main Experiment Pipeline
# ============================================================================

def run_single_experiment(config: GrayScottConfig, seed: int) -> Dict[str, float]:
    """Run a single experiment with clean and noisy training."""
    solver = GrayScottSolver(config)
    rsd = RSDAnalyzer2D_ReactionDiffusion(config)
    
    np.random.seed(seed * 1000)
    
    # Generate training data
    print(f"  Generating training data (seed={seed})...")
    train_data = []
    for i in range(config.n_train_trajectories):
        u0, v0 = solver.initial_condition_random_seeds(n_seeds=15, seed=seed*1000 + i)
        t, u_traj, v_traj = solver.solve(u0, v0, config.t_final, config.n_snapshots)
        train_data.append({'u': u_traj, 'v': v_traj})
    dt = t[1] - t[0]
    
    # Generate test data
    print(f"  Generating test data...")
    test_cases = []
    for i in range(config.n_test_trajectories):
        u0, v0 = solver.initial_condition_random_seeds(n_seeds=15, seed=seed*1000 + 500 + i)
        t, u_traj, v_traj = solver.solve(u0, v0, config.t_final, config.n_snapshots)
        test_cases.append({
            'u0': u0, 'v0': v0,
            'u_true': u_traj, 'v_true': v_traj
        })
    
    # Prepare training pairs
    inputs_u, inputs_v = [], []
    targets_u_clean, targets_v_clean = [], []
    targets_u_noisy, targets_v_noisy = [], []
    
    for data in train_data:
        u_traj, v_traj = data['u'], data['v']
        for ti in range(len(u_traj) - 1):
            inputs_u.append(u_traj[ti])
            inputs_v.append(v_traj[ti])
            targets_u_clean.append(u_traj[ti+1])
            targets_v_clean.append(v_traj[ti+1])
            
            # Add noise
            u_noisy, v_noisy = add_hf_noise_coupled(
                u_traj[ti+1], v_traj[ti+1],
                config.noise_level, config.nx, config.ny
            )
            targets_u_noisy.append(u_noisy)
            targets_v_noisy.append(v_noisy)
    
    # Train clean model
    print(f"  Training clean model...")
    model_clean = ConvolutionalSurrogate2D_Coupled(config.nx, config.ny, seed=seed)
    model_clean.train(inputs_u, inputs_v, targets_u_clean, targets_v_clean, 
                      lr=0.05, n_iter=100)
    
    # Train noisy model
    print(f"  Training noisy model...")
    model_noisy = ConvolutionalSurrogate2D_Coupled(config.nx, config.ny, seed=seed+10000)
    model_noisy.train(inputs_u, inputs_v, targets_u_noisy, targets_v_noisy,
                      lr=0.05, n_iter=100)
    
    # Evaluate
    print(f"  Evaluating...")
    results = {
        'clean_l2': [], 'noisy_l2': [],
        'clean_hfv': [], 'noisy_hfv': [],
        'clean_lfv': [], 'noisy_lfv': [],
    }
    
    for case in test_cases:
        # Rollouts
        u_clean, v_clean = rollout_coupled(model_clean, case['u0'], case['v0'], config.n_snapshots)
        u_noisy, v_noisy = rollout_coupled(model_noisy, case['u0'], case['v0'], config.n_snapshots)
        
        # Metrics
        m_clean = rsd.compute_all_metrics(u_clean, v_clean, case['u_true'], case['v_true'], dt)
        m_noisy = rsd.compute_all_metrics(u_noisy, v_noisy, case['u_true'], case['v_true'], dt)
        
        results['clean_l2'].append(m_clean['l2_error'])
        results['noisy_l2'].append(m_noisy['l2_error'])
        results['clean_hfv'].append(m_clean['hfv'])
        results['noisy_hfv'].append(m_noisy['hfv'])
        results['clean_lfv'].append(m_clean['lfv'])
        results['noisy_lfv'].append(m_noisy['lfv'])
    
    return {k: np.mean(v) for k, v in results.items()}


def run_full_experiment(n_seeds: int = 8) -> Dict:
    """Run full multi-seed experiment."""
    print("="*70)
    print("RSD CASE STUDY: 2D REACTION-DIFFUSION (GRAY-SCOTT)")
    print("="*70)
    
    config = GrayScottConfig(
        nx=64, ny=64,           # Reduced for speed
        Du=0.16, Dv=0.08,
        F=0.035, k=0.065,       # Spots pattern
        t_final=1000.0,         # Reduced for speed
        n_snapshots=15,
        n_train_trajectories=5,
        n_test_trajectories=3,
        noise_level=0.03,
    )
    
    print(f"\nConfiguration:")
    print(f"  PDE: Gray-Scott Reaction-Diffusion")
    print(f"       ∂u/∂t = Du∇²u - uv² + F(1-u)")
    print(f"       ∂v/∂t = Dv∇²v + uv² - (F+k)v")
    print(f"  Grid: {config.nx} x {config.ny}")
    print(f"  Parameters: Du={config.Du}, Dv={config.Dv}, F={config.F}, k={config.k}")
    print(f"  Pattern regime: spots")
    print(f"  Noise level: σ = {config.noise_level}")
    print(f"  Seeds: {n_seeds}")
    
    # Collect results
    all_results = {
        'clean_l2': [], 'noisy_l2': [],
        'clean_hfv': [], 'noisy_hfv': [],
        'clean_lfv': [], 'noisy_lfv': [],
    }
    
    for seed in range(1, n_seeds + 1):
        print(f"\n--- Seed {seed}/{n_seeds} ---")
        r = run_single_experiment(config, seed)
        
        for k, v in r.items():
            all_results[k].append(v)
        
        print(f"  Results: L2_clean={r['clean_l2']:.3f}, L2_noisy={r['noisy_l2']:.3f}, "
              f"HFV_clean={r['clean_hfv']:.3f}, HFV_noisy={r['noisy_hfv']:.3f}")
    
    # Statistics
    stats_result = {}
    for k, v in all_results.items():
        stats_result[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'sem': np.std(v) / np.sqrt(len(v)),
            'values': v,
        }
    
    # Statistical tests
    l2_tstat, l2_pval = stats.ttest_rel(all_results['noisy_l2'], all_results['clean_l2'])
    hfv_tstat, hfv_pval = stats.ttest_rel(all_results['noisy_hfv'], all_results['clean_hfv'])
    
    l2_diff = (stats_result['noisy_l2']['mean'] - stats_result['clean_l2']['mean']) / stats_result['clean_l2']['mean'] * 100
    hfv_diff = (stats_result['noisy_hfv']['mean'] - stats_result['clean_hfv']['mean']) / stats_result['clean_hfv']['mean'] * 100
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<12} {'Clean':<20} {'Noisy':<20} {'Δ (%)':<12} {'p-value':<12}")
    print("-"*76)
    print(f"{'L₂ Error':<12} {stats_result['clean_l2']['mean']:.3f} ± {stats_result['clean_l2']['sem']:.3f}          "
          f"{stats_result['noisy_l2']['mean']:.3f} ± {stats_result['noisy_l2']['sem']:.3f}          "
          f"{l2_diff:+.1f}%        {l2_pval:.4f}")
    print(f"{'HFV':<12} {stats_result['clean_hfv']['mean']:.3f} ± {stats_result['clean_hfv']['sem']:.3f}          "
          f"{stats_result['noisy_hfv']['mean']:.3f} ± {stats_result['noisy_hfv']['sem']:.3f}          "
          f"{hfv_diff:+.1f}%        {hfv_pval:.6f}")
    
    return {
        'config': {k: v for k, v in config.__dict__.items()},
        'stats': stats_result,
        'tests': {
            'l2_diff_pct': l2_diff,
            'hfv_diff_pct': hfv_diff,
            'l2_pval': l2_pval,
            'hfv_pval': hfv_pval,
        }
    }


# ============================================================================
# Visualization
# ============================================================================

def create_figures(results: Dict, output_dir: Path):
    """Create publication-quality figures."""
    
    plt.style.use('seaborn-v0_8-white')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    COLORS = {'clean': '#0072B2', 'noisy': '#D55E00'}
    stats = results['stats']
    
    # Figure 1: Main results (4-panel for RD)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    
    # Panel (a): Example pattern (placeholder)
    ax = axes[0, 0]
    ax.text(0.5, 0.5, 'Gray-Scott\nPattern\n(v field)', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.set_title('(a) Turing Pattern (v)')
    ax.axis('off')
    
    # Panel (b): Another view (placeholder)
    ax = axes[0, 1]
    ax.text(0.5, 0.5, 'Residual\nSpectrum', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.set_title('(b) Residual Power Spectrum')
    ax.axis('off')
    
    # Panel (c): L2 error
    ax = axes[1, 0]
    clean_l2 = stats['clean_l2']['values']
    noisy_l2 = stats['noisy_l2']['values']
    
    bp = ax.boxplot([clean_l2, noisy_l2], positions=[0, 1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['clean'])
    bp['boxes'][1].set_facecolor(COLORS['noisy'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Clean', 'Noisy'])
    ax.set_ylabel('Relative L₂ error')
    ax.set_title('(c) Solution Error')
    
    l2_diff = results['tests']['l2_diff_pct']
    l2_pval = results['tests']['l2_pval']
    ax.text(0.5, 0.95, f'Δ = {l2_diff:+.1f}%\np = {l2_pval:.3f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    
    # Panel (d): HFV
    ax = axes[1, 1]
    clean_hfv = stats['clean_hfv']['values']
    noisy_hfv = stats['noisy_hfv']['values']
    
    bp = ax.boxplot([clean_hfv, noisy_hfv], positions=[0, 1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['clean'])
    bp['boxes'][1].set_facecolor(COLORS['noisy'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Clean', 'Noisy'])
    ax.set_ylabel('HFV index')
    ax.set_title('(d) High-Frequency Violation')
    
    hfv_diff = results['tests']['hfv_diff_pct']
    hfv_pval = results['tests']['hfv_pval']
    pval_str = f'{hfv_pval:.2e}' if hfv_pval < 0.001 else f'{hfv_pval:.4f}'
    ax.text(0.5, 0.95, f'Δ = {hfv_diff:+.1f}%\np = {pval_str}',
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_grayscott_main.png', dpi=300)
    fig.savefig(output_dir / 'fig_grayscott_main.pdf')
    plt.close()
    
    # Figure 2: L2 vs HFV scatter
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    ax.scatter(clean_l2, clean_hfv, c=COLORS['clean'], label='Clean-trained',
               s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.scatter(noisy_l2, noisy_hfv, c=COLORS['noisy'], label='Noisy-trained',
               s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Relative L₂ error')
    ax.set_ylabel('HFV index')
    ax.legend()
    ax.set_title('L₂ Error vs HFV (Gray-Scott)')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_grayscott_scatter.png', dpi=300)
    fig.savefig(output_dir / 'fig_grayscott_scatter.pdf')
    plt.close()
    
    print(f"Figures saved to {output_dir}")


def generate_latex_table(results: Dict, output_dir: Path):
    """Generate LaTeX table."""
    stats = results['stats']
    tests = results['tests']
    
    latex = r"""
\begin{table}[h]
\centering
\caption{RSD results for 2D Gray-Scott reaction-diffusion. Mean $\pm$ SEM over 8 seeds.}
\label{tab:grayscott}
\vspace{0.5em}
\begin{tabular}{@{}lccc@{}}
\toprule
Metric & Clean-trained & Noisy-trained & $\Delta$ (\%) \\
\midrule
"""
    
    latex += f"Rel.~$L_2$ Error & ${stats['clean_l2']['mean']:.3f} \\pm {stats['clean_l2']['sem']:.3f}$ & "
    latex += f"${stats['noisy_l2']['mean']:.3f} \\pm {stats['noisy_l2']['sem']:.3f}$ & "
    latex += f"${tests['l2_diff_pct']:+.1f}$ (p={tests['l2_pval']:.3f}) \\\\\n"
    
    latex += f"HFV & ${stats['clean_hfv']['mean']:.3f} \\pm {stats['clean_hfv']['sem']:.3f}$ & "
    latex += f"${stats['noisy_hfv']['mean']:.3f} \\pm {stats['noisy_hfv']['sem']:.3f}$ & "
    pval_str = f"{tests['hfv_pval']:.2e}" if tests['hfv_pval'] < 0.001 else f"{tests['hfv_pval']:.4f}"
    latex += f"${tests['hfv_diff_pct']:+.1f}$ (p={pval_str}) \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'table_grayscott.tex', 'w') as f:
        f.write(latex)
    
    print(f"Table saved to {output_dir / 'table_grayscott.tex'}")


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = Path('/mnt/user-data/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run experiment
    results = run_full_experiment(n_seeds=8)
    
    # Create figures
    create_figures(results, output_dir)
    
    # Generate table
    generate_latex_table(results, output_dir)
    
    # Save results
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(output_dir / 'grayscott_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nAll results saved to {output_dir}")
    print("\n" + "="*70)
    print("2D GRAY-SCOTT CASE STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()