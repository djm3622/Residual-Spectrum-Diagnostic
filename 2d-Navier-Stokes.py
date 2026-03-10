#!/usr/bin/env python3
"""
RSD Case Study: 2D Navier-Stokes Equation
==========================================

PDE: Incompressible Navier-Stokes in vorticity-streamfunction formulation

    ∂ω/∂t + (u·∇)ω = ν ∇²ω
    
    where ω = ∇×u is vorticity, ψ is streamfunction (∇²ψ = -ω)
    and u = (∂ψ/∂y, -∂ψ/∂x)

This demonstrates RSD on:
- 2D spatial domain
- Nonlinear advection (like Burgers)
- Incompressibility constraint
- Turbulent dynamics

Author: A. Baheri
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.integrate import solve_ivp
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class NSConfig:
    """Configuration for 2D Navier-Stokes experiments."""
    # Grid
    nx: int = 64                    # Grid points in x
    ny: int = 64                    # Grid points in y
    Lx: float = 2 * np.pi           # Domain size in x
    Ly: float = 2 * np.pi           # Domain size in y
    
    # Physics
    nu: float = 0.001               # Kinematic viscosity (Re = 1/nu for unit velocity)
    
    # Time integration
    t_final: float = 1.0            # Final time
    n_snapshots: int = 20           # Number of snapshots to save
    dt_max: float = 0.01            # Maximum timestep for stability
    
    # Training
    n_train_trajectories: int = 8   # Number of training trajectories
    n_test_trajectories: int = 6    # Number of test trajectories
    noise_level: float = 0.04       # HF noise amplitude
    
    # RSD parameters
    omega_1_frac: float = 1/16      # Low-frequency boundary (fraction of N)
    omega_2_frac: float = 1/6       # High-frequency boundary (fraction of N)


# ============================================================================
# 2D Navier-Stokes Solver (Spectral Method)
# ============================================================================

class NavierStokes2D:
    """
    Pseudo-spectral solver for 2D incompressible Navier-Stokes.
    
    Uses vorticity-streamfunction formulation on periodic domain.
    Time integration: RK4 with adaptive stepping.
    Dealiasing: 2/3 rule.
    """
    
    def __init__(self, config: NSConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.nu = config.nu
        
        # Physical grid
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Wavenumbers
        self.kx = fftfreq(self.nx, d=self.dx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # Avoid division by zero
        
        # Dealiasing mask (2/3 rule)
        kx_max = self.nx // 3
        ky_max = self.ny // 3
        self.dealias_mask = (np.abs(self.KX) < kx_max * 2 * np.pi / self.Lx) & \
                           (np.abs(self.KY) < ky_max * 2 * np.pi / self.Ly)
        
        # Precompute inverse Laplacian for streamfunction
        self.inv_laplacian = -1.0 / self.K2
        self.inv_laplacian[0, 0] = 0.0  # Zero mean streamfunction
    
    def vorticity_to_velocity(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity field from vorticity via streamfunction.
        
        ∇²ψ = -ω  →  ψ̂ = ω̂ / k²
        u = ∂ψ/∂y, v = -∂ψ/∂x
        """
        omega_hat = fft2(omega)
        psi_hat = self.inv_laplacian * omega_hat
        
        # Velocity in spectral space
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat
        
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        
        return u, v
    
    def compute_nonlinear_term(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute nonlinear advection term: (u·∇)ω
        
        Uses dealiasing via 2/3 rule.
        """
        u, v = self.vorticity_to_velocity(omega)
        
        # Compute gradients in spectral space
        omega_hat = fft2(omega)
        domega_dx = np.real(ifft2(1j * self.KX * omega_hat))
        domega_dy = np.real(ifft2(1j * self.KY * omega_hat))
        
        # Nonlinear term in physical space
        advection = u * domega_dx + v * domega_dy
        
        # Dealias
        advection_hat = fft2(advection)
        advection_hat *= self.dealias_mask
        advection = np.real(ifft2(advection_hat))
        
        return advection
    
    def compute_rhs(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute right-hand side: -[(u·∇)ω] + ν∇²ω
        """
        omega_hat = fft2(omega)
        
        # Diffusion term (spectral)
        diffusion = np.real(ifft2(-self.nu * self.K2 * omega_hat))
        
        # Advection term
        advection = self.compute_nonlinear_term(omega)
        
        return -advection + diffusion
    
    def rk4_step(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """Fourth-order Runge-Kutta time step."""
        k1 = self.compute_rhs(omega)
        k2 = self.compute_rhs(omega + 0.5 * dt * k1)
        k3 = self.compute_rhs(omega + 0.5 * dt * k2)
        k4 = self.compute_rhs(omega + dt * k3)
        
        return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve(self, omega0: np.ndarray, t_final: float, n_snapshots: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate from omega0 to t_final, saving n_snapshots.
        
        Returns:
            t: Array of snapshot times
            omega_traj: Array of shape (n_snapshots, nx, ny)
        """
        t_save = np.linspace(0, t_final, n_snapshots)
        omega_traj = np.zeros((n_snapshots, self.nx, self.ny))
        omega_traj[0] = omega0.copy()
        
        omega = omega0.copy()
        t = 0.0
        save_idx = 1
        
        # Adaptive timestep based on CFL
        dt = min(self.config.dt_max, 0.1 * self.dx / (np.max(np.abs(omega)) + 1e-10))
        
        while save_idx < n_snapshots:
            # Adjust dt to hit save times exactly
            if t + dt > t_save[save_idx]:
                dt = t_save[save_idx] - t
            
            omega = self.rk4_step(omega, dt)
            t += dt
            
            # Save if we've reached a save time
            if t >= t_save[save_idx] - 1e-10:
                omega_traj[save_idx] = omega.copy()
                save_idx += 1
            
            # Update timestep
            u, v = self.vorticity_to_velocity(omega)
            max_vel = max(np.max(np.abs(u)), np.max(np.abs(v))) + 1e-10
            dt = min(self.config.dt_max, 0.25 * self.dx / max_vel)
        
        return t_save, omega_traj
    
    def random_initial_condition(self, seed: Optional[int] = None, 
                                  n_modes: int = 4, 
                                  amplitude: float = 1.0) -> np.ndarray:
        """
        Generate random smooth initial vorticity field.
        
        Uses superposition of low-wavenumber Fourier modes.
        """
        if seed is not None:
            np.random.seed(seed)
        
        omega = np.zeros((self.nx, self.ny))
        
        for _ in range(n_modes):
            kx = np.random.randint(1, 4)
            ky = np.random.randint(1, 4)
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            amp = amplitude * np.random.randn() / np.sqrt(kx**2 + ky**2)
            
            omega += amp * np.sin(kx * self.X + phase_x) * np.sin(ky * self.Y + phase_y)
        
        return omega
    
    def taylor_green_vortex(self, amplitude: float = 1.0) -> np.ndarray:
        """
        Taylor-Green vortex initial condition.
        
        Classic benchmark with known analytical solution (for short times).
        ω = 2A cos(x) cos(y)
        """
        return 2 * amplitude * np.cos(self.X) * np.cos(self.Y)
    
    def double_shear_layer(self, delta: float = 0.05, amplitude: float = 0.05) -> np.ndarray:
        """
        Double shear layer initial condition.
        
        Two parallel shear layers that roll up into vortices.
        Good test of nonlinear dynamics.
        """
        # Base shear profile
        y_norm = self.Y / self.Ly
        u = np.where(y_norm < 0.5,
                    np.tanh((y_norm - 0.25) / delta),
                    np.tanh((0.75 - y_norm) / delta))
        
        # Add perturbation
        v = amplitude * np.sin(2 * np.pi * self.X / self.Lx)
        
        # Compute vorticity: ω = ∂v/∂x - ∂u/∂y
        u_hat = fft2(u)
        v_hat = fft2(v)
        omega_hat = 1j * self.KX * v_hat - 1j * self.KY * u_hat
        omega = np.real(ifft2(omega_hat))
        
        return omega


# ============================================================================
# High-Frequency Noise Injection (2D)
# ============================================================================

def add_hf_noise_2d(field: np.ndarray, noise_level: float, 
                    nx: int, ny: int, hf_fraction: float = 0.25) -> np.ndarray:
    """
    Add high-frequency noise to a 2D field.
    
    Injects noise in the high-frequency band of the spectrum.
    
    Args:
        field: 2D array to corrupt
        noise_level: Standard deviation of noise
        nx, ny: Grid dimensions
        hf_fraction: Fraction of Nyquist to start noise injection
    """
    # Create noise in spectral space
    noise_hat = np.zeros((nx, ny), dtype=complex)
    
    kx = fftfreq(nx, d=1.0/nx)
    ky = fftfreq(ny, d=1.0/ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)
    
    # High-frequency mask
    k_min = hf_fraction * min(nx, ny) / 2
    k_max = min(nx, ny) / 2
    hf_mask = (K_mag >= k_min) & (K_mag <= k_max)
    
    # Random phases and amplitudes
    phases = np.random.uniform(0, 2*np.pi, (nx, ny))
    amplitudes = np.random.randn(nx, ny) * noise_level
    
    # Inject noise only in HF band
    noise_hat[hf_mask] = amplitudes[hf_mask] * np.exp(1j * phases[hf_mask])
    
    # Ensure conjugate symmetry for real output
    noise = np.real(ifft2(noise_hat))
    
    # Normalize to desired noise level
    noise = noise / (np.std(noise) + 1e-10) * noise_level * np.std(field)
    
    return field + noise


# ============================================================================
# Neural Network Surrogate Models
# ============================================================================

class LinearSurrogate2D:
    """
    Simple linear model for 2D fields: ω(t+1) = W @ flatten(ω(t)) + b
    
    This is a placeholder - replace with FNO/CNN for real experiments.
    """
    
    def __init__(self, nx: int, ny: int, seed: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        self.n_features = nx * ny
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with identity + small perturbation
        self.W = np.eye(self.n_features) + np.random.randn(self.n_features, self.n_features) * 0.001
        self.b = np.zeros(self.n_features)
    
    def forward(self, omega: np.ndarray) -> np.ndarray:
        """Predict next timestep."""
        omega_flat = omega.flatten()
        omega_next_flat = self.W @ omega_flat + self.b
        return omega_next_flat.reshape(self.nx, self.ny)
    
    def train(self, inputs: List[np.ndarray], targets: List[np.ndarray], 
              lr: float = 0.01, n_iter: int = 100):
        """
        Train with gradient descent.
        
        Args:
            inputs: List of input vorticity fields
            targets: List of target vorticity fields (next timestep)
        """
        X = np.array([inp.flatten() for inp in inputs])  # (N, nx*ny)
        Y = np.array([tgt.flatten() for tgt in targets])  # (N, nx*ny)
        
        for _ in range(n_iter):
            # Forward pass
            pred = X @ self.W.T + self.b
            error = pred - Y
            
            # Gradient descent
            grad_W = (error.T @ X) / len(X)
            grad_b = np.mean(error, axis=0)
            
            self.W -= lr * grad_W
            self.b -= lr * grad_b


class ConvolutionalSurrogate2D:
    """
    Simple convolutional model using FFT-based convolution.
    
    Learns a spectral filter: ω̂(t+1) = H ⊙ ω̂(t)
    where H is a learned transfer function.
    """
    
    def __init__(self, nx: int, ny: int, seed: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize transfer function close to identity
        self.H = np.ones((nx, ny), dtype=complex)
        self.H += (np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)) * 0.01
    
    def forward(self, omega: np.ndarray) -> np.ndarray:
        """Predict next timestep via spectral multiplication."""
        omega_hat = fft2(omega)
        omega_next_hat = self.H * omega_hat
        return np.real(ifft2(omega_next_hat))
    
    def train(self, inputs: List[np.ndarray], targets: List[np.ndarray],
              lr: float = 0.1, n_iter: int = 100):
        """Train the spectral filter."""
        for _ in range(n_iter):
            grad_H = np.zeros_like(self.H)
            
            for inp, tgt in zip(inputs, targets):
                inp_hat = fft2(inp)
                tgt_hat = fft2(tgt)
                pred_hat = self.H * inp_hat
                
                # Gradient of |pred_hat - tgt_hat|^2 w.r.t. H
                error_hat = pred_hat - tgt_hat
                grad_H += np.conj(inp_hat) * error_hat
            
            grad_H /= len(inputs)
            self.H -= lr * grad_H


class FNOSurrogate2D:
    """
    Simplified Fourier Neural Operator for 2D.
    
    Architecture:
        1. Lift to higher dimension
        2. Fourier layers: spectral convolution + pointwise MLP
        3. Project back to physical space
    
    This is a simplified version - for real experiments use the 
    official FNO implementation from Li et al.
    """
    
    def __init__(self, nx: int, ny: int, 
                 modes: int = 12,
                 width: int = 32,
                 n_layers: int = 4,
                 seed: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        self.modes = modes  # Number of Fourier modes to keep
        self.width = width  # Hidden channel width
        self.n_layers = n_layers
        
        if seed is not None:
            np.random.seed(seed)
        
        # Lifting layer: 1 -> width
        self.lift_W = np.random.randn(width, 1) * 0.1
        self.lift_b = np.zeros(width)
        
        # Fourier layers
        self.spectral_weights = []
        self.pointwise_W = []
        self.pointwise_b = []
        
        for _ in range(n_layers):
            # Complex weights for spectral convolution
            R = np.random.randn(width, width, modes, modes) * 0.1
            I = np.random.randn(width, width, modes, modes) * 0.1
            self.spectral_weights.append(R + 1j * I)
            
            # Pointwise linear layer
            self.pointwise_W.append(np.random.randn(width, width) * 0.1)
            self.pointwise_b.append(np.zeros(width))
        
        # Projection layer: width -> 1
        self.proj_W = np.random.randn(1, width) * 0.1
        self.proj_b = np.zeros(1)
    
    def spectral_conv(self, x_hat: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Spectral convolution layer.
        
        Args:
            x_hat: (width, nx, ny) complex array
            weights: (width, width, modes, modes) complex weights
        
        Returns:
            (width, nx, ny) complex array
        """
        width = x_hat.shape[0]
        modes = self.modes
        
        # Truncate to low modes
        x_modes = x_hat[:, :modes, :modes]
        
        # Spectral convolution (einsum over channels)
        out_modes = np.einsum('ijkl,jkl->ikl', weights, x_modes)
        
        # Pad back to full resolution
        out_hat = np.zeros_like(x_hat)
        out_hat[:, :modes, :modes] = out_modes
        
        return out_hat
    
    def forward(self, omega: np.ndarray) -> np.ndarray:
        """Forward pass through FNO."""
        # Lift: (nx, ny) -> (width, nx, ny)
        x = np.einsum('ij,kij->kij', omega, 
                      np.tile(self.lift_W, (1, self.nx, self.ny)).reshape(self.width, self.nx, self.ny))
        x += self.lift_b[:, None, None]
        
        # Fourier layers
        for i in range(self.n_layers):
            # Spectral path
            x_hat = np.array([fft2(x[c]) for c in range(self.width)])
            x_spectral = self.spectral_conv(x_hat, self.spectral_weights[i])
            x_spectral = np.array([np.real(ifft2(x_spectral[c])) for c in range(self.width)])
            
            # Pointwise path
            x_pointwise = np.einsum('ij,jkl->ikl', self.pointwise_W[i], x)
            x_pointwise += self.pointwise_b[i][:, None, None]
            
            # Combine and activation
            x = x_spectral + x_pointwise
            x = np.maximum(x, 0)  # ReLU
        
        # Project: (width, nx, ny) -> (nx, ny)
        out = np.einsum('ij,jkl->kl', self.proj_W, x)
        out += self.proj_b[0]
        
        return out
    
    def train(self, inputs: List[np.ndarray], targets: List[np.ndarray],
              lr: float = 0.001, n_iter: int = 500):
        """
        Train FNO with gradient descent.
        
        Note: This is a simplified training loop. For real experiments,
        use PyTorch with Adam optimizer and proper backprop.
        """
        warnings.warn("FNO training is simplified. Use PyTorch for real experiments.")
        
        for iteration in range(n_iter):
            total_loss = 0
            
            for inp, tgt in zip(inputs, targets):
                pred = self.forward(inp)
                loss = np.mean((pred - tgt)**2)
                total_loss += loss
                
                # Simplified gradient update (finite differences)
                # In practice, use autograd
                eps = 1e-5
                for layer_idx in range(self.n_layers):
                    # Update spectral weights
                    for i in range(min(4, self.width)):
                        for j in range(min(4, self.width)):
                            for k in range(min(4, self.modes)):
                                for l in range(min(4, self.modes)):
                                    # Real part
                                    self.spectral_weights[layer_idx][i,j,k,l] += eps
                                    loss_plus = np.mean((self.forward(inp) - tgt)**2)
                                    self.spectral_weights[layer_idx][i,j,k,l] -= 2*eps
                                    loss_minus = np.mean((self.forward(inp) - tgt)**2)
                                    self.spectral_weights[layer_idx][i,j,k,l] += eps
                                    
                                    grad = (loss_plus - loss_minus) / (2*eps)
                                    self.spectral_weights[layer_idx][i,j,k,l] -= lr * grad
            
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}: Loss = {total_loss/len(inputs):.6f}")


# ============================================================================
# Rollout Function
# ============================================================================

def rollout_2d(model, omega0: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Generate trajectory by autoregressive rollout.
    
    Args:
        model: Surrogate model with .forward() method
        omega0: Initial vorticity field
        n_steps: Number of timesteps
    
    Returns:
        Trajectory array of shape (n_steps, nx, ny)
    """
    nx, ny = omega0.shape
    trajectory = np.zeros((n_steps, nx, ny))
    trajectory[0] = omega0.copy()
    
    omega = omega0.copy()
    for t in range(1, n_steps):
        omega = model.forward(omega)
        trajectory[t] = omega.copy()
    
    return trajectory


# ============================================================================
# RSD Analysis for 2D
# ============================================================================

class RSDAnalyzer2D:
    """
    Residual Spectrum Diagnostics for 2D Navier-Stokes.
    
    Computes physics residual: r = ∂ω/∂t + (u·∇)ω - ν∇²ω
    Analyzes its 2D power spectrum to compute HFV and LFV.
    """
    
    def __init__(self, config: NSConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.nu = config.nu
        
        # Wavenumbers
        self.kx = fftfreq(self.nx, d=config.Lx/config.nx) * 2 * np.pi
        self.ky = fftfreq(self.ny, d=config.Ly/config.ny) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)
        
        # Frequency bands (in terms of wavenumber magnitude)
        k_nyq = min(self.nx, self.ny) / 2
        self.k_low = max(1, k_nyq * config.omega_1_frac)
        self.k_high = k_nyq * config.omega_2_frac
        self.k_max = k_nyq
        
        # Band masks
        self.low_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_low)
        self.mid_mask = (self.K_mag > self.k_low) & (self.K_mag <= self.k_high)
        self.high_mask = (self.K_mag > self.k_high) & (self.K_mag <= self.k_max)
        self.total_mask = (self.K_mag >= 1) & (self.K_mag <= self.k_max)
        
        # NS solver for computing nonlinear term
        self.ns_solver = NavierStokes2D(config)
    
    def compute_residual(self, omega_traj: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute physics residual at each timestep.
        
        Residual: r = ∂ω/∂t + (u·∇)ω - ν∇²ω
        
        Args:
            omega_traj: (n_steps, nx, ny) trajectory
            dt: Time step between snapshots
        
        Returns:
            (n_steps-2, nx, ny) array of residuals
        """
        n_steps = omega_traj.shape[0]
        residuals = []
        
        for t in range(1, n_steps - 1):
            omega = omega_traj[t]
            omega_hat = fft2(omega)
            
            # Time derivative (central difference)
            d_omega_dt = (omega_traj[t+1] - omega_traj[t-1]) / (2 * dt)
            
            # Advection term: (u·∇)ω
            advection = self.ns_solver.compute_nonlinear_term(omega)
            
            # Diffusion term: ν∇²ω
            K2 = self.KX**2 + self.KY**2
            diffusion = np.real(ifft2(-self.nu * K2 * omega_hat))
            
            # Residual: should be zero for true solution
            residual = d_omega_dt + advection - diffusion
            residuals.append(residual)
        
        return np.array(residuals)
    
    def compute_power_spectrum(self, residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D power spectrum of residuals.
        
        Returns:
            k_bins: Wavenumber magnitude bins
            power: Azimuthally-averaged power spectrum
        """
        # Compute 2D power spectrum
        if residuals.ndim == 2:
            residuals = residuals[np.newaxis, :, :]
        
        power_2d = np.mean(np.abs(fft2(residuals, axes=(-2, -1)))**2, axis=0)
        
        # Azimuthal average
        k_max = int(min(self.nx, self.ny) / 2)
        k_bins = np.arange(0, k_max + 1)
        power_1d = np.zeros(len(k_bins))
        
        for i, k in enumerate(k_bins):
            mask = (np.abs(self.K_mag - k) < 0.5)
            if np.any(mask):
                power_1d[i] = np.mean(power_2d[mask])
        
        return k_bins, power_1d
    
    def compute_hfv(self, residuals: np.ndarray) -> float:
        """
        Compute High-Frequency Violation index.
        
        HFV = Σ P(k) for k > k_high / Σ P(k) for k >= 1
        """
        if residuals.ndim == 2:
            residuals = residuals[np.newaxis, :, :]
        
        power_2d = np.mean(np.abs(fft2(residuals, axes=(-2, -1)))**2, axis=0)
        
        hf_energy = np.sum(power_2d[self.high_mask])
        total_energy = np.sum(power_2d[self.total_mask])
        
        return hf_energy / (total_energy + 1e-10)
    
    def compute_lfv(self, residuals: np.ndarray) -> float:
        """
        Compute Low-Frequency Violation index.
        
        LFV = Σ P(k) for 1 <= k <= k_low / Σ P(k) for k >= 1
        """
        if residuals.ndim == 2:
            residuals = residuals[np.newaxis, :, :]
        
        power_2d = np.mean(np.abs(fft2(residuals, axes=(-2, -1)))**2, axis=0)
        
        lf_energy = np.sum(power_2d[self.low_mask])
        total_energy = np.sum(power_2d[self.total_mask])
        
        return lf_energy / (total_energy + 1e-10)
    
    def compute_all_metrics(self, omega_traj: np.ndarray, omega_true: np.ndarray, 
                           dt: float) -> Dict[str, float]:
        """
        Compute all RSD metrics for a trajectory.
        
        Returns:
            Dictionary with L2 error, HFV, LFV, and residual magnitude
        """
        # L2 error
        l2_error = np.linalg.norm(omega_traj - omega_true) / np.linalg.norm(omega_true)
        
        # Physics residual
        residuals = self.compute_residual(omega_traj, dt)
        
        # RSD metrics
        hfv = self.compute_hfv(residuals)
        lfv = self.compute_lfv(residuals)
        
        # Residual magnitude
        residual_mag = np.mean(np.abs(residuals))
        
        return {
            'l2_error': l2_error,
            'hfv': hfv,
            'lfv': lfv,
            'residual_mag': residual_mag,
        }


# ============================================================================
# Main Experiment Pipeline
# ============================================================================

def run_single_experiment(config: NSConfig, seed: int) -> Dict[str, float]:
    """
    Run a single experiment with clean and noisy training.
    
    Returns metrics for both models.
    """
    ns = NavierStokes2D(config)
    rsd = RSDAnalyzer2D(config)
    
    np.random.seed(seed * 1000)
    
    # Generate training data
    print(f"  Generating training data (seed={seed})...")
    train_trajectories = []
    for i in range(config.n_train_trajectories):
        omega0 = ns.random_initial_condition(seed=seed*1000 + i)
        t, omega_traj = ns.solve(omega0, config.t_final, config.n_snapshots)
        train_trajectories.append(omega_traj)
    dt = t[1] - t[0]
    
    # Generate test data
    print(f"  Generating test data...")
    test_cases = []
    for i in range(config.n_test_trajectories):
        omega0 = ns.random_initial_condition(seed=seed*1000 + 500 + i)
        t, omega_traj = ns.solve(omega0, config.t_final, config.n_snapshots)
        test_cases.append({'omega0': omega0, 'omega_true': omega_traj})
    
    # Prepare training pairs
    inputs_clean = []
    targets_clean = []
    targets_noisy = []
    
    for traj in train_trajectories:
        for ti in range(len(traj) - 1):
            inputs_clean.append(traj[ti])
            targets_clean.append(traj[ti+1])
            targets_noisy.append(
                add_hf_noise_2d(traj[ti+1], config.noise_level, config.nx, config.ny)
            )
    
    # Train models
    print(f"  Training clean model...")
    model_clean = ConvolutionalSurrogate2D(config.nx, config.ny, seed=seed)
    model_clean.train(inputs_clean, targets_clean, lr=0.1, n_iter=100)
    
    print(f"  Training noisy model...")
    model_noisy = ConvolutionalSurrogate2D(config.nx, config.ny, seed=seed+10000)
    model_noisy.train(inputs_clean, targets_noisy, lr=0.1, n_iter=100)
    
    # Evaluate on test set
    print(f"  Evaluating...")
    results = {
        'clean_l2': [], 'noisy_l2': [],
        'clean_hfv': [], 'noisy_hfv': [],
        'clean_lfv': [], 'noisy_lfv': [],
    }
    
    for case in test_cases:
        # Rollout predictions
        omega_clean = rollout_2d(model_clean, case['omega0'], config.n_snapshots)
        omega_noisy = rollout_2d(model_noisy, case['omega0'], config.n_snapshots)
        
        # Compute metrics
        metrics_clean = rsd.compute_all_metrics(omega_clean, case['omega_true'], dt)
        metrics_noisy = rsd.compute_all_metrics(omega_noisy, case['omega_true'], dt)
        
        results['clean_l2'].append(metrics_clean['l2_error'])
        results['noisy_l2'].append(metrics_noisy['l2_error'])
        results['clean_hfv'].append(metrics_clean['hfv'])
        results['noisy_hfv'].append(metrics_noisy['hfv'])
        results['clean_lfv'].append(metrics_clean['lfv'])
        results['noisy_lfv'].append(metrics_noisy['lfv'])
    
    # Return mean over test cases
    return {k: np.mean(v) for k, v in results.items()}


def run_full_experiment(n_seeds: int = 10) -> Dict:
    """
    Run full multi-seed experiment.
    """
    print("="*70)
    print("RSD CASE STUDY: 2D NAVIER-STOKES EQUATION")
    print("="*70)
    
    config = NSConfig(
        nx=64, ny=64,
        nu=0.001,
        t_final=1.0,
        n_snapshots=20,
        n_train_trajectories=8,
        n_test_trajectories=6,
        noise_level=0.04,
    )
    
    print(f"\nConfiguration:")
    print(f"  PDE: 2D Incompressible Navier-Stokes (vorticity form)")
    print(f"  Grid: {config.nx} x {config.ny}")
    print(f"  Reynolds number: Re ≈ {1/config.nu:.0f}")
    print(f"  Noise level: σ = {config.noise_level}")
    print(f"  Training trajectories: {config.n_train_trajectories}")
    print(f"  Seeds: {n_seeds}")
    
    # Collect results across seeds
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
    
    # Compute statistics
    stats = {}
    for k, v in all_results.items():
        stats[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'sem': np.std(v) / np.sqrt(len(v)),
            'values': v,
        }
    
    # Statistical tests
    l2_tstat, l2_pval = stats.ttest_rel(all_results['noisy_l2'], all_results['clean_l2'])
    hfv_tstat, hfv_pval = stats.ttest_rel(all_results['noisy_hfv'], all_results['clean_hfv'])
    
    l2_diff = (stats['noisy_l2']['mean'] - stats['clean_l2']['mean']) / stats['clean_l2']['mean'] * 100
    hfv_diff = (stats['noisy_hfv']['mean'] - stats['clean_hfv']['mean']) / stats['clean_hfv']['mean'] * 100
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<12} {'Clean':<20} {'Noisy':<20} {'Δ (%)':<12} {'p-value':<12}")
    print("-"*76)
    print(f"{'L₂ Error':<12} {stats['clean_l2']['mean']:.3f} ± {stats['clean_l2']['sem']:.3f}          "
          f"{stats['noisy_l2']['mean']:.3f} ± {stats['noisy_l2']['sem']:.3f}          "
          f"{l2_diff:+.1f}%        {l2_pval:.4f}")
    print(f"{'HFV':<12} {stats['clean_hfv']['mean']:.3f} ± {stats['clean_hfv']['sem']:.3f}          "
          f"{stats['noisy_hfv']['mean']:.3f} ± {stats['noisy_hfv']['sem']:.3f}          "
          f"{hfv_diff:+.1f}%        {hfv_pval:.6f}")
    
    return {
        'config': config.__dict__,
        'stats': stats,
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
    """Create publication-quality figures for 2D NS case study."""
    
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
    
    # Figure 1: Main results (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    
    # Panel (a): Example vorticity fields
    # (Would need actual computed fields - placeholder)
    ax = axes[0]
    ax.text(0.5, 0.5, 'Vorticity\nVisualization', ha='center', va='center', 
            fontsize=12, transform=ax.transAxes)
    ax.set_title('(a) Vorticity Field')
    ax.axis('off')
    
    # Panel (b): L2 error boxplot
    ax = axes[1]
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
    ax.set_title('(b) Solution Error')
    
    l2_diff = results['tests']['l2_diff_pct']
    l2_pval = results['tests']['l2_pval']
    ax.text(0.5, 0.95, f'Δ = {l2_diff:+.1f}%\np = {l2_pval:.3f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    
    # Panel (c): HFV boxplot
    ax = axes[2]
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
    ax.set_title('(c) High-Frequency Violation')
    
    hfv_diff = results['tests']['hfv_diff_pct']
    hfv_pval = results['tests']['hfv_pval']
    pval_str = f'{hfv_pval:.2e}' if hfv_pval < 0.001 else f'{hfv_pval:.4f}'
    ax.text(0.5, 0.95, f'Δ = {hfv_diff:+.1f}%\np = {pval_str}',
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_ns2d_main.png', dpi=300)
    fig.savefig(output_dir / 'fig_ns2d_main.pdf')
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
    ax.set_title('L₂ Error vs HFV (2D Navier-Stokes)')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_ns2d_scatter.png', dpi=300)
    fig.savefig(output_dir / 'fig_ns2d_scatter.pdf')
    plt.close()
    
    print(f"Figures saved to {output_dir}")


def generate_latex_table(results: Dict, output_dir: Path):
    """Generate LaTeX table for NS results."""
    stats = results['stats']
    tests = results['tests']
    
    latex = r"""
\begin{table}[h]
\centering
\caption{RSD results for 2D Navier-Stokes (Re $\approx$ 1000). Mean $\pm$ SEM over 10 seeds.}
\label{tab:ns2d}
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
    
    with open(output_dir / 'table_ns2d.tex', 'w') as f:
        f.write(latex)
    
    print(f"Table saved to {output_dir / 'table_ns2d.tex'}")


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = Path('/mnt/user-data/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run experiment
    results = run_full_experiment(n_seeds=10)
    
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
    
    with open(output_dir / 'ns2d_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nAll results saved to {output_dir}")
    print("\n" + "="*70)
    print("2D NAVIER-STOKES CASE STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()