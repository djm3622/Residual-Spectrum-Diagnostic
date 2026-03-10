"""Surrogate models for 2D Navier-Stokes trajectories."""

from __future__ import annotations

from typing import Dict, List, Protocol

import numpy as np
from scipy.fft import fft2, ifft2


class OneStepModel(Protocol):
    """Minimal model protocol used by run scripts."""

    def forward(self, omega: np.ndarray) -> np.ndarray:
        ...

    def train(self, inputs: List[np.ndarray], targets: List[np.ndarray], lr: float, n_iter: int) -> None:
        ...

    def state_dict(self) -> Dict[str, np.ndarray]:
        ...


class LinearSurrogate2D:
    """Dense linear baseline: ω(t+1) = W·ω(t) + b."""

    def __init__(self, nx: int, ny: int, seed: int | None = None):
        self.nx = nx
        self.ny = ny
        self.n_features = nx * ny

        if seed is not None:
            np.random.seed(seed)

        self.W = np.eye(self.n_features) + np.random.randn(self.n_features, self.n_features) * 0.001
        self.b = np.zeros(self.n_features)

    def forward(self, omega: np.ndarray) -> np.ndarray:
        flat = omega.flatten()
        pred = self.W @ flat + self.b
        return pred.reshape(self.nx, self.ny)

    def train(self, inputs: List[np.ndarray], targets: List[np.ndarray], lr: float = 0.01, n_iter: int = 100) -> None:
        X = np.array([inp.flatten() for inp in inputs])
        Y = np.array([tgt.flatten() for tgt in targets])

        for _ in range(n_iter):
            pred = X @ self.W.T + self.b
            error = pred - Y

            grad_W = (error.T @ X) / len(X)
            grad_b = np.mean(error, axis=0)

            self.W -= lr * grad_W
            self.b -= lr * grad_b

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}


class ConvolutionalSurrogate2D:
    """Spectral filter baseline: ω^(t+1) = H ⊙ ω^(t)."""

    def __init__(self, nx: int, ny: int, seed: int | None = None):
        self.nx = nx
        self.ny = ny

        if seed is not None:
            np.random.seed(seed)

        self.H = np.ones((nx, ny), dtype=complex)
        self.H += (np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)) * 0.01

    def forward(self, omega: np.ndarray) -> np.ndarray:
        omega_hat = fft2(omega)
        pred_hat = self.H * omega_hat
        return np.real(ifft2(pred_hat))

    def train(self, inputs: List[np.ndarray], targets: List[np.ndarray], lr: float = 0.1, n_iter: int = 100) -> None:
        for _ in range(n_iter):
            grad_H = np.zeros_like(self.H)

            for inp, tgt in zip(inputs, targets):
                inp_hat = fft2(inp)
                tgt_hat = fft2(tgt)
                pred_hat = self.H * inp_hat
                error_hat = pred_hat - tgt_hat
                grad_H += np.conj(inp_hat) * error_hat

            grad_H /= len(inputs)
            self.H -= lr * grad_H

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"H": self.H}


def rollout_2d(model: OneStepModel, omega0: np.ndarray, n_steps: int) -> np.ndarray:
    """Autoregressive rollout for one-step models."""
    nx, ny = omega0.shape
    trajectory = np.zeros((n_steps, nx, ny))
    trajectory[0] = omega0.copy()

    omega = omega0.copy()
    for step in range(1, n_steps):
        omega = model.forward(omega)
        trajectory[step] = omega

    return trajectory


def build_model(method: str, nx: int, ny: int, seed: int) -> OneStepModel:
    """Factory for NS surrogate models selected by CLI method arg."""
    normalized = method.strip().lower()

    if normalized in {"conv", "convolutional", "spectral"}:
        return ConvolutionalSurrogate2D(nx, ny, seed=seed)
    if normalized in {"linear", "dense"}:
        return LinearSurrogate2D(nx, ny, seed=seed)

    raise ValueError(
        f"Unsupported method '{method}'. Use one of: conv, convolutional, spectral, linear, dense"
    )
