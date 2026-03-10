"""Surrogate models for coupled 2D Gray-Scott trajectories."""

from __future__ import annotations

from typing import Dict, List, Protocol, Tuple

import numpy as np
from scipy.fft import fft2, ifft2


class CoupledOneStepModel(Protocol):
    """Minimal protocol for coupled one-step models."""

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def train(
        self,
        inputs_u: List[np.ndarray],
        inputs_v: List[np.ndarray],
        targets_u: List[np.ndarray],
        targets_v: List[np.ndarray],
        lr: float,
        n_iter: int,
    ) -> None:
        ...

    def state_dict(self) -> Dict[str, np.ndarray]:
        ...


class ConvolutionalSurrogate2DCoupled:
    """Coupled spectral filters for (u, v) updates in Fourier space."""

    def __init__(self, nx: int, ny: int, seed: int | None = None):
        self.nx = nx
        self.ny = ny

        if seed is not None:
            np.random.seed(seed)

        self.Huu = np.ones((nx, ny), dtype=complex) * 0.9
        self.Huv = np.zeros((nx, ny), dtype=complex)
        self.Hvu = np.zeros((nx, ny), dtype=complex)
        self.Hvv = np.ones((nx, ny), dtype=complex) * 0.9

        for weights in (self.Huu, self.Huv, self.Hvu, self.Hvv):
            weights += (np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)) * 0.01

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_hat = fft2(u)
        v_hat = fft2(v)

        u_next_hat = self.Huu * u_hat + self.Huv * v_hat
        v_next_hat = self.Hvu * u_hat + self.Hvv * v_hat

        u_next = np.real(ifft2(u_next_hat))
        v_next = np.real(ifft2(v_next_hat))
        return np.clip(u_next, 0, 1), np.clip(v_next, 0, 1)

    def train(
        self,
        inputs_u: List[np.ndarray],
        inputs_v: List[np.ndarray],
        targets_u: List[np.ndarray],
        targets_v: List[np.ndarray],
        lr: float = 0.05,
        n_iter: int = 100,
    ) -> None:
        for _ in range(n_iter):
            grad_Huu = np.zeros_like(self.Huu)
            grad_Huv = np.zeros_like(self.Huv)
            grad_Hvu = np.zeros_like(self.Hvu)
            grad_Hvv = np.zeros_like(self.Hvv)

            for u_in, v_in, u_tgt, v_tgt in zip(inputs_u, inputs_v, targets_u, targets_v):
                u_hat = fft2(u_in)
                v_hat = fft2(v_in)
                u_tgt_hat = fft2(u_tgt)
                v_tgt_hat = fft2(v_tgt)

                u_pred_hat = self.Huu * u_hat + self.Huv * v_hat
                v_pred_hat = self.Hvu * u_hat + self.Hvv * v_hat

                err_u = u_pred_hat - u_tgt_hat
                err_v = v_pred_hat - v_tgt_hat

                grad_Huu += np.conj(u_hat) * err_u
                grad_Huv += np.conj(v_hat) * err_u
                grad_Hvu += np.conj(u_hat) * err_v
                grad_Hvv += np.conj(v_hat) * err_v

            n = len(inputs_u)
            self.Huu -= lr * grad_Huu / n
            self.Huv -= lr * grad_Huv / n
            self.Hvu -= lr * grad_Hvu / n
            self.Hvv -= lr * grad_Hvv / n

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "Huu": self.Huu,
            "Huv": self.Huv,
            "Hvu": self.Hvu,
            "Hvv": self.Hvv,
        }


class LinearSurrogate2DCoupled:
    """Dense linear baseline over flattened concatenated (u, v)."""

    def __init__(self, nx: int, ny: int, seed: int | None = None):
        self.nx = nx
        self.ny = ny
        self.n_features = 2 * nx * ny

        if seed is not None:
            np.random.seed(seed)

        self.W = np.eye(self.n_features) * 0.95
        self.W += np.random.randn(self.n_features, self.n_features) * 0.001
        self.b = np.zeros(self.n_features)

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([u.flatten(), v.flatten()])
        y = self.W @ x + self.b

        n = self.nx * self.ny
        u_next = y[:n].reshape(self.nx, self.ny)
        v_next = y[n:].reshape(self.nx, self.ny)
        return np.clip(u_next, 0, 1), np.clip(v_next, 0, 1)

    def train(
        self,
        inputs_u: List[np.ndarray],
        inputs_v: List[np.ndarray],
        targets_u: List[np.ndarray],
        targets_v: List[np.ndarray],
        lr: float = 0.01,
        n_iter: int = 100,
    ) -> None:
        X = np.array([np.concatenate([u.flatten(), v.flatten()]) for u, v in zip(inputs_u, inputs_v)])
        Y = np.array([np.concatenate([u.flatten(), v.flatten()]) for u, v in zip(targets_u, targets_v)])

        for _ in range(n_iter):
            pred = X @ self.W.T + self.b
            error = pred - Y

            self.W -= lr * (error.T @ X) / len(X)
            self.b -= lr * np.mean(error, axis=0)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}


def rollout_coupled(
    model: CoupledOneStepModel,
    u0: np.ndarray,
    v0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout for coupled one-step models."""
    nx, ny = u0.shape
    u_traj = np.zeros((n_steps, nx, ny))
    v_traj = np.zeros((n_steps, nx, ny))

    u_traj[0] = u0.copy()
    v_traj[0] = v0.copy()

    u, v = u0.copy(), v0.copy()
    for step in range(1, n_steps):
        u, v = model.forward(u, v)
        u_traj[step] = u
        v_traj[step] = v

    return u_traj, v_traj


def build_model(method: str, nx: int, ny: int, seed: int) -> CoupledOneStepModel:
    """Factory for coupled RD surrogate models selected by CLI method arg."""
    normalized = method.strip().lower()

    if normalized in {"conv", "convolutional", "spectral"}:
        return ConvolutionalSurrogate2DCoupled(nx, ny, seed=seed)
    if normalized in {"linear", "dense"}:
        return LinearSurrogate2DCoupled(nx, ny, seed=seed)

    raise ValueError(
        f"Unsupported method '{method}'. Use one of: conv, convolutional, spectral, linear, dense"
    )
