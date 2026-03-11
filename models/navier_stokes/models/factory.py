"""Factory and rollout helpers for Navier-Stokes surrogates."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..helpers.interfaces import OneStepModel
from ..helpers.sanitization import sanitize_field
from .convolutional_surrogate import ConvolutionalSurrogate2D
from .neural_operator_surrogate import NeuralOperatorSurrogate2D


def rollout_2d(model: OneStepModel, omega0: np.ndarray, n_steps: int) -> np.ndarray:
    """Autoregressive rollout for one-step models."""
    nx, ny = omega0.shape
    trajectory = np.zeros((n_steps, nx, ny), dtype=np.float32)
    trajectory[0] = np.asarray(omega0, dtype=np.float32)

    omega = np.asarray(omega0, dtype=np.float32)
    for step in range(1, n_steps):
        omega = model.forward(omega)
        omega = sanitize_field(omega)
        trajectory[step] = omega

    return trajectory


def build_model(
    method: str,
    nx: int,
    ny: int,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    model_width: int = 64,
    model_depth: int = 5,
    operator_config: Optional[Mapping[str, Any]] = None,
) -> OneStepModel:
    """Factory for NS surrogate models selected by CLI method arg."""
    normalized = method.strip().lower().replace("-", "_")

    if normalized in {"conv", "convolutional", "spectral", "nonlinear"}:
        return ConvolutionalSurrogate2D(
            nx,
            ny,
            seed=seed,
            device=device,
            loss=loss,
            model_width=model_width,
            model_depth=model_depth,
        )
    if normalized in {"fno", "neuralop_fno", "operator_fno"}:
        return NeuralOperatorSurrogate2D(
            nx,
            ny,
            operator="fno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
        )
    if normalized in {"tfno", "neuralop_tfno", "operator_tfno"}:
        return NeuralOperatorSurrogate2D(
            nx,
            ny,
            operator="tfno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
        )
    if normalized in {"uno", "neuralop_uno", "operator_uno"}:
        return NeuralOperatorSurrogate2D(
            nx,
            ny,
            operator="uno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
        )

    raise ValueError(
        "Unsupported method "
        f"'{method}'. Use one of: conv, fno, tfno, uno"
    )
