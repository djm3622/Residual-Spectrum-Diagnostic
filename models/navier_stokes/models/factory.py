"""Factory and rollout helpers for Navier-Stokes surrogates."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..helpers.interfaces import OneStepModel
from ..helpers.sanitization import sanitize_field
from .convolutional_surrogate import ConvolutionalSurrogate2D
from .neural_operator_surrogate import NeuralOperatorSurrogate2D


def rollout_2d(
    model: OneStepModel,
    omega0: np.ndarray,
    n_steps: int,
    context: np.ndarray | None = None,
) -> np.ndarray:
    """Autoregressive rollout for one-step models."""
    rollout_fn = getattr(model, "rollout", None)
    if callable(rollout_fn):
        try:
            return np.asarray(
                rollout_fn(omega0, n_steps, context=context),
                dtype=np.float32,
            )
        except TypeError:
            return np.asarray(rollout_fn(omega0, n_steps), dtype=np.float32)

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
    operator_config: Optional[Mapping[str, Any]] = None,
    baseline_config: Optional[Mapping[str, Any]] = None,
    temporal_config: Optional[Mapping[str, Any]] = None,
) -> OneStepModel:
    """Factory for NS surrogate models selected by CLI method arg."""
    normalized = method.strip().lower().replace("-", "_")

    if normalized in {"tfno", "neuralop_tfno", "operator_tfno"}:
        return NeuralOperatorSurrogate2D(
            nx,
            ny,
            operator="tfno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
            temporal_config=temporal_config,
        )
    if normalized in {"itfno", "implicit_tfno", "neuralop_itfno", "operator_itfno"}:
        return NeuralOperatorSurrogate2D(
            nx,
            ny,
            operator="itfno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
            temporal_config=temporal_config,
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
            temporal_config=temporal_config,
        )
    if normalized in {"rno", "neuralop_rno", "operator_rno"}:
        return NeuralOperatorSurrogate2D(
            nx,
            ny,
            operator="rno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
            temporal_config=temporal_config,
        )
    if normalized in {"conv", "convolutional", "legacy_conv"}:
        return ConvolutionalSurrogate2D(
            nx,
            ny,
            seed=seed,
            device=device,
            loss=loss,
            architecture="legacy_conv",
            baseline_config=baseline_config,
            temporal_config=temporal_config,
        )
    if normalized in {"swin", "swin_transformer", "swin_t"}:
        return ConvolutionalSurrogate2D(
            nx,
            ny,
            seed=seed,
            device=device,
            loss=loss,
            architecture="swin",
            baseline_config=baseline_config,
            temporal_config=temporal_config,
        )
    if normalized in {"attn_unet", "attention_unet", "unet_attn"}:
        return ConvolutionalSurrogate2D(
            nx,
            ny,
            seed=seed,
            device=device,
            loss=loss,
            architecture="attn_unet",
            baseline_config=baseline_config,
            temporal_config=temporal_config,
        )

    raise ValueError(
        "Unsupported method "
        f"'{method}'. Use one of: tfno, itfno, uno, rno, conv, swin, attn_unet"
    )
