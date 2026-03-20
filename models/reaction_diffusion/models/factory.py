"""Factory and rollout helpers for reaction-diffusion surrogates."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from data.reaction_diffusion import GrayScottConfig

from ..helpers.interfaces import CoupledOneStepModel
from ..helpers.sanitization import sanitize_species
from .convolutional_surrogate import ConvolutionalSurrogate2DCoupled
from .neural_operator_surrogate import NeuralOperatorSurrogate2DCoupled
from .physics_surrogate import PhysicsConsistentSurrogate2DCoupled


def rollout_coupled(
    model: CoupledOneStepModel,
    u0: np.ndarray,
    v0: np.ndarray,
    n_steps: int,
    context_u: np.ndarray | None = None,
    context_v: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout for coupled one-step models."""
    rollout_fn = getattr(model, "rollout", None)
    if callable(rollout_fn):
        try:
            pred_u, pred_v = rollout_fn(u0, v0, n_steps, context_u=context_u, context_v=context_v)
        except TypeError:
            pred_u, pred_v = rollout_fn(u0, v0, n_steps)
        return np.asarray(pred_u, dtype=np.float32), np.asarray(pred_v, dtype=np.float32)

    nx, ny = u0.shape
    u_traj = np.zeros((n_steps, nx, ny), dtype=np.float32)
    v_traj = np.zeros((n_steps, nx, ny), dtype=np.float32)

    u_traj[0] = np.asarray(u0, dtype=np.float32)
    v_traj[0] = np.asarray(v0, dtype=np.float32)

    u = np.asarray(u0, dtype=np.float32)
    v = np.asarray(v0, dtype=np.float32)
    for step in range(1, n_steps):
        u, v = model.forward(u, v)
        u = sanitize_species(u)
        v = sanitize_species(v)
        u_traj[step] = u
        v_traj[step] = v

    return u_traj, v_traj


def build_model(
    method: str,
    nx: int,
    ny: int,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    config: Optional[GrayScottConfig] = None,
    snapshot_dt: Optional[float] = None,
    operator_config: Optional[Mapping[str, Any]] = None,
    baseline_config: Optional[Mapping[str, Any]] = None,
    temporal_config: Optional[Mapping[str, Any]] = None,
) -> CoupledOneStepModel:
    """Factory for coupled RD surrogate models selected by CLI method arg."""
    normalized = method.strip().lower().replace("-", "_")

    if normalized in {"physics", "gray_scott", "grayscott", "rd_physics"}:
        if config is None or snapshot_dt is None:
            raise ValueError("Physics RD model requires GrayScottConfig and snapshot_dt.")
        return PhysicsConsistentSurrogate2DCoupled(config=config, snapshot_dt=snapshot_dt, device=device)
    if normalized in {"tfno", "neuralop_tfno", "operator_tfno"}:
        return NeuralOperatorSurrogate2DCoupled(
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
        return NeuralOperatorSurrogate2DCoupled(
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
        return NeuralOperatorSurrogate2DCoupled(
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
        return NeuralOperatorSurrogate2DCoupled(
            nx,
            ny,
            operator="rno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
            temporal_config=temporal_config,
        )
    if normalized in {"wno", "neuralop_wno", "operator_wno"}:
        return NeuralOperatorSurrogate2DCoupled(
            nx,
            ny,
            operator="wno",
            seed=seed,
            device=device,
            loss=loss,
            operator_config=operator_config,
            temporal_config=temporal_config,
        )
    if normalized in {"conv", "convolutional", "legacy_conv"}:
        return ConvolutionalSurrogate2DCoupled(
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
        return ConvolutionalSurrogate2DCoupled(
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
        return ConvolutionalSurrogate2DCoupled(
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
        f"'{method}'. Use one of: tfno, itfno, uno, wno, rno, conv, swin, attn_unet, physics"
    )
