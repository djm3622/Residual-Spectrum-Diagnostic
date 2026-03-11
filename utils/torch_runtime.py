"""Torch device selection and acceleration helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def _mps_backend() -> object | None:
    return getattr(torch.backends, "mps", None)


def _mps_is_built() -> bool:
    backend = _mps_backend()
    return bool(backend is not None and backend.is_built())


def _mps_is_available() -> bool:
    backend = _mps_backend()
    return bool(backend is not None and backend.is_available())


def resolve_torch_device(device: str | None = None) -> torch.device:
    """Resolve a user-facing device string to an available torch.device."""
    requested = (device or "auto").strip().lower()
    if requested not in DEVICE_CHOICES:
        options = ", ".join(DEVICE_CHOICES)
        raise ValueError(f"Unsupported device '{device}'. Use one of: {options}.")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not _mps_is_built():
            raise ValueError("Requested device 'mps' but this PyTorch build has no MPS support.")
        if not _mps_is_available():
            raise ValueError("Requested device 'mps' but no compatible Apple GPU is available.")
        return torch.device("mps")

    return torch.device("cpu")


def configure_torch_backend(device: torch.device) -> None:
    """Enable backend-specific speedups that preserve the same model logic."""
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def build_adam_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    """Build Adam optimizer with optional CUDA fused kernels."""
    params = list(parameters)
    has_complex_params = any(param.is_complex() for param in params)

    if device.type == "cuda":
        try:
            if not has_complex_params:
                return torch.optim.Adam(params, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(params, lr=lr)


def build_grad_scaler(device: torch.device) -> torch.cuda.amp.GradScaler | None:
    """AMP is disabled for stability, so no GradScaler is needed."""
    _ = device
    return None


def maybe_disable_grad_scaler_for_complex_params(
    grad_scaler: torch.cuda.amp.GradScaler | None,
    model: torch.nn.Module,
) -> torch.cuda.amp.GradScaler | None:
    """Disable CUDA GradScaler for models with complex-valued parameters.

    PyTorch AMP unscale kernels do not support complex grads on CUDA.
    """
    if grad_scaler is None:
        return None

    for param in model.parameters():
        if param.is_complex():
            return None
    return grad_scaler


def train_autocast(device: torch.device):
    """Autocast is disabled globally; always train in full precision."""
    _ = device
    return nullcontext()