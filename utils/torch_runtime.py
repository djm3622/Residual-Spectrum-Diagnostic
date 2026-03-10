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
    if device.type == "cuda":
        try:
            return torch.optim.Adam(parameters, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(parameters, lr=lr)


def build_grad_scaler(device: torch.device) -> torch.cuda.amp.GradScaler | None:
    """Create a CUDA GradScaler when available."""
    if device.type != "cuda":
        return None

    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        try:
            return amp_module.GradScaler("cuda", enabled=True)
        except TypeError:
            return amp_module.GradScaler(enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def train_autocast(device: torch.device):
    """Context manager for CUDA mixed precision training."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
