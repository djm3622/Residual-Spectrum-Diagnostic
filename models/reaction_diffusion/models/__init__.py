"""Concrete reaction-diffusion model implementations and factory."""

from .factory import build_model, rollout_coupled

__all__ = ["build_model", "rollout_coupled"]
