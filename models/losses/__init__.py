"""Shared loss utilities for model training objectives."""

from .objectives import LOSS_CHOICES, ObjectiveLoss, normalize_loss_name

__all__ = ["LOSS_CHOICES", "ObjectiveLoss", "normalize_loss_name"]
