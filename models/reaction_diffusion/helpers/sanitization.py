"""Numerical stabilization helpers for reaction-diffusion predictions."""

from __future__ import annotations

import numpy as np


def sanitize_species(field: np.ndarray) -> np.ndarray:
    """Keep species values finite while preserving signed data ranges."""
    clean = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(clean, -1e6, 1e6, out=clean)
    return clean
