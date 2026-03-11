"""Numerical stabilization helpers for Navier-Stokes predictions."""

from __future__ import annotations

import numpy as np


def sanitize_field(field: np.ndarray) -> np.ndarray:
    """Convert non-finite outputs to bounded finite values."""
    clean = np.nan_to_num(field, nan=0.0, posinf=1e6, neginf=-1e6)
    np.clip(clean, -1e6, 1e6, out=clean)
    return clean
