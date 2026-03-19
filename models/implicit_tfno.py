"""Implicit TFNO variant with shared spectral block updates."""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
from neuralop.models import TFNO


class ImplicitTFNO(nn.Module):
    """Iterative TFNO with tied hidden-layer parameters.

    This module reuses a single TFNO hidden block for multiple fixed-point-style
    iterations:

        x_{k+1} = x_k + dt * (F(x_k) - x_k)

    where ``F`` is a one-layer TFNO hidden update (same parameters for all k).
    """

    def __init__(
        self,
        *,
        implicit_steps: int = 6,
        implicit_dt: float = 1.0,
        implicit_relaxation: bool = True,
        **tfno_kwargs: Any,
    ) -> None:
        super().__init__()
        steps = max(1, int(implicit_steps))
        dt = float(implicit_dt)
        if dt <= 0.0:
            raise ValueError(f"implicit_dt must be > 0, got {dt}.")

        self.implicit_steps = steps
        self.implicit_dt = dt
        self.implicit_relaxation = bool(implicit_relaxation)

        tfno_ctor_kwargs = dict(tfno_kwargs)
        tfno_ctor_kwargs["n_layers"] = 1
        self.backbone = TFNO(**tfno_ctor_kwargs)

        # Keep parity with FNO/TFNO attributes expected by some callers.
        self.n_layers = self.implicit_steps

    def _resolve_output_shapes(self, output_shape: Any) -> List[Any]:
        if output_shape is None:
            return [None] * self.implicit_steps
        if isinstance(output_shape, tuple):
            return [None] * (self.implicit_steps - 1) + [output_shape]
        if isinstance(output_shape, list):
            if len(output_shape) >= self.implicit_steps:
                return list(output_shape[: self.implicit_steps])
            if len(output_shape) == 0:
                return [None] * self.implicit_steps
            tail = output_shape[-1]
            return list(output_shape) + [tail] * (self.implicit_steps - len(output_shape))
        return [None] * self.implicit_steps

    def forward(self, x: torch.Tensor, output_shape: Any = None, **kwargs: Any) -> torch.Tensor:
        del kwargs  # Kept for API-compatibility with neuralop forward signatures.

        output_shapes = self._resolve_output_shapes(output_shape)
        x = self.backbone.lifting(x)

        if self.backbone.domain_padding is not None:
            x = self.backbone.domain_padding.pad(x)

        for step in range(self.implicit_steps):
            candidate = self.backbone.fno_blocks(x, index=0, output_shape=output_shapes[step])
            if self.implicit_relaxation:
                x = x + self.implicit_dt * (candidate - x)
            else:
                x = candidate

        if self.backbone.domain_padding is not None:
            x = self.backbone.domain_padding.unpad(x)

        x = self.backbone.projection(x)
        return x
