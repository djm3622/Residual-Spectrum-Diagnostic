"""Neural network layer stack for Navier-Stokes one-step prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import PeriodicConvBlock


class NSNonlinearOneStepNet(nn.Module):
    """One-step nonlinear residual map: omega_{t+1} = omega_t + Delta(omega_t)."""

    def __init__(self, width: int = 64, depth: int = 5):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList([PeriodicConvBlock(width) for _ in range(depth)])
        self.out_proj = nn.Conv2d(width, 1, kernel_size=1)
        # Learn residual scale from near-identity initialization for stable rollout.
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        with torch.no_grad():
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        # omega: [batch, nx, ny]
        x = omega.unsqueeze(1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        delta = self.out_proj(h).squeeze(1)
        return omega + self.residual_scale * delta
