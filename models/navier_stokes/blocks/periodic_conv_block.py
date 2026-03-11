"""Periodic convolution residual block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicConvBlock(nn.Module):
    """Residual periodic convolution block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.gelu(x + y)
