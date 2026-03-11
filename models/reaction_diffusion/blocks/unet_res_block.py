"""Residual U-Net block for coupled species fields."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetResBlock(nn.Module):
    """Residual block with periodic convolutions and wider kernels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode="circular",
        )
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.gelu(residual + y)
