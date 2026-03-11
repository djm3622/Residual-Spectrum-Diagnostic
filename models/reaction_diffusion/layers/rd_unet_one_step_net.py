"""U-Net style layer stack for reaction-diffusion one-step prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import UNetResBlock


class RDUNetOneStepNet(nn.Module):
    """U-Net style one-step map with physics features and bounded update scale."""

    def __init__(self, width: int = 48):
        super().__init__()
        in_channels = 8
        w1 = int(width)
        w2 = int(width * 2)
        w3 = int(width * 4)

        self.enc0 = UNetResBlock(in_channels, w1, kernel_size=5)
        self.enc1 = UNetResBlock(w1, w2, kernel_size=5)
        self.enc2 = UNetResBlock(w2, w3, kernel_size=5)
        self.bottleneck = nn.Sequential(
            UNetResBlock(w3, w3, kernel_size=5),
            UNetResBlock(w3, w3, kernel_size=5),
        )
        self.dec1 = UNetResBlock(w3 + w2, w2, kernel_size=5)
        self.dec0 = UNetResBlock(w2 + w1, w1, kernel_size=5)
        self.spatial_out = nn.Conv2d(w1, 2, kernel_size=1)

        self.reaction = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(w1, w1, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(w1, 2, kernel_size=1),
        )

        self.diff_raw = nn.Parameter(torch.tensor([0.06, 0.06], dtype=torch.float32))
        self.step_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.step_max = 0.35

        with torch.no_grad():
            nn.init.zeros_(self.spatial_out.weight)
            nn.init.zeros_(self.spatial_out.bias)

    @staticmethod
    def _laplacian_periodic(field: torch.Tensor) -> torch.Tensor:
        return (
            torch.roll(field, shifts=1, dims=-2)
            + torch.roll(field, shifts=-1, dims=-2)
            + torch.roll(field, shifts=1, dims=-1)
            + torch.roll(field, shifts=-1, dims=-1)
            - 4.0 * field
        )

    @staticmethod
    def _downsample(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    @staticmethod
    def _upsample(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u = state[:, 0:1]
        v = state[:, 1:2]
        lap_u = self._laplacian_periodic(u)
        lap_v = self._laplacian_periodic(v)
        uv = u * v
        uv2 = uv * v
        features = torch.cat([u, v, lap_u, lap_v, uv, u * u, v * v, uv2], dim=1)

        e0 = self.enc0(features)
        e1 = self.enc1(self._downsample(e0))
        e2 = self.enc2(self._downsample(e1))
        b = self.bottleneck(e2)

        d1 = self._upsample(b, out_hw=(e1.shape[-2], e1.shape[-1]))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self._upsample(d1, out_hw=(e0.shape[-2], e0.shape[-1]))
        d0 = self.dec0(torch.cat([d0, e0], dim=1))

        spatial_delta = self.spatial_out(d0)
        reaction_delta = self.reaction(features)
        diff_delta = F.softplus(self.diff_raw).view(1, 2, 1, 1) * torch.cat([lap_u, lap_v], dim=1)

        step = self.step_max * torch.sigmoid(self.step_raw)
        delta = reaction_delta + diff_delta + 0.5 * spatial_delta
        next_state = state + step * delta
        return torch.clamp(next_state, 0.0, 1.0)
