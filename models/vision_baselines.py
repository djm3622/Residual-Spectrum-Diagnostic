"""Shared vision baseline architectures for dense 2D field prediction."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import Swin_T_Weights, swin_t

    _TORCHVISION_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path
    Swin_T_Weights = None  # type: ignore[assignment]
    swin_t = None  # type: ignore[assignment]
    _TORCHVISION_IMPORT_ERROR = exc


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _resolve_group_norm_groups(channels: int) -> int:
    channels = max(1, int(channels))
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _build_norm(norm: str, channels: int) -> nn.Module:
    normalized = _normalize_name(norm)
    if normalized in {"none", "identity", "off"}:
        return nn.Identity()
    if normalized in {"batch_norm", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"instance_norm", "instancenorm", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    groups = _resolve_group_norm_groups(channels)
    return nn.GroupNorm(groups, channels)


def _build_activation(name: str) -> nn.Module:
    normalized = _normalize_name(name)
    if normalized in {"relu"}:
        return nn.ReLU(inplace=True)
    if normalized in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    return nn.GELU()


def _adapt_patch_embed_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    in_ch = max(1, int(in_channels))
    if conv.in_channels == in_ch:
        return conv

    replacement = nn.Conv2d(
        in_ch,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        source = conv.weight.detach()
        if in_ch < source.shape[1]:
            source = source[:, :in_ch, :, :]
        elif in_ch > source.shape[1]:
            repeat = int((in_ch + source.shape[1] - 1) // source.shape[1])
            source = source.repeat(1, repeat, 1, 1)[:, :in_ch, :, :]
        source = source * (float(conv.in_channels) / float(in_ch))
        replacement.weight.copy_(source)
        if conv.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(conv.bias.detach())
    return replacement


class SCSE2D(nn.Module):
    """Concurrent spatial + channel squeeze/excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        ch = max(1, int(channels))
        red = max(1, int(reduction))
        hidden = max(1, ch // red)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, kernel_size=1),
            nn.Gelu(inplace=True),
            nn.Conv2d(hidden, ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class ConvBlock2D(nn.Module):
    """Two-convolution residual-free block with configurable normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "group_norm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        p = max(0.0, float(dropout))
        act = _build_activation(activation)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _build_norm(norm, out_channels),
            act,
            nn.Dropout2d(p=p) if p > 0.0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _build_norm(norm, out_channels),
            _build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpFuseBlock2D(nn.Module):
    """Upsample + skip fusion block used by decoder stacks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "group_norm",
        activation: str = "gelu",
        dropout: float = 0.0,
        use_attention: bool = False,
        scse_reduction: int = 16,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(
            out_channels + max(0, int(skip_channels)),
            out_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.attn = SCSE2D(out_channels, reduction=scse_reduction) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.attn(self.conv(x))


class AttentionUNet2D(nn.Module):
    """U-Net style dense model with SCSE attention in decoder stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        norm: str = "group_norm",
        activation: str = "gelu",
        dropout: float = 0.0,
        scse_reduction: int = 16,
    ):
        super().__init__()
        base = max(8, int(base_channels))
        levels = max(2, min(5, int(depth)))
        channels = [base * (2**idx) for idx in range(levels + 1)]

        self.stem = ConvBlock2D(
            in_channels,
            channels[0],
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.encoder_down = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        for idx in range(levels):
            self.encoder_down.append(
                nn.Conv2d(channels[idx], channels[idx + 1], kernel_size=3, stride=2, padding=1)
            )
            self.encoder_blocks.append(
                ConvBlock2D(
                    channels[idx + 1],
                    channels[idx + 1],
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )

        self.bottleneck = ConvBlock2D(
            channels[-1],
            channels[-1],
            norm=norm,
            activation=activation,
            dropout=dropout,
        )

        self.decoder = nn.ModuleList()
        for idx in range(levels - 1, -1, -1):
            self.decoder.append(
                UpFuseBlock2D(
                    in_channels=channels[idx + 1],
                    skip_channels=channels[idx],
                    out_channels=channels[idx],
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    use_attention=True,
                    scse_reduction=scse_reduction,
                )
            )

        self.out_proj = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        skips = []
        h = self.stem(x)
        skips.append(h)

        for down, block in zip(self.encoder_down, self.encoder_blocks):
            h = block(down(h))
            skips.append(h)

        h = self.bottleneck(skips[-1])
        for idx, dec in enumerate(self.decoder):
            skip = skips[-2 - idx] if idx < len(skips) - 1 else None
            h = dec(h, skip=skip)

        out = self.out_proj(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class SwinDenseFieldModel2D(nn.Module):
    """Swin-T backbone with lightweight decoder for dense field regression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        decoder_channels: Sequence[int] | None = None,
        norm: str = "group_norm",
        activation: str = "gelu",
        dropout: float = 0.0,
        use_attention: bool = True,
        scse_reduction: int = 16,
    ):
        super().__init__()
        if swin_t is None:
            message = (
                "Method 'swin' requires torchvision with Swin support. "
                "Install/upgrade torchvision to use this baseline."
            )
            if _TORCHVISION_IMPORT_ERROR is not None:
                raise ImportError(f"{message}. Import error: {_TORCHVISION_IMPORT_ERROR}") from _TORCHVISION_IMPORT_ERROR
            raise ImportError(message)

        weights = None
        if bool(pretrained):
            if Swin_T_Weights is None:
                raise ImportError("Pretrained Swin weights requested but torchvision Swin weights are unavailable.")
            weights = Swin_T_Weights.IMAGENET1K_V1
        backbone = swin_t(weights=weights)
        self.features = backbone.features

        patch_embed = self.features[0][0]
        if not isinstance(patch_embed, nn.Conv2d):
            raise TypeError("Unexpected Swin patch-embedding module type.")
        self.features[0][0] = _adapt_patch_embed_conv(patch_embed, in_channels)

        if bool(freeze_backbone):
            for param in self.features.parameters():
                param.requires_grad = False

        dec = list(decoder_channels) if decoder_channels is not None else [384, 192, 96, 64, 64]
        if len(dec) < 5:
            raise ValueError("swin.decoder_channels must provide at least five values.")
        d3, d2, d1, d0a, d0b = [max(16, int(v)) for v in dec[:5]]

        self.decode3 = UpFuseBlock2D(
            in_channels=768,
            skip_channels=384,
            out_channels=d3,
            norm=norm,
            activation=activation,
            dropout=dropout,
            use_attention=use_attention,
            scse_reduction=scse_reduction,
        )
        self.decode2 = UpFuseBlock2D(
            in_channels=d3,
            skip_channels=192,
            out_channels=d2,
            norm=norm,
            activation=activation,
            dropout=dropout,
            use_attention=use_attention,
            scse_reduction=scse_reduction,
        )
        self.decode1 = UpFuseBlock2D(
            in_channels=d2,
            skip_channels=96,
            out_channels=d1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            use_attention=use_attention,
            scse_reduction=scse_reduction,
        )
        self.decode0a = UpFuseBlock2D(
            in_channels=d1,
            skip_channels=0,
            out_channels=d0a,
            norm=norm,
            activation=activation,
            dropout=dropout,
            use_attention=use_attention,
            scse_reduction=scse_reduction,
        )
        self.decode0b = UpFuseBlock2D(
            in_channels=d0a,
            skip_channels=0,
            out_channels=d0b,
            norm=norm,
            activation=activation,
            dropout=dropout,
            use_attention=use_attention,
            scse_reduction=scse_reduction,
        )
        self.out_proj = nn.Conv2d(d0b, out_channels, kernel_size=1)

    @staticmethod
    def _bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}.")
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]
        y0 = self.features[0](x)
        y1 = self.features[1](y0)  # [B, H/4, W/4, 96]
        y2 = self.features[2](y1)
        y3 = self.features[3](y2)  # [B, H/8, W/8, 192]
        y4 = self.features[4](y3)
        y5 = self.features[5](y4)  # [B, H/16, W/16, 384]
        y6 = self.features[6](y5)
        y7 = self.features[7](y6)  # [B, H/32, W/32, 768]

        s1 = self._bchw(y1)
        s2 = self._bchw(y3)
        s3 = self._bchw(y5)
        h = self._bchw(y7)

        h = self.decode3(h, s3)
        h = self.decode2(h, s2)
        h = self.decode1(h, s1)
        h = self.decode0a(h, None)
        h = self.decode0b(h, None)
        out = self.out_proj(h)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class SingleChannelFieldWrapper(nn.Module):
    """Adapter for models consuming [B,1,H,W] and producing [B,1,H,W]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected [B,H,W] tensor for NS wrapper, got {tuple(x.shape)}.")
        y = self.model(x.unsqueeze(1))
        if y.ndim != 4 or int(y.shape[1]) != 1:
            raise ValueError(f"Wrapped model returned unexpected output shape {tuple(y.shape)}.")
        return y[:, 0]


def resolve_baseline_config(
    architecture: str,
    baseline_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Merge baseline-model config sections (common + architecture-specific)."""
    merged: Dict[str, Any] = {}
    if not isinstance(baseline_config, Mapping):
        return merged

    common = baseline_config.get("common")
    if isinstance(common, Mapping):
        merged.update(common)

    key = _normalize_name(architecture)
    aliases = {key}
    if key == "swin":
        aliases.update({"swin_transformer", "swin_t"})
    if key in {"attn_unet", "attention_unet"}:
        aliases.update({"attn_unet", "attention_unet", "unet_attn"})
    if key in {"legacy_conv", "conv"}:
        aliases.update({"legacy_conv", "conv"})

    for alias in aliases:
        section = baseline_config.get(alias)
        if isinstance(section, Mapping):
            merged.update(section)
    return merged


def _as_int_list(value: Any, default: Sequence[int]) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return [int(v) for v in default]
    result = []
    for item in value:
        try:
            result.append(int(item))
        except Exception:
            continue
    return result if result else [int(v) for v in default]


def build_dense_field_model(
    architecture: str,
    in_channels: int,
    out_channels: int,
    config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Construct a dense field model for the requested baseline architecture."""
    arch = _normalize_name(architecture)
    cfg = dict(config or {})
    norm = str(cfg.get("norm", "group_norm"))
    activation = str(cfg.get("activation", "gelu"))
    dropout = float(cfg.get("dropout", 0.0))
    scse_reduction = max(1, int(cfg.get("scse_reduction", 16)))

    if arch in {"attn_unet", "attention_unet", "unet_attn"}:
        return AttentionUNet2D(
            in_channels=max(1, int(in_channels)),
            out_channels=max(1, int(out_channels)),
            base_channels=max(8, int(cfg.get("base_channels", 64))),
            depth=max(2, int(cfg.get("depth", 4))),
            norm=norm,
            activation=activation,
            dropout=dropout,
            scse_reduction=scse_reduction,
        )

    if arch in {"swin", "swin_transformer", "swin_t"}:
        decoder_channels = _as_int_list(cfg.get("decoder_channels"), default=[384, 192, 96, 64, 64])
        return SwinDenseFieldModel2D(
            in_channels=max(1, int(in_channels)),
            out_channels=max(1, int(out_channels)),
            pretrained=bool(cfg.get("pretrained", False)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            decoder_channels=decoder_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
            use_attention=bool(cfg.get("use_attention", True)),
            scse_reduction=scse_reduction,
        )

    raise ValueError(
        f"Unsupported baseline architecture '{architecture}'. "
        "Use one of: swin, attn_unet."
    )
