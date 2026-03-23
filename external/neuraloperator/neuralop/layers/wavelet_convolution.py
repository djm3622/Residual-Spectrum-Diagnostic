import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .base_spectral_conv import BaseSpectralConv
from .resample import resample
from ..utils import validate_scaling_factor

Number = Union[int, float]


def _require_pytorch_wavelets_2d():
    try:
        from pytorch_wavelets import DWT, IDWT
    except ImportError:
        try:
            from pytorch_wavelets import DWTForward as DWT
            from pytorch_wavelets import DWTInverse as IDWT
        except ImportError as exc:
            raise ImportError(
                "WaveletConv2d requires optional dependency 'pytorch-wavelets'. "
                "Install with: pip install neuraloperator[wavelets]"
            ) from exc
    return DWT, IDWT


def _require_pytorch_wavelets_1d():
    try:
        from pytorch_wavelets import DWT1D, IDWT1D
    except ImportError:
        try:
            from pytorch_wavelets import DWT1DForward as DWT1D
            from pytorch_wavelets import DWT1DInverse as IDWT1D
        except ImportError as exc:
            raise ImportError(
                "WaveletConv1d requires optional dependency 'pytorch-wavelets'. "
                "Install with: pip install neuraloperator[wavelets]"
            ) from exc
    return DWT1D, IDWT1D


def _require_ptwt_pywt():
    try:
        import pywt
        from ptwt.conv_transform_3 import wavedec3, waverec3
    except ImportError as exc:
        raise ImportError(
            "WaveletConv3d requires optional dependencies 'ptwt' and 'PyWavelets'. "
            "Install with: pip install neuraloperator[wavelets3d]"
        ) from exc
    return pywt, wavedec3, waverec3


class _WaveletConvBase(BaseSpectralConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int, ...], List[int]],
        order: int,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        bias: bool = True,
        device=None,
    ):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = order
        self.n_modes = n_modes
        self.resolution_scaling_factor = validate_scaling_factor(
            resolution_scaling_factor, self.order
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, *([1] * self.order)))
        else:
            self.bias = None

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes] * self.order
        else:
            n_modes = list(n_modes)
        if len(n_modes) != self.order:
            raise ValueError(
                f"Expected {self.order} mode dimensions, got {len(n_modes)} from n_modes={n_modes}"
            )
        self._n_modes = [int(m) for m in n_modes]

    def _effective_levels(self, spatial_shape: Tuple[int, ...], base_resolution):
        if base_resolution is None:
            return None
        ref = base_resolution[-1]
        cur = spatial_shape[-1]
        if ref <= 0 or cur <= 0:
            return None
        if cur > ref:
            factor = int(math.log2(cur // ref))
            return factor
        if cur < ref:
            factor = int(math.log2(ref // cur))
            return -factor
        return 0

    def _get_output_shape(self, in_shape: Tuple[int, ...], output_shape=None):
        if output_shape is not None:
            return tuple(output_shape)
        if self.resolution_scaling_factor is not None:
            return tuple(
                round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)
            )
        return in_shape

    def transform(self, x, output_shape=None):
        in_shape = tuple(x.shape[2:])
        out_shape = self._get_output_shape(in_shape, output_shape=output_shape)
        if in_shape == out_shape:
            return x
        return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def _apply_bias(self, x):
        if self.bias is not None:
            return x + self.bias
        return x


class WaveletConv1d(_WaveletConvBase):
    """1D WNO wavelet convolution (DWT), mirroring reference WNO logic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int], List[int]],
        *,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
        wavelet_mode: str = "symmetric",
        base_resolution: Optional[Union[int, Tuple[int]]] = None,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        bias: bool = True,
        max_n_modes=None,
        rank: float = 1.0,
        factorization=None,
        implementation: str = "factorized",
        fixed_rank_modes: bool = False,
        separable: bool = False,
        fno_block_precision: str = "full",
        decomposition_kwargs: Optional[dict] = None,
        complex_data: bool = False,
        device=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            order=1,
            resolution_scaling_factor=resolution_scaling_factor,
            bias=bias,
            device=device,
        )
        if complex_data:
            raise ValueError("WaveletConv1d currently supports real-valued inputs only.")
        if separable and in_channels != out_channels:
            raise ValueError(
                "WaveletConv1d with separable=True requires in_channels == out_channels."
            )

        self.wavelet = wavelet
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_mode = wavelet_mode
        if base_resolution is None:
            self.base_resolution = None
        elif isinstance(base_resolution, int):
            self.base_resolution = (base_resolution,)
        else:
            self.base_resolution = tuple(base_resolution)
        self._DWT1D, self._IDWT1D = _require_pytorch_wavelets_1d()

        m1 = self.n_modes[0]
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1))

    def _mul1d(self, x, weight):
        return torch.einsum("bix,iox->box", x, weight)

    def _active_modes(self, coeff_shape, weight_shape):
        return min(coeff_shape[-1], weight_shape[-1], self.n_modes[0])

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        if x.ndim != 3:
            raise ValueError(f"WaveletConv1d expects 3D input [B,C,L], got shape {tuple(x.shape)}")

        level_offset = self._effective_levels(tuple(x.shape[-1:]), self.base_resolution)
        levels = self.wavelet_levels if level_offset is None else max(1, self.wavelet_levels + level_offset)

        dwt = self._DWT1D(wave=self.wavelet, J=levels, mode=self.wavelet_mode).to(x.device)
        x_ft, x_coeff = dwt(x)

        out_ft = torch.zeros(
            x_ft.shape[0],
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=x_ft.dtype,
        )
        out_coeff = [
            torch.zeros(
                coeffs.shape[0],
                self.out_channels,
                coeffs.shape[-1],
                device=x.device,
                dtype=coeffs.dtype,
            )
            for coeffs in x_coeff
        ]

        m1 = self._active_modes(x_ft.shape, self.weights1.shape)
        out_ft[:, :, :m1] = self._mul1d(x_ft[:, :, :m1], self.weights1[:, :, :m1])

        m1 = self._active_modes(x_coeff[-1].shape, self.weights2.shape)
        out_coeff[-1][:, :, :m1] = self._mul1d(
            x_coeff[-1][:, :, :m1], self.weights2[:, :, :m1]
        )

        idwt = self._IDWT1D(wave=self.wavelet, mode=self.wavelet_mode).to(x.device)
        x = idwt((out_ft, out_coeff))
        x = self._apply_bias(x)

        in_shape = tuple(x.shape[2:])
        out_shape = self._get_output_shape(in_shape, output_shape=output_shape)
        if in_shape != out_shape:
            x = resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
        return x


class WaveletConv2d(_WaveletConvBase):
    """2D WNO wavelet convolution (DWT), mirroring reference WNO logic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int, int], List[int]],
        *,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
        wavelet_mode: str = "symmetric",
        base_resolution: Optional[Tuple[int, int]] = None,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        bias: bool = True,
        max_n_modes=None,
        rank: float = 1.0,
        factorization=None,
        implementation: str = "factorized",
        fixed_rank_modes: bool = False,
        separable: bool = False,
        fno_block_precision: str = "full",
        decomposition_kwargs: Optional[dict] = None,
        complex_data: bool = False,
        device=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            order=2,
            resolution_scaling_factor=resolution_scaling_factor,
            bias=bias,
            device=device,
        )
        if complex_data:
            raise ValueError("WaveletConv2d currently supports real-valued inputs only.")
        if separable and in_channels != out_channels:
            raise ValueError(
                "WaveletConv2d with separable=True requires in_channels == out_channels."
            )

        self.wavelet = wavelet
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_mode = wavelet_mode
        self.base_resolution = base_resolution
        self._DWT, self._IDWT = _require_pytorch_wavelets_2d()

        m1, m2 = self.n_modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2))

    def _mul2d(self, x, weight):
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def _active_modes(self, coeff_shape, weight_shape):
        m1 = min(coeff_shape[-2], weight_shape[-2], self.n_modes[0])
        m2 = min(coeff_shape[-1], weight_shape[-1], self.n_modes[1])
        return m1, m2

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, int]] = None):
        if x.ndim != 4:
            raise ValueError(f"WaveletConv2d expects 4D input [B,C,H,W], got shape {tuple(x.shape)}")

        level_offset = self._effective_levels(tuple(x.shape[-2:]), self.base_resolution)
        levels = self.wavelet_levels if level_offset is None else max(1, self.wavelet_levels + level_offset)

        dwt = self._DWT(J=levels, mode=self.wavelet_mode, wave=self.wavelet).to(x.device)
        x_ft, x_coeff = dwt(x)

        out_ft = torch.zeros(
            x_ft.shape[0],
            self.out_channels,
            x_ft.shape[-2],
            x_ft.shape[-1],
            device=x.device,
            dtype=x_ft.dtype,
        )
        out_coeff = [
            torch.zeros(
                coeffs.shape[0],
                self.out_channels,
                coeffs.shape[2],
                coeffs.shape[3],
                coeffs.shape[4],
                device=x.device,
                dtype=coeffs.dtype,
            )
            for coeffs in x_coeff
        ]

        m1, m2 = self._active_modes(x_ft.shape, self.weights1.shape)
        out_ft[:, :, :m1, :m2] = self._mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )

        detail_weights = (self.weights2, self.weights3, self.weights4)
        for detail_idx, weight in enumerate(detail_weights):
            m1, m2 = self._active_modes(x_coeff[-1][:, :, detail_idx].shape, weight.shape)
            out_coeff[-1][:, :, detail_idx, :m1, :m2] = self._mul2d(
                x_coeff[-1][:, :, detail_idx, :m1, :m2], weight[:, :, :m1, :m2]
            )

        idwt = self._IDWT(mode=self.wavelet_mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))
        x = self._apply_bias(x)

        in_shape = tuple(x.shape[2:])
        out_shape = self._get_output_shape(in_shape, output_shape=output_shape)
        if in_shape != out_shape:
            x = resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
        return x


class WaveletConv3d(_WaveletConvBase):
    """3D WNO wavelet convolution (DWT), mirroring reference WNO logic."""

    _detail_keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int, int, int], List[int]],
        *,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
        wavelet_mode: str = "periodic",
        base_resolution: Optional[Tuple[int, int, int]] = None,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        bias: bool = True,
        max_n_modes=None,
        rank: float = 1.0,
        factorization=None,
        implementation: str = "factorized",
        fixed_rank_modes: bool = False,
        separable: bool = False,
        fno_block_precision: str = "full",
        decomposition_kwargs: Optional[dict] = None,
        complex_data: bool = False,
        device=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            order=3,
            resolution_scaling_factor=resolution_scaling_factor,
            bias=bias,
            device=device,
        )
        if complex_data:
            raise ValueError("WaveletConv3d currently supports real-valued inputs only.")
        if separable and in_channels != out_channels:
            raise ValueError(
                "WaveletConv3d with separable=True requires in_channels == out_channels."
            )

        self.wavelet = wavelet
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_mode = wavelet_mode
        self.base_resolution = base_resolution
        pywt, wavedec3, waverec3 = _require_ptwt_pywt()
        self._pywt = pywt
        self._wavedec3 = wavedec3
        self._waverec3 = waverec3
        self._wavelet_obj = pywt.Wavelet(self.wavelet)

        m1, m2, m3 = self.n_modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights5 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights6 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights7 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))
        self.weights8 = nn.Parameter(scale * torch.rand(in_channels, out_channels, m1, m2, m3))

    def _mul3d(self, x, weight):
        if x.ndim == 4:
            return torch.einsum("ixyz,ioxyz->oxyz", x, weight)
        if x.ndim == 5:
            return torch.einsum("bixyz,ioxyz->boxyz", x, weight)
        raise ValueError(f"Expected 4D or 5D coeff tensor, got shape {tuple(x.shape)}")

    def _active_modes(self, coeff_shape, weight_shape):
        m1 = min(coeff_shape[-3], weight_shape[-3], self.n_modes[0])
        m2 = min(coeff_shape[-2], weight_shape[-2], self.n_modes[1])
        m3 = min(coeff_shape[-1], weight_shape[-1], self.n_modes[2])
        return m1, m2, m3

    def _apply_weight_3d(self, coeff, weight):
        if coeff.ndim == 4:
            out = torch.zeros(
                self.out_channels,
                coeff.shape[-3],
                coeff.shape[-2],
                coeff.shape[-1],
                device=coeff.device,
                dtype=coeff.dtype,
            )
        elif coeff.ndim == 5:
            out = torch.zeros(
                coeff.shape[0],
                self.out_channels,
                coeff.shape[-3],
                coeff.shape[-2],
                coeff.shape[-1],
                device=coeff.device,
                dtype=coeff.dtype,
            )
        else:
            raise ValueError(f"Expected 4D or 5D coeff tensor, got shape {tuple(coeff.shape)}")
        m1, m2, m3 = self._active_modes(coeff.shape, weight.shape)
        if coeff.ndim == 4:
            out[:, :m1, :m2, :m3] = self._mul3d(
                coeff[:, :m1, :m2, :m3], weight[:, :, :m1, :m2, :m3]
            )
        else:
            out[:, :, :m1, :m2, :m3] = self._mul3d(
                coeff[:, :, :m1, :m2, :m3], weight[:, :, :m1, :m2, :m3]
            )
        return out

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, int, int]] = None):
        if x.ndim != 5:
            raise ValueError(
                f"WaveletConv3d expects 5D input [B,C,D,H,W], got shape {tuple(x.shape)}"
            )

        level_offset = self._effective_levels(tuple(x.shape[-3:]), self.base_resolution)
        levels = self.wavelet_levels if level_offset is None else max(1, self.wavelet_levels + level_offset)

        detail_weights = (
            self.weights2,
            self.weights3,
            self.weights4,
            self.weights5,
            self.weights6,
            self.weights7,
            self.weights8,
        )

        coeffs = list(self._wavedec3(x, self._wavelet_obj, level=levels, mode=self.wavelet_mode))
        coeffs[0] = self._apply_weight_3d(coeffs[0], self.weights1)
        for key, weight in zip(self._detail_keys, detail_weights):
            coeffs[1][key] = self._apply_weight_3d(coeffs[1][key], weight)
        for j in range(2, len(coeffs)):
            coeffs[j] = {k: torch.zeros_like(v) for k, v in coeffs[j].items()}

        xr = self._waverec3(coeffs, self._wavelet_obj)

        xr = self._apply_bias(xr)

        in_shape = tuple(xr.shape[2:])
        out_shape = self._get_output_shape(in_shape, output_shape=output_shape)
        if in_shape != out_shape:
            xr = resample(xr, 1.0, list(range(2, xr.ndim)), output_shape=out_shape)
        return xr


class WaveletConv(BaseSpectralConv):
    """Dimension-dispatch wavelet convolution for WNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, Tuple[int, ...], List[int]],
        *,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
        wavelet_mode: str = "symmetric",
        wavelet_transform: str = "dwt",
        base_resolution: Optional[Tuple[int, ...]] = None,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        bias: bool = True,
        max_n_modes=None,
        rank: float = 1.0,
        factorization=None,
        implementation: str = "factorized",
        fixed_rank_modes: bool = False,
        separable: bool = False,
        fno_block_precision: str = "full",
        decomposition_kwargs: Optional[dict] = None,
        complex_data: bool = False,
        device=None,
        **kwargs,
    ):
        super().__init__(device=device)
        if wavelet_transform.lower() != "dwt":
            raise ValueError(
                f"WaveletConv currently supports wavelet_transform='dwt' only, got '{wavelet_transform}'."
            )

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        n_dim = len(n_modes)
        if n_dim == 1:
            self.conv = WaveletConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                n_modes=n_modes,
                wavelet=wavelet,
                wavelet_levels=wavelet_levels,
                wavelet_mode=wavelet_mode,
                base_resolution=base_resolution,
                resolution_scaling_factor=resolution_scaling_factor,
                bias=bias,
                max_n_modes=max_n_modes,
                rank=rank,
                factorization=factorization,
                implementation=implementation,
                fixed_rank_modes=fixed_rank_modes,
                separable=separable,
                fno_block_precision=fno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data,
                device=device,
                **kwargs,
            )
        elif n_dim == 2:
            self.conv = WaveletConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                n_modes=n_modes,
                wavelet=wavelet,
                wavelet_levels=wavelet_levels,
                wavelet_mode=wavelet_mode,
                base_resolution=base_resolution,
                resolution_scaling_factor=resolution_scaling_factor,
                bias=bias,
                max_n_modes=max_n_modes,
                rank=rank,
                factorization=factorization,
                implementation=implementation,
                fixed_rank_modes=fixed_rank_modes,
                separable=separable,
                fno_block_precision=fno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data,
                device=device,
                **kwargs,
            )
        elif n_dim == 3:
            self.conv = WaveletConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                n_modes=n_modes,
                wavelet=wavelet,
                wavelet_levels=wavelet_levels,
                wavelet_mode=wavelet_mode,
                base_resolution=base_resolution,
                resolution_scaling_factor=resolution_scaling_factor,
                bias=bias,
                max_n_modes=max_n_modes,
                rank=rank,
                factorization=factorization,
                implementation=implementation,
                fixed_rank_modes=fixed_rank_modes,
                separable=separable,
                fno_block_precision=fno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data,
                device=device,
                **kwargs,
            )
        else:
            raise ValueError(
                f"WaveletConv currently supports 1D, 2D and 3D only, got len(n_modes)={n_dim}."
            )

    @property
    def n_modes(self):
        return self.conv.n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.conv.n_modes = n_modes

    def transform(self, x, output_shape=None):
        return self.conv.transform(x, output_shape=output_shape)

    def forward(self, x, output_shape=None):
        return self.conv(x, output_shape=output_shape)
