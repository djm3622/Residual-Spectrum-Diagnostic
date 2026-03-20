from typing import List, Tuple, Union, Literal

import torch.nn as nn
import torch.nn.functional as F

from .fno import FNO, partialclass
from ..layers.wavelet_convolution import WaveletConv

Number = Union[float, int]


class WNO(FNO, name="WNO"):
    """Wavelet Neural Operator (WNO) using wavelet-domain convolution kernels.

    This class keeps FNO-style API and infrastructure while replacing the Fourier
    convolution module with a wavelet convolution module.
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        blocks: int = None,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity=F.mish,
        norm: Literal["ada_in", "group_norm", "instance_norm", "batch_norm", None] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = False,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        # WNO-specific parameters
        wavelet: str = "db4",
        wavelet_levels: int = 2,
        wavelet_mode: str = "symmetric",
        wavelet_transform: str = "dwt",
        base_resolution: Union[Tuple[int, ...], List[int]] = None,
    ):
        if blocks is not None:
            n_layers = int(blocks)

        conv_module = partialclass(
            "ConfiguredWaveletConv",
            WaveletConv,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
            wavelet_mode=wavelet_mode,
            wavelet_transform=wavelet_transform,
            base_resolution=base_resolution,
        )

        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=resolution_scaling_factor,
            domain_padding=domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=False,
        )
