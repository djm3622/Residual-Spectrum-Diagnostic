import pytest
import torch

from neuralop.models import WNO
from neuralop.models.base_model import get_model

from config.reaction_diffusion_2d_config import Default as RD2DDefault
from config.reaction_diffusion_3d_config import Default as RD3DDefault


try:
    import pytorch_wavelets  # noqa: F401

    HAS_PYTORCH_WAVELETS = True
except ImportError:
    HAS_PYTORCH_WAVELETS = False

try:
    import ptwt  # noqa: F401
    import pywt  # noqa: F401

    HAS_PTWT_3D = True
except ImportError:
    HAS_PTWT_3D = False


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_rd2d_config_get_model_and_forward():
    cfg = RD2DDefault().to_dict()
    model = get_model(cfg)
    assert isinstance(model, WNO)

    x = torch.randn(
        2,
        cfg.model.data_channels,
        cfg.data.spatial_size,
        cfg.data.spatial_size,
    )
    y = model(x)
    assert y.shape == (
        2,
        cfg.model.out_channels,
        cfg.data.spatial_size,
        cfg.data.spatial_size,
    )


@pytest.mark.skipif(
    not (HAS_PYTORCH_WAVELETS and HAS_PTWT_3D),
    reason="3D wavelet deps not installed (ptwt + PyWavelets + pytorch-wavelets)",
)
def test_rd3d_config_get_model_and_forward():
    cfg = RD3DDefault().to_dict()
    model = get_model(cfg)
    assert isinstance(model, WNO)

    x = torch.randn(
        1,
        cfg.model.data_channels,
        cfg.data.block_size,
        cfg.data.spatial_size,
        cfg.data.spatial_size,
    )
    y = model(x)
    assert y.shape == (
        1,
        cfg.model.out_channels,
        cfg.data.block_size,
        cfg.data.spatial_size,
        cfg.data.spatial_size,
    )
