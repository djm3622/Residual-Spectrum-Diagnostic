import pytest
import torch

from neuralop.models import WNO
from neuralop.models.base_model import available_models, get_model


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


def test_wno_registration():
    assert "wno" in available_models()


class _NoPatchingConfig:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, key):
        raise KeyError(key)


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wno_1d_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WNO(
        n_modes=(16,),
        in_channels=2,
        out_channels=1,
        hidden_channels=12,
        n_layers=2,
        wavelet_levels=2,
        wavelet="db4",
    ).to(device)

    x = torch.randn(2, 2, 64, device=device)
    y = model(x)
    assert y.shape == (2, 1, 64)

    y.sum().backward()
    assert all(param.grad is not None for param in model.parameters())


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wno_2d_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WNO(
        n_modes=(8, 8),
        in_channels=3,
        out_channels=1,
        hidden_channels=16,
        n_layers=2,
        wavelet_levels=2,
        wavelet="db4",
    ).to(device)

    x = torch.randn(2, 3, 32, 32, device=device)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)

    y.sum().backward()
    assert all(param.grad is not None for param in model.parameters())


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_get_model_dispatch_wno():
    config = _NoPatchingConfig(
        model={
            "model_arch": "wno",
            "data_channels": 2,
            "out_channels": 1,
            "n_modes": [8, 8],
            "hidden_channels": 8,
            "n_layers": 2,
            "wavelet_levels": 2,
        }
    )
    model = get_model(config)
    assert isinstance(model, WNO)


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_get_model_dispatch_wno_with_blocks_alias():
    config = _NoPatchingConfig(
        model={
            "model_arch": "wno",
            "data_channels": 2,
            "out_channels": 1,
            "n_modes": [8, 8],
            "hidden_channels": 8,
            "blocks": 3,
            "wavelet_levels": 2,
        }
    )
    model = get_model(config)
    assert isinstance(model, WNO)
    assert model.n_layers == 3


@pytest.mark.skipif(
    not (HAS_PYTORCH_WAVELETS and HAS_PTWT_3D),
    reason="3D wavelet deps not installed (ptwt + PyWavelets + pytorch-wavelets)",
)
def test_wno_3d_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WNO(
        n_modes=(4, 4, 4),
        in_channels=2,
        out_channels=1,
        hidden_channels=8,
        n_layers=2,
        wavelet_levels=1,
        wavelet_mode="periodic",
    ).to(device)

    x = torch.randn(1, 2, 8, 8, 8, device=device)
    y = model(x)
    assert y.shape == (1, 1, 8, 8, 8)

    y.sum().backward()
    assert all(param.grad is not None for param in model.parameters())
