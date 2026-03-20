import pytest
import torch

from ..wavelet_convolution import WaveletConv, WaveletConv2d, WaveletConv3d


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
def test_wavelet_conv1d_shape_and_grad():
    from ..wavelet_convolution import WaveletConv1d

    conv = WaveletConv1d(
        in_channels=3,
        out_channels=5,
        n_modes=(16,),
        wavelet="db4",
        wavelet_levels=2,
    )
    x = torch.randn(2, 3, 64, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 5, 64)

    y.sum().backward()
    assert all(param.grad is not None for param in conv.parameters())


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wavelet_conv2d_shape_and_grad():
    conv = WaveletConv2d(
        in_channels=4,
        out_channels=6,
        n_modes=(8, 8),
        wavelet="db4",
        wavelet_levels=2,
    )
    x = torch.randn(2, 4, 32, 32, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 6, 32, 32)

    y.sum().backward()
    assert all(param.grad is not None for param in conv.parameters())


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wavelet_conv2d_resolution_scaling():
    conv = WaveletConv2d(
        in_channels=2,
        out_channels=2,
        n_modes=(8, 8),
        wavelet_levels=2,
        resolution_scaling_factor=0.5,
    )
    x = torch.randn(1, 2, 32, 32)
    y = conv(x)
    assert y.shape == (1, 2, 16, 16)


@pytest.mark.skipif(
    not (HAS_PYTORCH_WAVELETS and HAS_PTWT_3D),
    reason="3D wavelet deps not installed (ptwt + PyWavelets + pytorch-wavelets)",
)
def test_wavelet_conv3d_shape_and_grad():
    conv = WaveletConv3d(
        in_channels=2,
        out_channels=2,
        n_modes=(4, 4, 4),
        wavelet="db4",
        wavelet_levels=1,
        wavelet_mode="periodic",
    )
    x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    y = conv(x)
    assert y.shape == (1, 2, 8, 8, 8)

    y.sum().backward()
    assert all(param.grad is not None for param in conv.parameters())


def test_wavelet_conv_missing_dep_error(monkeypatch):
    import neuralop.layers.wavelet_convolution as wc

    def _missing():
        raise ImportError("mock missing pytorch-wavelets")

    monkeypatch.setattr(wc, "_require_pytorch_wavelets_2d", _missing)
    with pytest.raises(ImportError, match="pytorch-wavelets"):
        wc.WaveletConv2d(2, 2, n_modes=(4, 4))


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wavelet_conv_dispatch_1d():
    conv = WaveletConv(2, 2, n_modes=(16,))
    x = torch.randn(1, 2, 64)
    y = conv(x)
    assert y.shape == (1, 2, 64)


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wavelet_conv_dispatch_2d():
    conv = WaveletConv(2, 2, n_modes=(8, 8))
    x = torch.randn(1, 2, 16, 16)
    y = conv(x)
    assert y.shape == (1, 2, 16, 16)
