import importlib.util
import os
from pathlib import Path

import pytest
import torch

from ..wavelet_convolution import WaveletConv2d, WaveletConv3d


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


def _load_reference_wavelet_module():
    root = os.environ.get("WNO_REFERENCE_PATH")
    if not root:
        pytest.skip("WNO_REFERENCE_PATH is not set; skipping reference parity tests.")
    ref_file = Path(root) / "Version 2.0.0" / "wavelet_convolution.py"
    if not ref_file.exists():
        pytest.skip(f"Reference wavelet file not found at {ref_file}")

    spec = importlib.util.spec_from_file_location("wno_reference_wavelet_convolution", ref_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_weights_2d(ref_layer, test_layer):
    test_layer.weights1.data.copy_(ref_layer.weights1.data)
    test_layer.weights2.data.copy_(ref_layer.weights2.data)
    test_layer.weights3.data.copy_(ref_layer.weights3.data)
    test_layer.weights4.data.copy_(ref_layer.weights4.data)


def _copy_weights_3d(ref_layer, test_layer):
    for idx in range(1, 9):
        getattr(test_layer, f"weights{idx}").data.copy_(getattr(ref_layer, f"weights{idx}").data)


def _reference_waveconv3d_forward_compat(ref_mod, ref_layer, x):
    # Equivalent to reference forward with a compatibility fix for ptwt returning tuples.
    xr = torch.zeros(x.shape, device=x.device)
    for i in range(x.shape[0]):
        if x.shape[-1] > ref_layer.size[-1]:
            factor = int(ref_mod.np.log2(x.shape[-1] // ref_layer.size[-1]))
            coeffs = list(
                ref_mod.wavedec3(
                    x[i, ...],
                    ref_mod.pywt.Wavelet(ref_layer.wavelet),
                    level=ref_layer.level + factor,
                    mode=ref_layer.mode,
                )
            )
        elif x.shape[-1] < ref_layer.size[-1]:
            factor = int(ref_mod.np.log2(ref_layer.size[-1] // x.shape[-1]))
            coeffs = list(
                ref_mod.wavedec3(
                    x[i, ...],
                    ref_mod.pywt.Wavelet(ref_layer.wavelet),
                    level=ref_layer.level - factor,
                    mode=ref_layer.mode,
                )
            )
        else:
            coeffs = list(
                ref_mod.wavedec3(
                    x[i, ...],
                    ref_mod.pywt.Wavelet(ref_layer.wavelet),
                    level=ref_layer.level,
                    mode=ref_layer.mode,
                )
            )

        coeffs[0] = ref_layer.mul3d(coeffs[0].clone(), ref_layer.weights1)
        coeffs[1]["aad"] = ref_layer.mul3d(coeffs[1]["aad"].clone(), ref_layer.weights2)
        coeffs[1]["ada"] = ref_layer.mul3d(coeffs[1]["ada"].clone(), ref_layer.weights3)
        coeffs[1]["add"] = ref_layer.mul3d(coeffs[1]["add"].clone(), ref_layer.weights4)
        coeffs[1]["daa"] = ref_layer.mul3d(coeffs[1]["daa"].clone(), ref_layer.weights5)
        coeffs[1]["dad"] = ref_layer.mul3d(coeffs[1]["dad"].clone(), ref_layer.weights6)
        coeffs[1]["dda"] = ref_layer.mul3d(coeffs[1]["dda"].clone(), ref_layer.weights7)
        coeffs[1]["ddd"] = ref_layer.mul3d(coeffs[1]["ddd"].clone(), ref_layer.weights8)

        for jj in range(2, ref_layer.level + 1):
            if jj < len(coeffs):
                coeffs[jj] = {
                    key: torch.zeros([*coeffs[jj][key].shape], device=x.device)
                    for key in coeffs[jj].keys()
                }

        xr[i, ...] = ref_mod.waverec3(coeffs, ref_mod.pywt.Wavelet(ref_layer.wavelet))

    return xr


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wavelet_conv2d_reference_parity():
    ref_mod = _load_reference_wavelet_module()
    in_channels, out_channels = 4, 4
    level = 2
    size = [32, 32]
    wavelet = "db4"

    ref_layer = ref_mod.WaveConv2d(in_channels, out_channels, level, size, wavelet)
    n_modes = tuple(ref_layer.weights1.shape[-2:])
    test_layer = WaveletConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        wavelet=wavelet,
        wavelet_levels=level,
        base_resolution=tuple(size),
        bias=False,
    )
    _copy_weights_2d(ref_layer, test_layer)

    x = torch.randn(2, in_channels, *size)
    y_ref = ref_layer(x)
    y_test = test_layer(x)
    torch.testing.assert_close(y_test, y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not (HAS_PYTORCH_WAVELETS and HAS_PTWT_3D),
    reason="3D wavelet deps not installed (ptwt + PyWavelets + pytorch-wavelets)",
)
def test_wavelet_conv3d_reference_parity():
    ref_mod = _load_reference_wavelet_module()
    in_channels, out_channels = 2, 2
    level = 1
    size = [8, 8, 8]
    wavelet = "db4"
    mode = "periodic"

    ref_layer = ref_mod.WaveConv3d(in_channels, out_channels, level, size, wavelet, mode=mode)
    n_modes = tuple(ref_layer.weights1.shape[-3:])
    test_layer = WaveletConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        wavelet=wavelet,
        wavelet_levels=level,
        wavelet_mode=mode,
        base_resolution=tuple(size),
        bias=False,
    )
    _copy_weights_3d(ref_layer, test_layer)

    x = torch.randn(1, in_channels, *size)
    y_ref = _reference_waveconv3d_forward_compat(ref_mod, ref_layer, x)
    y_test = test_layer(x)
    torch.testing.assert_close(y_test, y_ref, rtol=1e-4, atol=1e-4)
