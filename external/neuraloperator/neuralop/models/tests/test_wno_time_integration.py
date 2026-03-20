import math

import pytest
import torch
import torch.nn.functional as F

from neuralop.models import WNO


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


def _laplacian_2d(u):
    return (
        torch.roll(u, shifts=1, dims=-1)
        + torch.roll(u, shifts=-1, dims=-1)
        + torch.roll(u, shifts=1, dims=-2)
        + torch.roll(u, shifts=-1, dims=-2)
        - 4.0 * u
    )


def _reaction_diffusion_step(u, dt=0.05, diffusion=0.15, reaction=0.4):
    # Smooth logistic reaction-diffusion dynamics on periodic grids.
    return u + dt * (diffusion * _laplacian_2d(u) + reaction * u * (1.0 - u))


def _generate_rd_sequence(batch_size, n_steps, height, width, device):
    u = torch.rand(batch_size, height, width, device=device)
    seq = [u]
    for _ in range(n_steps - 1):
        u = _reaction_diffusion_step(seq[-1])
        seq.append(u)
    return torch.stack(seq, dim=1)


def _rollout_next_step_2d(model, history, horizon):
    preds = []
    state = history
    for _ in range(horizon):
        pred = model(state).squeeze(1)
        preds.append(pred)
        state = torch.cat([state[:, 1:], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def _history_to_3d_block_input(history, block_size):
    # history: [B, T_in, H, W] -> [B, T_in, T_block, H, W]
    return history.unsqueeze(2).repeat(1, 1, block_size, 1, 1)


def _rollout_next_block_3d(model, history, n_blocks, block_size):
    preds = []
    state = history
    for _ in range(n_blocks):
        block_in = _history_to_3d_block_input(state, block_size)
        pred_block = model(block_in).squeeze(1)
        preds.append(pred_block)
        state = torch.cat([state[:, block_size:], pred_block], dim=1)
    return torch.cat(preds, dim=1)


@pytest.mark.skipif(not HAS_PYTORCH_WAVELETS, reason="pytorch-wavelets not installed")
def test_wno_2d_reaction_diffusion_time_integration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = 2
    t_in = 4
    horizon = 3
    height = 16
    width = 16

    seq = _generate_rd_sequence(batch, t_in + horizon, height, width, device)
    history = seq[:, :t_in]
    target = seq[:, t_in : t_in + horizon]

    model = WNO(
        n_modes=(8, 8),
        in_channels=t_in,
        out_channels=1,
        hidden_channels=16,
        n_layers=2,
        wavelet_levels=2,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    pred = _rollout_next_step_2d(model, history, horizon)
    loss_before = F.mse_loss(pred, target)

    opt.zero_grad()
    loss_before.backward()
    opt.step()

    pred_after = _rollout_next_step_2d(model, history, horizon)
    loss_after = F.mse_loss(pred_after, target)

    assert pred.shape == target.shape
    assert math.isfinite(loss_before.item())
    assert math.isfinite(loss_after.item())


@pytest.mark.skipif(
    not (HAS_PYTORCH_WAVELETS and HAS_PTWT_3D),
    reason="3D wavelet deps not installed (ptwt + PyWavelets + pytorch-wavelets)",
)
def test_wno_3d_reaction_diffusion_next_block_time_integration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = 1
    t_in = 8
    block_size = 8
    n_blocks = 1
    horizon = block_size * n_blocks
    height = 16
    width = 16

    seq = _generate_rd_sequence(batch, t_in + horizon, height, width, device)
    history = seq[:, :t_in]
    target = seq[:, t_in : t_in + horizon]

    model = WNO(
        n_modes=(block_size, 4, 4),
        in_channels=t_in,
        out_channels=1,
        hidden_channels=12,
        n_layers=2,
        wavelet_levels=1,
        wavelet_mode="periodic",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    pred = _rollout_next_block_3d(model, history, n_blocks=n_blocks, block_size=block_size)
    loss_before = F.mse_loss(pred, target)

    opt.zero_grad()
    loss_before.backward()
    opt.step()

    pred_after = _rollout_next_block_3d(model, history, n_blocks=n_blocks, block_size=block_size)
    loss_after = F.mse_loss(pred_after, target)

    assert pred.shape == target.shape
    assert math.isfinite(loss_before.item())
    assert math.isfinite(loss_after.item())
