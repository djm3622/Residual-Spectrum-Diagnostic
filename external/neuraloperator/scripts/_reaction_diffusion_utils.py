from typing import Tuple

import torch

from neuralop.data.datasets.tensor_dataset import TensorDataset


def laplacian_2d(u: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(u, shifts=1, dims=-1)
        + torch.roll(u, shifts=-1, dims=-1)
        + torch.roll(u, shifts=1, dims=-2)
        + torch.roll(u, shifts=-1, dims=-2)
        - 4.0 * u
    )


def reaction_diffusion_step(
    u: torch.Tensor, dt: float, diffusion: float, reaction: float
) -> torch.Tensor:
    return u + dt * (diffusion * laplacian_2d(u) + reaction * u * (1.0 - u))


def generate_rd_sequence(
    batch_size: int,
    n_steps: int,
    height: int,
    width: int,
    *,
    dt: float,
    diffusion: float,
    reaction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    u = torch.rand(batch_size, height, width, generator=generator)
    seq = [u]
    for _ in range(n_steps - 1):
        u = reaction_diffusion_step(seq[-1], dt=dt, diffusion=diffusion, reaction=reaction)
        seq.append(u)
    return torch.stack(seq, dim=1)


def make_rd_2d_next_step_dataset(
    n_samples: int,
    n_steps_input: int,
    spatial_size: int,
    *,
    dt: float,
    diffusion: float,
    reaction: float,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_samples, n_steps_input, spatial_size, spatial_size)
    y = torch.zeros(n_samples, 1, spatial_size, spatial_size)
    for i in range(n_samples):
        seq = generate_rd_sequence(
            1,
            n_steps_input + 1,
            spatial_size,
            spatial_size,
            dt=dt,
            diffusion=diffusion,
            reaction=reaction,
            generator=generator,
        )[0]
        x[i] = seq[:n_steps_input]
        y[i, 0] = seq[n_steps_input]
    return TensorDataset(x=x, y=y)


def make_rd_3d_next_block_dataset(
    n_samples: int,
    n_steps_input: int,
    block_size: int,
    spatial_size: int,
    *,
    dt: float,
    diffusion: float,
    reaction: float,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_samples, n_steps_input, block_size, spatial_size, spatial_size)
    y = torch.zeros(n_samples, 1, block_size, spatial_size, spatial_size)
    for i in range(n_samples):
        seq = generate_rd_sequence(
            1,
            n_steps_input + block_size,
            spatial_size,
            spatial_size,
            dt=dt,
            diffusion=diffusion,
            reaction=reaction,
            generator=generator,
        )[0]
        history = seq[:n_steps_input]
        target_block = seq[n_steps_input : n_steps_input + block_size]

        # Encode temporal history as channels and tile across the depth block.
        x[i] = history.unsqueeze(1).repeat(1, block_size, 1, 1)
        y[i, 0] = target_block
    return TensorDataset(x=x, y=y)


@torch.no_grad()
def rollout_autoregressive_2d(
    model,
    history: torch.Tensor,
    rollout_steps: int,
) -> torch.Tensor:
    preds = []
    state = history
    for _ in range(rollout_steps):
        pred = model(state).squeeze(1)
        preds.append(pred)
        state = torch.cat([state[:, 1:], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


@torch.no_grad()
def rollout_next_block_3d(
    model,
    history: torch.Tensor,
    *,
    block_size: int,
    n_blocks: int,
) -> torch.Tensor:
    preds = []
    state = history
    for _ in range(n_blocks):
        block_input = state.unsqueeze(2).repeat(1, 1, block_size, 1, 1)
        pred_block = model(block_input).squeeze(1)
        preds.append(pred_block)
        state = torch.cat([state[:, block_size:], pred_block], dim=1)
    return torch.cat(preds, dim=1)


def rollout_targets_2d(
    n_rollouts: int,
    n_steps_input: int,
    rollout_steps: int,
    spatial_size: int,
    *,
    dt: float,
    diffusion: float,
    reaction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    seq = generate_rd_sequence(
        n_rollouts,
        n_steps_input + rollout_steps,
        spatial_size,
        spatial_size,
        dt=dt,
        diffusion=diffusion,
        reaction=reaction,
        generator=generator,
    )
    return seq[:, :n_steps_input], seq[:, n_steps_input:]


def rollout_targets_3d(
    n_rollouts: int,
    n_steps_input: int,
    block_size: int,
    n_blocks: int,
    spatial_size: int,
    *,
    dt: float,
    diffusion: float,
    reaction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    horizon = block_size * n_blocks
    seq = generate_rd_sequence(
        n_rollouts,
        n_steps_input + horizon,
        spatial_size,
        spatial_size,
        dt=dt,
        diffusion=diffusion,
        reaction=reaction,
        generator=generator,
    )
    return seq[:, :n_steps_input], seq[:, n_steps_input:]
