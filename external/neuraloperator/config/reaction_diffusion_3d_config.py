from typing import Any, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, WNO_Medium3d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class ReactionDiffusion3DOptConfig(OptimizationConfig):
    n_epochs: int = 20
    learning_rate: float = 2e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-5
    scheduler: str = "StepLR"
    step_size: int = 10
    gamma: float = 0.5


class ReactionDiffusion3DDatasetConfig(ConfigBase):
    n_train: int = 128
    n_test: int = 32
    batch_size: int = 8
    test_batch_size: int = 8
    spatial_size: int = 16
    n_steps_input: int = 8
    block_size: int = 8
    n_rollout_blocks: int = 1
    n_rollout_eval: int = 4
    dt: float = 0.05
    diffusion: float = 0.15
    reaction: float = 0.4
    train_seed: int = 0
    test_seed: int = 123
    rollout_seed: int = 456


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = WNO_Medium3d(
        data_channels=8,
        out_channels=1,
        n_modes=[8, 8, 8],
        hidden_channels=24,
        blocks=4,
        wavelet="db4",
        wavelet_levels=1,
        wavelet_mode="periodic",
    )
    opt: OptimizationConfig = ReactionDiffusion3DOptConfig()
    data: ReactionDiffusion3DDatasetConfig = ReactionDiffusion3DDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
