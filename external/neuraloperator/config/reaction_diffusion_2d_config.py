from typing import Any, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, WNO_Medium2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class ReactionDiffusion2DOptConfig(OptimizationConfig):
    n_epochs: int = 20
    learning_rate: float = 2e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-5
    scheduler: str = "StepLR"
    step_size: int = 10
    gamma: float = 0.5


class ReactionDiffusion2DDatasetConfig(ConfigBase):
    n_train: int = 256
    n_test: int = 64
    batch_size: int = 16
    test_batch_size: int = 16
    spatial_size: int = 32
    n_steps_input: int = 4
    rollout_steps: int = 4
    n_rollout_eval: int = 8
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
    model: ModelConfig = WNO_Medium2d(
        data_channels=4,
        out_channels=1,
        n_modes=[16, 16],
        hidden_channels=32,
        blocks=4,
        wavelet="db4",
        wavelet_levels=2,
        wavelet_mode="symmetric",
    )
    opt: OptimizationConfig = ReactionDiffusion2DOptConfig()
    data: ReactionDiffusion2DDatasetConfig = ReactionDiffusion2DDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
