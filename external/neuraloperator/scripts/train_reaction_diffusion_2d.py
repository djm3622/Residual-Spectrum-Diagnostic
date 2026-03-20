"""
Train WNO on a synthetic 2D reaction-diffusion next-step task and evaluate rollout integration.
"""

import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from neuralop import LpLoss, Trainer, get_model
from neuralop.training import setup, AdamW
from neuralop.utils import get_wandb_api_key, count_model_params

from _reaction_diffusion_utils import (
    make_rd_2d_next_step_dataset,
    rollout_autoregressive_2d,
    rollout_targets_2d,
)

# Configuration setup
from zencfg import make_config_from_cli

sys.path.insert(0, "../")
from config.reaction_diffusion_2d_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

device, is_logger = setup(config)

# WandB logging configuration
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    wandb_name = config.wandb.name or "wno_rd_2d"
    wandb.init(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )

config.verbose = config.verbose and is_logger
if config.verbose:
    print(f"##### CONFIG #####\n{config}")
    sys.stdout.flush()

train_db = make_rd_2d_next_step_dataset(
    n_samples=config.data.n_train,
    n_steps_input=config.data.n_steps_input,
    spatial_size=config.data.spatial_size,
    dt=config.data.dt,
    diffusion=config.data.diffusion,
    reaction=config.data.reaction,
    seed=config.data.train_seed,
)
test_db = make_rd_2d_next_step_dataset(
    n_samples=config.data.n_test,
    n_steps_input=config.data.n_steps_input,
    spatial_size=config.data.spatial_size,
    dt=config.data.dt,
    diffusion=config.data.diffusion,
    reaction=config.data.reaction,
    seed=config.data.test_seed,
)

train_loader = DataLoader(train_db, batch_size=config.data.batch_size, shuffle=True)
test_loaders = {
    "rd2d": DataLoader(test_db, batch_size=config.data.test_batch_size, shuffle=False)
}

model = get_model(config).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
elif config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
else:
    raise ValueError(f"Unsupported scheduler: {config.opt.scheduler}")

l2loss = LpLoss(d=2, p=2)
train_loss = l2loss
eval_losses = {"l2": l2loss}

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    wandb_log=config.wandb.log,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
)

if is_logger and config.verbose:
    print(f"n_params: {count_model_params(model)}")
    print("Starting 2D reaction-diffusion training...")

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# Time-integration evaluation (autoregressive rollout)
model.eval()
history, target = rollout_targets_2d(
    n_rollouts=config.data.n_rollout_eval,
    n_steps_input=config.data.n_steps_input,
    rollout_steps=config.data.rollout_steps,
    spatial_size=config.data.spatial_size,
    dt=config.data.dt,
    diffusion=config.data.diffusion,
    reaction=config.data.reaction,
    seed=config.data.rollout_seed,
)
history = history.to(device)
target = target.to(device)
pred = rollout_autoregressive_2d(model, history, config.data.rollout_steps)
rollout_mse = F.mse_loss(pred, target).item()

if is_logger:
    print(f"2D rollout MSE ({config.data.rollout_steps} steps): {rollout_mse:.6e}")
    if config.wandb.log:
        wandb.log({"rd2d_rollout_mse": rollout_mse})
        wandb.finish()
