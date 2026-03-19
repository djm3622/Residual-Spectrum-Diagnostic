"""Convolutional Navier-Stokes surrogate model."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.losses import ObjectiveLoss
from models.vision_baselines import (
    SingleChannelFieldWrapper,
    build_dense_field_model,
    resolve_baseline_config,
)
from utils.progress import progress_range
from utils.torch_runtime import (
    build_adam_optimizer,
    build_grad_scaler,
    clone_state_dict,
    configure_torch_backend,
    move_optimizer_state_to_device,
    resolve_torch_device,
    train_autocast,
)

from ..helpers.sanitization import sanitize_field
from ..layers import NSNonlinearOneStepNet


class ConvolutionalSurrogate2D:
    """Nonlinear one-step periodic residual integrator for Navier-Stokes."""

    def __init__(
        self,
        nx: int,
        ny: int,
        seed: int | None = None,
        device: str = "auto",
        loss: str = "combined",
        architecture: str = "legacy_conv",
        baseline_config: Mapping[str, Any] | None = None,
        model_width: int = 64,
        model_depth: int = 5,
    ):
        self.nx = nx
        self.ny = ny
        self.architecture = str(architecture).strip().lower().replace("-", "_")
        self.device = resolve_torch_device(device)
        configure_torch_backend(self.device)
        self.grad_scaler = build_grad_scaler(self.device)
        self.objective_loss = ObjectiveLoss(nx=nx, ny=ny, device=self.device, loss=loss)
        self.loss_name = self.objective_loss.loss_name

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        resolved_cfg = resolve_baseline_config(self.architecture, baseline_config)
        if self.architecture in {"legacy_conv", "conv"}:
            width = max(8, int(resolved_cfg.get("model_width", model_width)))
            depth = max(1, int(resolved_cfg.get("model_depth", model_depth)))
            net: torch.nn.Module = NSNonlinearOneStepNet(width=width, depth=depth)
        elif self.architecture in {"swin", "swin_transformer", "swin_t", "attn_unet", "attention_unet", "unet_attn"}:
            dense_model = build_dense_field_model(
                architecture=self.architecture,
                in_channels=1,
                out_channels=1,
                config=resolved_cfg,
            )
            net = SingleChannelFieldWrapper(dense_model)
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}' for ConvolutionalSurrogate2D. "
                "Use one of: legacy_conv, swin, attn_unet."
            )

        self.net = net.to(self.device)
        self.net.eval()

    def forward(self, omega: np.ndarray) -> np.ndarray:
        inp = np.asarray(omega, dtype=np.float32)[np.newaxis, ...]
        x = torch.from_numpy(inp).to(self.device)
        with torch.inference_mode():
            pred = self.net(x)[0].cpu().numpy()
        return sanitize_field(pred)

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        lr: float = 0.001,
        n_iter: int = 100,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,
        use_one_cycle_lr: bool = False,
        one_cycle_pct_start: float = 0.3,
        one_cycle_div_factor: float = 25.0,
        one_cycle_final_div_factor: float = 10000.0,
        trajectory: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
        val_inputs: List[np.ndarray] | None = None,
        val_targets: List[np.ndarray] | None = None,
        checkpoint_callback: Callable[[int, float, Dict[str, Any]], None] | None = None,
        early_stopping_patience: int | None = None,
        resume_state: Dict[str, Any] | None = None,
        train_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
        val_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
        dataloader_num_workers: int = 0,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        return _train_convolutional_surrogate_2d(**locals())

    def state_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        for key, tensor in self.net.state_dict().items():
            payload[key] = tensor.detach().cpu().numpy()
        payload["_architecture"] = np.asarray(self.architecture)
        return payload

def _train_convolutional_surrogate_2d(
    self,
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    lr: float = 0.001,
    n_iter: int = 100,
    batch_size: int = 32,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    use_one_cycle_lr: bool = False,
    one_cycle_pct_start: float = 0.3,
    one_cycle_div_factor: float = 25.0,
    one_cycle_final_div_factor: float = 10000.0,
    trajectory: List[np.ndarray] | None = None,
    rollout_horizon: int = 1,
    rollout_weight: float = 0.0,
    val_inputs: List[np.ndarray] | None = None,
    val_targets: List[np.ndarray] | None = None,
    checkpoint_callback: Callable[[int, float, Dict[str, Any]], None] | None = None,
    early_stopping_patience: int | None = None,
    resume_state: Dict[str, Any] | None = None,
    train_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    val_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    dataloader_num_workers: int = 0,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    if train_dataset is None:
        if not inputs:
            raise ValueError("Training inputs are empty.")
        train_x = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
        train_y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        train_dataset = TensorDataset(train_x, train_y)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if val_dataset is None:
        has_val_inputs = (
            val_inputs is not None
            and val_targets is not None
            and len(val_inputs) > 0
            and len(val_inputs) == len(val_targets)
        )
        if has_val_inputs:
            val_x = torch.from_numpy(np.asarray(val_inputs, dtype=np.float32))
            val_y = torch.from_numpy(np.asarray(val_targets, dtype=np.float32))
            val_dataset = TensorDataset(val_x, val_y)

    n_samples = len(train_dataset)
    batch = max(1, min(int(batch_size), n_samples))
    num_workers = max(0, int(dataloader_num_workers))
    pin_memory = str(self.device).startswith("cuda")
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    has_val = val_dataset is not None and len(val_dataset) > 0
    if has_val and val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    else:
        val_loader = None

    clip = max(float(grad_clip), 1e-8)
    wd = max(0.0, float(weight_decay))
    one_cycle_enabled = bool(use_one_cycle_lr)
    pct_start = float(np.clip(one_cycle_pct_start, 1e-4, 0.9999))
    div_factor = max(float(one_cycle_div_factor), 1.0)
    final_div_factor = max(float(one_cycle_final_div_factor), 1.0)
    horizon = max(1, int(rollout_horizon))
    rollout_w = max(0.0, float(rollout_weight))
    patience_raw = int(early_stopping_patience) if early_stopping_patience is not None else 0
    patience = max(1, patience_raw) if patience_raw > 0 else None
    monitor_validation = has_val and (
        show_progress
        or checkpoint_callback is not None
        or patience is not None
    )
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state: Dict[str, Any] | None = None

    use_rollout = rollout_w > 0.0 and horizon > 1 and trajectory is not None and len(trajectory) > 0
    if use_rollout:
        seq = torch.from_numpy(np.asarray(trajectory, dtype=np.float32)).to(self.device)
        n_traj, n_steps, _, _ = seq.shape
        max_start = n_steps - horizon - 1
        if max_start < 0:
            use_rollout = False
        else:
            rollout_batch = max(1, min(batch // 2, n_traj))
    else:
        seq = None
        rollout_batch = 0
        max_start = -1

    optimizer = build_adam_optimizer(
        self.net.parameters(),
        lr=float(lr),
        device=self.device,
        weight_decay=wd,
    )
    total_iter = max(1, int(n_iter))
    steps_per_epoch = max(1, int(np.ceil(n_samples / batch)))
    total_steps = max(1, total_iter * steps_per_epoch)
    scheduler = None
    if one_cycle_enabled:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(lr),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
    start_epoch = 1
    if isinstance(resume_state, dict):
        model_state = resume_state.get("model_state")
        if isinstance(model_state, dict):
            self.net.load_state_dict(model_state)
        optimizer_state = resume_state.get("optimizer_state")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(optimizer, self.device)
        scheduler_state = resume_state.get("scheduler_state")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        grad_scaler_state = resume_state.get("grad_scaler_state")
        if self.grad_scaler is not None and isinstance(grad_scaler_state, dict):
            self.grad_scaler.load_state_dict(grad_scaler_state)
        rng_state = resume_state.get("rng_state")
        if isinstance(rng_state, dict):
            python_state = rng_state.get("python")
            if python_state is not None:
                try:
                    random.setstate(python_state)
                except Exception:
                    pass
            numpy_state = rng_state.get("numpy")
            if numpy_state is not None:
                try:
                    np.random.set_state(numpy_state)
                except Exception:
                    pass
            torch_state = rng_state.get("torch")
            if isinstance(torch_state, torch.Tensor):
                try:
                    torch.set_rng_state(torch_state.cpu())
                except Exception:
                    pass
            torch_cuda_state = rng_state.get("torch_cuda")
            if torch.cuda.is_available() and isinstance(torch_cuda_state, list):
                try:
                    torch.cuda.set_rng_state_all(
                        [
                            state.cpu() if isinstance(state, torch.Tensor) else state
                            for state in torch_cuda_state
                        ]
                    )
                except Exception:
                    pass
        best_val_loss = float(resume_state.get("best_val_loss", best_val_loss))
        epochs_without_improvement = int(
            max(0, int(resume_state.get("epochs_without_improvement", epochs_without_improvement)))
        )
        best_model_state = resume_state.get("best_model_state")
        if isinstance(best_model_state, dict):
            best_state = clone_state_dict(best_model_state)
        elif np.isfinite(best_val_loss):
            best_state = clone_state_dict(self.net.state_dict())
        start_epoch = max(1, int(resume_state.get("epoch", 0)) + 1)

    def _capture_training_state(epoch_idx: int, val_loss_value: float) -> Dict[str, Any]:
        cuda_rng_state = None
        if torch.cuda.is_available():
            try:
                cuda_rng_state = [state.cpu() for state in torch.cuda.get_rng_state_all()]
            except Exception:
                cuda_rng_state = None
        return {
            "epoch": int(epoch_idx),
            "val_loss": float(val_loss_value) if np.isfinite(val_loss_value) else float("nan"),
            "model_state": clone_state_dict(self.net.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().cpu(),
                "torch_cuda": cuda_rng_state,
            },
            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else float("inf"),
            "epochs_without_improvement": int(epochs_without_improvement),
            "best_model_state": clone_state_dict(best_state) if best_state is not None else None,
            "loss_name": str(self.loss_name),
        }

    def _compute_validation_loss() -> float:
        if not has_val or val_loader is None:
            return float("nan")

        was_training = self.net.training
        self.net.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for xb_val_cpu, yb_val_cpu in val_loader:
                xb_val = xb_val_cpu.to(self.device, non_blocking=True)
                yb_val = yb_val_cpu.to(self.device, non_blocking=True)
                with train_autocast(self.device):
                    pred_val = self.net(xb_val)
                    loss_val = self.objective_loss(pred_val, yb_val)
                if torch.isfinite(loss_val):
                    bs = int(xb_val.shape[0])
                    total += float(loss_val.item()) * bs
                    count += bs
        if was_training:
            self.net.train()

        if count == 0:
            return float("nan")
        return total / float(count)

    self.net.train()
    iter_desc = progress_desc or "Training iterations"
    if start_epoch > total_iter:
        self.net.eval()
        return
    remaining_iter = total_iter - start_epoch + 1
    iter_progress = progress_range(remaining_iter, enabled=show_progress, desc=iter_desc)
    for iter_offset, _ in enumerate(iter_progress):
        epoch_idx = start_epoch + iter_offset
        train_loss_sum = 0.0
        train_loss_count = 0
        for xb_cpu, yb_cpu in train_loader:
            xb = xb_cpu.to(self.device, non_blocking=True)
            yb = yb_cpu.to(self.device, non_blocking=True)

            with train_autocast(self.device):
                pred = self.net(xb)
                one_step_loss = self.objective_loss(pred, yb)

                rollout_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                if use_rollout and seq is not None and rollout_batch > 0:
                    traj_idx = torch.randint(0, seq.shape[0], (rollout_batch,), device=self.device)
                    start_idx = torch.randint(0, max_start + 1, (rollout_batch,), device=self.device)

                    state_roll = seq[traj_idx, start_idx]
                    for offset in range(1, horizon + 1):
                        pred_roll = self.net(state_roll)
                        target_roll = seq[traj_idx, start_idx + offset]
                        rollout_loss = rollout_loss + self.objective_loss(pred_roll, target_roll)
                        state_roll = pred_roll
                    rollout_loss = rollout_loss / float(horizon)

                loss = one_step_loss + rollout_w * rollout_loss

            if not torch.isfinite(loss):
                continue

            train_loss_sum += float(loss.detach().item())
            train_loss_count += 1

            optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=clip)
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        val_loss_value = _compute_validation_loss() if monitor_validation else float("nan")
        if show_progress and hasattr(iter_progress, "set_postfix"):
            train_loss_value = (
                train_loss_sum / float(train_loss_count) if train_loss_count > 0 else float("nan")
            )
            lr_value = float(optimizer.param_groups[0]["lr"])
            postfix = {
                "train_loss": f"{train_loss_value:.3e}" if np.isfinite(train_loss_value) else "nan",
                "val_loss": f"{val_loss_value:.3e}" if np.isfinite(val_loss_value) else "n/a",
                "lr": f"{lr_value:.2e}",
            }
            iter_progress.set_postfix(postfix, refresh=False)
        if patience is not None and has_val:
            if np.isfinite(val_loss_value) and float(val_loss_value) < (best_val_loss - 1e-12):
                best_val_loss = float(val_loss_value)
                best_state = clone_state_dict(self.net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        training_state = _capture_training_state(epoch_idx, float(val_loss_value))
        if checkpoint_callback is not None:
            checkpoint_callback(epoch_idx, float(val_loss_value), training_state)
        if patience is not None and has_val and epochs_without_improvement >= patience:
            break
    if best_state is not None:
        self.net.load_state_dict(best_state)
    self.net.eval()

