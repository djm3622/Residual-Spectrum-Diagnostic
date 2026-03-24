"""Convolutional coupled reaction-diffusion surrogate model."""

from __future__ import annotations

import random
import inspect
from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from models.losses import ObjectiveLoss
from models.vision_baselines import build_dense_field_model, resolve_baseline_config
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

from ..helpers.sanitization import sanitize_species
from ..layers import RDUNetOneStepNet


class ConvolutionalSurrogate2DCoupled:
    """Nonlinear one-step periodic residual integrator for Gray-Scott dynamics."""

    def __init__(
        self,
        nx: int,
        ny: int,
        seed: int | None = None,
        device: str = "auto",
        loss: str = "combined",
        architecture: str = "legacy_conv",
        baseline_config: Mapping[str, Any] | None = None,
        temporal_config: Mapping[str, Any] | None = None,
    ):
        self.nx = nx
        self.ny = ny
        self.architecture = str(architecture).strip().lower().replace("-", "_")
        self.device = resolve_torch_device(device)
        configure_torch_backend(self.device)
        self.grad_scaler = build_grad_scaler(self.device)
        self.objective_loss = ObjectiveLoss(nx=nx, ny=ny, device=self.device, loss=loss)
        self.loss_name = self.objective_loss.loss_name
        temporal_cfg = dict(temporal_config or {})
        self.temporal_enabled = bool(temporal_cfg.get("enabled", False))
        self.temporal_window = max(2, int(temporal_cfg.get("input_steps", 20))) if self.temporal_enabled else 1
        if self.temporal_enabled:
            output_steps = max(2, int(temporal_cfg.get("output_steps", self.temporal_window)))
            if output_steps != self.temporal_window:
                raise ValueError(
                    "Temporal RD baseline path requires output_steps == input_steps "
                    f"(got input_steps={self.temporal_window}, output_steps={output_steps})."
                )
        self.temporal_target_mode = str(temporal_cfg.get("target_mode", "next_block")).strip().lower().replace("-", "_")
        if self.temporal_target_mode not in {"shifted", "next_block"}:
            self.temporal_target_mode = "next_block"

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        resolved_cfg = resolve_baseline_config(self.architecture, baseline_config)
        if self.architecture in {"legacy_conv", "conv"}:
            width = max(8, int(resolved_cfg.get("width", 48)))
            self.net = RDUNetOneStepNet(width=width).to(self.device)
            if self.temporal_enabled:
                self.temporal_enabled = False
        elif self.architecture in {"swin", "swin_transformer", "swin_t", "attn_unet", "attention_unet", "unet_attn"}:
            in_channels = 2 * self.temporal_window if self.temporal_enabled else 2
            self.net = build_dense_field_model(
                architecture=self.architecture,
                in_channels=in_channels,
                out_channels=in_channels,
                config=resolved_cfg,
            ).to(self.device)
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}' for ConvolutionalSurrogate2DCoupled. "
                "Use one of: legacy_conv, swin, attn_unet."
            )
        self.net.eval()

    def _prepare_window(self, field_window: np.ndarray) -> np.ndarray:
        arr = np.asarray(field_window, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected window with shape [T,H,W], got {arr.shape}.")
        if arr.shape[1:] != (self.nx, self.ny):
            raise ValueError(
                f"Window spatial shape mismatch: expected {(self.nx, self.ny)}, got {arr.shape[1:]}."
            )
        if arr.shape[0] < self.temporal_window:
            pad = np.repeat(arr[-1][np.newaxis, ...], self.temporal_window - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > self.temporal_window:
            arr = arr[-self.temporal_window :]
        return np.asarray(arr, dtype=np.float32)

    def _stack_windows(self, u_window: np.ndarray, v_window: np.ndarray) -> np.ndarray:
        # Simple block prediction path for baseline models: append time along channel dim.
        uw = self._prepare_window(u_window)
        vw = self._prepare_window(v_window)
        return np.concatenate([uw, vw], axis=0).astype(np.float32, copy=False)

    def _split_pred_window(self, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pred.ndim != 3:
            raise ValueError(f"Expected prediction tensor [C,H,W], got {pred.shape}.")
        channels = pred.shape[0]
        expected = 2 * self.temporal_window
        if channels != expected:
            raise ValueError(f"Expected {expected} output channels for temporal block, got {channels}.")
        pred_u = np.asarray([sanitize_species(frame) for frame in pred[: self.temporal_window]], dtype=np.float32)
        pred_v = np.asarray([sanitize_species(frame) for frame in pred[self.temporal_window :]], dtype=np.float32)
        return pred_u, pred_v

    def predict_window(self, u_window: np.ndarray, v_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.temporal_enabled:
            raise ValueError("predict_window is only available when temporal mode is enabled.")
        stacked = self._stack_windows(u_window, v_window)[np.newaxis, ...]
        xb = torch.from_numpy(stacked).to(self.device)
        with torch.inference_mode():
            pred = self.net(xb)[0].detach().cpu().numpy()
        return self._split_pred_window(pred)

    def rollout(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        n_steps: int,
        context_u: np.ndarray | None = None,
        context_v: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        total_steps = max(1, int(n_steps))
        u0_f32 = sanitize_species(np.asarray(u0, dtype=np.float32))
        v0_f32 = sanitize_species(np.asarray(v0, dtype=np.float32))
        u_traj = np.zeros((total_steps, self.nx, self.ny), dtype=np.float32)
        v_traj = np.zeros((total_steps, self.nx, self.ny), dtype=np.float32)
        u_traj[0] = u0_f32
        v_traj[0] = v0_f32
        if total_steps == 1:
            return u_traj, v_traj
        if not self.temporal_enabled:
            u_cur = u0_f32
            v_cur = v0_f32
            for step in range(1, total_steps):
                u_cur, v_cur = self.forward(u_cur, v_cur)
                u_traj[step] = u_cur
                v_traj[step] = v_cur
            return u_traj, v_traj

        if (
            context_u is not None
            and context_v is not None
            and np.asarray(context_u).ndim == 3
            and np.asarray(context_v).ndim == 3
            and np.asarray(context_u).shape[0] > 0
            and np.asarray(context_v).shape[0] > 0
        ):
            u_window = self._prepare_window(np.asarray(context_u, dtype=np.float32)[: self.temporal_window])
            v_window = self._prepare_window(np.asarray(context_v, dtype=np.float32)[: self.temporal_window])
        else:
            u_window = np.repeat(u0_f32[np.newaxis, ...], self.temporal_window, axis=0)
            v_window = np.repeat(v0_f32[np.newaxis, ...], self.temporal_window, axis=0)

        known_steps = min(total_steps, self.temporal_window)
        u_traj[:known_steps] = u_window[:known_steps]
        v_traj[:known_steps] = v_window[:known_steps]
        cursor = known_steps

        while cursor < total_steps:
            pred_u_window, pred_v_window = self.predict_window(u_window, v_window)
            if self.temporal_target_mode == "next_block":
                take = min(pred_u_window.shape[0], total_steps - cursor)
                u_traj[cursor : cursor + take] = pred_u_window[:take]
                v_traj[cursor : cursor + take] = pred_v_window[:take]
                cursor += take
                u_window = pred_u_window
                v_window = pred_v_window
            else:
                next_u = sanitize_species(pred_u_window[-1])
                next_v = sanitize_species(pred_v_window[-1])
                u_traj[cursor] = next_u
                v_traj[cursor] = next_v
                cursor += 1
                u_window = np.concatenate([u_window[1:], next_u[np.newaxis, ...]], axis=0)
                v_window = np.concatenate([v_window[1:], next_v[np.newaxis, ...]], axis=0)
        return u_traj, v_traj

    def forward(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_arr = np.asarray(u, dtype=np.float32)
        v_arr = np.asarray(v, dtype=np.float32)
        if self.temporal_enabled:
            if u_arr.ndim == 2 and v_arr.ndim == 2:
                u_window = np.repeat(u_arr[np.newaxis, ...], self.temporal_window, axis=0)
                v_window = np.repeat(v_arr[np.newaxis, ...], self.temporal_window, axis=0)
            else:
                u_window = self._prepare_window(u_arr)
                v_window = self._prepare_window(v_arr)
            pred_u_window, pred_v_window = self.predict_window(u_window, v_window)
            if self.temporal_target_mode == "next_block":
                return sanitize_species(pred_u_window[0]), sanitize_species(pred_v_window[0])
            return sanitize_species(pred_u_window[-1]), sanitize_species(pred_v_window[-1])
        x = np.stack([u_arr, v_arr], axis=0).astype(np.float32, copy=False)[np.newaxis, ...]
        xb = torch.from_numpy(x).to(self.device)
        with torch.inference_mode():
            pred = self.net(xb)[0].cpu().numpy()
        u_next = sanitize_species(pred[0])
        v_next = sanitize_species(pred[1])
        return u_next, v_next

    def _compute_channel_rms(self, stats_loader: DataLoader) -> torch.Tensor:
        sum_sq: torch.Tensor | None = None
        count = 0
        for _, yb_cpu in stats_loader:
            yb = yb_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
            reduce_dims = tuple(idx for idx in range(yb.ndim) if idx != 1)
            batch_sum_sq = torch.sum(yb * yb, dim=reduce_dims, keepdim=True)
            if sum_sq is None:
                sum_sq = batch_sum_sq
            else:
                sum_sq = sum_sq + batch_sum_sq
            count += int(yb.numel() // max(1, int(yb.shape[1])))
        if sum_sq is None or count <= 0:
            raise ValueError("Training dataset is empty.")
        channel_rms = torch.sqrt(torch.clamp(sum_sq / float(count), min=1e-12))
        return torch.clamp(channel_rms, min=1e-6)

    def train(
        self,
        inputs_u: List[np.ndarray],
        inputs_v: List[np.ndarray],
        targets_u: List[np.ndarray],
        targets_v: List[np.ndarray],
        lr: float = 0.001,
        n_iter: int = 100,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,
        use_one_cycle_lr: bool = False,
        one_cycle_pct_start: float = 0.3,
        one_cycle_div_factor: float = 25.0,
        one_cycle_final_div_factor: float = 10000.0,
        trajectory_u: List[np.ndarray] | None = None,
        trajectory_v: List[np.ndarray] | None = None,
        rollout_horizon: int = 1,
        rollout_weight: float = 0.0,
        val_inputs_u: List[np.ndarray] | None = None,
        val_inputs_v: List[np.ndarray] | None = None,
        val_targets_u: List[np.ndarray] | None = None,
        val_targets_v: List[np.ndarray] | None = None,
        pair_steps: List[int] | None = None,
        checkpoint_callback: (
            Callable[[int, float, Dict[str, Any]], None]
            | Callable[[int, float], None]
            | None
        ) = None,
        early_stopping_patience: int | None = None,
        resume_state: Dict[str, Any] | None = None,
        train_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
        val_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
        dataloader_num_workers: int = 0,
        u_weight: float = 1.0,
        v_weight: float = 1.0,
        channel_balance_cap: float = 3.0,
        dynamics_weight: float = 0.0,
        early_step_bias: float = 0.0,
        early_step_decay: float = 24.0,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        return _train_convolutional_surrogate_2d_coupled(**locals())

    def state_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        for key, tensor in self.net.state_dict().items():
            payload[key] = tensor.detach().cpu().numpy()
        payload["_architecture"] = np.asarray(self.architecture)
        return payload

def _train_convolutional_surrogate_2d_coupled(
    self,
    inputs_u: List[np.ndarray],
    inputs_v: List[np.ndarray],
    targets_u: List[np.ndarray],
    targets_v: List[np.ndarray],
    lr: float = 0.001,
    n_iter: int = 100,
    batch_size: int = 32,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    use_one_cycle_lr: bool = False,
    one_cycle_pct_start: float = 0.3,
    one_cycle_div_factor: float = 25.0,
    one_cycle_final_div_factor: float = 10000.0,
    trajectory_u: List[np.ndarray] | None = None,
    trajectory_v: List[np.ndarray] | None = None,
    rollout_horizon: int = 1,
    rollout_weight: float = 0.0,
    val_inputs_u: List[np.ndarray] | None = None,
    val_inputs_v: List[np.ndarray] | None = None,
    val_targets_u: List[np.ndarray] | None = None,
    val_targets_v: List[np.ndarray] | None = None,
    pair_steps: List[int] | None = None,
    checkpoint_callback: (
        Callable[[int, float, Dict[str, Any]], None]
        | Callable[[int, float], None]
        | None
    ) = None,
    early_stopping_patience: int | None = None,
    resume_state: Dict[str, Any] | None = None,
    train_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    val_dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    dataloader_num_workers: int = 0,
    u_weight: float = 1.0,
    v_weight: float = 1.0,
    channel_balance_cap: float = 3.0,
    dynamics_weight: float = 0.0,
    early_step_bias: float = 0.0,
    early_step_decay: float = 24.0,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> None:
    if train_dataset is None:
        if not inputs_u:
            raise ValueError("Training inputs are empty.")

        x_u = np.asarray(inputs_u, dtype=np.float32)
        x_v = np.asarray(inputs_v, dtype=np.float32)
        y_u = np.asarray(targets_u, dtype=np.float32)
        y_v = np.asarray(targets_v, dtype=np.float32)
        train_dataset = TensorDataset(
            torch.from_numpy(np.stack([x_u, x_v], axis=1)),
            torch.from_numpy(np.stack([y_u, y_v], axis=1)),
        )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if val_dataset is None:
        has_val_inputs = (
            val_inputs_u is not None
            and val_inputs_v is not None
            and val_targets_u is not None
            and val_targets_v is not None
            and len(val_inputs_u) > 0
            and len(val_inputs_u) == len(val_inputs_v) == len(val_targets_u) == len(val_targets_v)
        )
        if has_val_inputs:
            val_dataset = TensorDataset(
                torch.from_numpy(
                    np.stack(
                        [
                            np.asarray(val_inputs_u, dtype=np.float32),
                            np.asarray(val_inputs_v, dtype=np.float32),
                        ],
                        axis=1,
                    )
                ),
                torch.from_numpy(
                    np.stack(
                        [
                            np.asarray(val_targets_u, dtype=np.float32),
                            np.asarray(val_targets_v, dtype=np.float32),
                        ],
                        axis=1,
                    )
                ),
            )

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

    early_bias = max(0.0, float(early_step_bias))
    early_decay = max(1.0, float(early_step_decay))
    sample_weights_cpu: torch.Tensor | None = None
    if pair_steps is not None and len(pair_steps) == n_samples and early_bias > 0.0:
        step_arr = np.asarray(pair_steps, dtype=np.float32)
        step_weights = 1.0 + early_bias * np.exp(-step_arr / early_decay)
        sample_weights_cpu = torch.from_numpy(step_weights.astype(np.float64, copy=False))
        sample_weights_cpu = sample_weights_cpu / torch.sum(sample_weights_cpu)

    if sample_weights_cpu is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights_cpu,
            num_samples=n_samples,
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    aux_loader_kwargs: Dict[str, Any] = {
        "batch_size": batch,
        "num_workers": 0,
        "pin_memory": pin_memory,
    }
    stats_loader = DataLoader(train_dataset, shuffle=False, **aux_loader_kwargs)
    has_val = val_dataset is not None and len(val_dataset) > 0
    if has_val and val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **aux_loader_kwargs)
    else:
        val_loader = None

    # Keep coupled training balanced without letting small-magnitude channels dominate.
    channel_rms = self._compute_channel_rms(stats_loader)
    balance_cap = max(1.0, float(channel_balance_cap))
    reference_scale = torch.max(channel_rms)
    min_scale = torch.clamp(reference_scale / balance_cap, min=1e-6)
    channel_scale = torch.clamp(channel_rms, min=min_scale)

    channel_weight = torch.tensor(
        [max(float(u_weight), 1e-6), max(float(v_weight), 1e-6)],
        dtype=torch.float32,
        device=self.device,
    )
    channel_weight = channel_weight / torch.mean(channel_weight)
    sample_x, _ = train_dataset[0]
    weight_shape = [1, 2] + [1] * (sample_x.ndim - 1)
    channel_weight_sqrt = torch.sqrt(channel_weight).view(*weight_shape)

    def _scaled_state(state: torch.Tensor) -> torch.Tensor:
        return (state / channel_scale) * channel_weight_sqrt

    def _predict_state(x_state: torch.Tensor) -> torch.Tensor:
        if self.temporal_enabled:
            if x_state.ndim != 5:
                raise ValueError(
                    "Temporal baseline mode expects [B,2,T,H,W] tensors, "
                    f"got {tuple(x_state.shape)}."
                )
            batch_local, channels_local, time_local, _, _ = x_state.shape
            if int(channels_local) != 2:
                raise ValueError(f"Expected 2 channels for coupled temporal input, got {channels_local}.")
            x_flat = torch.cat([x_state[:, 0], x_state[:, 1]], dim=1)
            pred_flat = self.net(x_flat)
            if pred_flat.ndim != 4 or int(pred_flat.shape[1]) != int(2 * time_local):
                raise ValueError(
                    "Temporal baseline output must have channel-appended shape "
                    f"[B,{2 * int(time_local)},H,W]; got {tuple(pred_flat.shape)}."
                )
            pred_u = pred_flat[:, :time_local]
            pred_v = pred_flat[:, time_local : 2 * time_local]
            return torch.stack([pred_u, pred_v], dim=1).reshape(
                batch_local,
                2,
                time_local,
                pred_flat.shape[-2],
                pred_flat.shape[-1],
            )
        return self.net(x_state)

    clip = max(float(grad_clip), 1e-8)
    wd = max(0.0, float(weight_decay))
    one_cycle_enabled = bool(use_one_cycle_lr)
    pct_start = float(np.clip(one_cycle_pct_start, 1e-4, 0.9999))
    div_factor = max(float(one_cycle_div_factor), 1.0)
    final_div_factor = max(float(one_cycle_final_div_factor), 1.0)
    horizon = max(1, int(rollout_horizon))
    rollout_w = max(0.0, float(rollout_weight))
    dynamics_w = max(0.0, float(dynamics_weight))
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

    use_rollout = (
        (not self.temporal_enabled)
        and
        rollout_w > 0.0
        and horizon > 1
        and trajectory_u is not None
        and trajectory_v is not None
        and len(trajectory_u) > 0
    )
    if use_rollout:
        seq_u = torch.from_numpy(np.asarray(trajectory_u, dtype=np.float32)).to(self.device)
        seq_v = torch.from_numpy(np.asarray(trajectory_v, dtype=np.float32)).to(self.device)
        n_traj, n_steps, _, _ = seq_u.shape
        max_start = n_steps - horizon - 1
        if max_start < 0:
            use_rollout = False
        else:
            rollout_batch = max(1, min(batch // 2, n_traj))
            if early_bias > 0.0:
                start_steps = torch.arange(max_start + 1, device=self.device, dtype=torch.float32)
                rollout_start_weights = 1.0 + early_bias * torch.exp(-start_steps / early_decay)
                rollout_start_weights = rollout_start_weights / torch.sum(rollout_start_weights)
            else:
                rollout_start_weights = None
    else:
        seq_u = None
        seq_v = None
        rollout_batch = 0
        max_start = -1
        rollout_start_weights = None

    optimizer = build_adam_optimizer(
        self.net.parameters(),
        lr=float(lr),
        device=self.device,
        weight_decay=wd,
    )
    total_iter = max(1, int(n_iter))
    steps_per_epoch = max(1, len(train_loader))
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
    checkpoint_callback_arity = 0
    if checkpoint_callback is not None:
        try:
            checkpoint_callback_arity = len(inspect.signature(checkpoint_callback).parameters)
        except (TypeError, ValueError):
            checkpoint_callback_arity = 3

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
                    pred_val = _predict_state(xb_val)
                    one_step_val = self.objective_loss(
                        _scaled_state(pred_val),
                        _scaled_state(yb_val),
                    )
                    dynamics_val = torch.zeros((), device=self.device, dtype=one_step_val.dtype)
                    if dynamics_w > 0.0:
                        dynamics_val = self.objective_loss(
                            _scaled_state(pred_val - xb_val),
                            _scaled_state(yb_val - xb_val),
                        )
                    loss_val = one_step_val + dynamics_w * dynamics_val
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
                pred = _predict_state(xb)
                pred_scaled = _scaled_state(pred)
                target_scaled = _scaled_state(yb)
                one_step_loss = self.objective_loss(pred_scaled, target_scaled)

                rollout_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                if use_rollout and seq_u is not None and seq_v is not None and rollout_batch > 0:
                    traj_idx = torch.randint(0, seq_u.shape[0], (rollout_batch,), device=self.device)
                    if rollout_start_weights is not None:
                        start_idx = torch.multinomial(rollout_start_weights, rollout_batch, replacement=True)
                    else:
                        start_idx = torch.randint(0, max_start + 1, (rollout_batch,), device=self.device)

                    cur_u = seq_u[traj_idx, start_idx]
                    cur_v = seq_v[traj_idx, start_idx]
                    state_roll = torch.stack([cur_u, cur_v], dim=1)

                    for offset in range(1, horizon + 1):
                        pred_roll = self.net(state_roll)
                        tgt_u = seq_u[traj_idx, start_idx + offset]
                        tgt_v = seq_v[traj_idx, start_idx + offset]
                        tgt = torch.stack([tgt_u, tgt_v], dim=1)
                        rollout_loss = rollout_loss + self.objective_loss(
                            _scaled_state(pred_roll),
                            _scaled_state(tgt),
                        )
                        state_roll = pred_roll

                    rollout_loss = rollout_loss / float(horizon)

                dynamics_loss = torch.zeros((), device=self.device, dtype=one_step_loss.dtype)
                if dynamics_w > 0.0:
                    dynamics_loss = self.objective_loss(
                        _scaled_state(pred - xb),
                        _scaled_state(yb - xb),
                    )

                loss = one_step_loss + rollout_w * rollout_loss + dynamics_w * dynamics_loss

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
        if checkpoint_callback is not None:
            if checkpoint_callback_arity >= 3:
                training_state = _capture_training_state(epoch_idx, float(val_loss_value))
                checkpoint_callback(epoch_idx, float(val_loss_value), training_state)
            else:
                checkpoint_callback(epoch_idx, float(val_loss_value))
        if patience is not None and has_val and epochs_without_improvement >= patience:
            break
    if best_state is not None:
        self.net.load_state_dict(best_state)
    self.net.eval()
