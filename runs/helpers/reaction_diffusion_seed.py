"""Single-seed training/evaluation orchestration for reaction-diffusion runs."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from data.reaction_diffusion import GrayScottConfig, GrayScottSolver
from eval.metrics import build_metric_vs_l2 as _build_metric_vs_l2
from eval.metrics import build_paired_bootstrap_gap as _build_paired_bootstrap_gap
from eval.metrics import safe_mean as _safe_mean
from eval.reaction_diffusion import extract_panel_frames as _extract_panel_frames
from eval.reaction_diffusion import extract_target_frame as _extract_target_frame
from models.reaction_diffusion import build_model, rollout_coupled
from runs.helpers.common import load_best_checkpoint_for_eval as _load_best_checkpoint_for_eval
from runs.helpers.common import move_model_device as _move_model_device
from runs.helpers.indexed_datasets import (
    ReactionDiffusionIndexedPairDataset as _ReactionDiffusionIndexedPairDataset,
)
from runs.helpers.indexed_datasets import resolve_dataloader_num_workers as _resolve_dataloader_num_workers
from runs.helpers.reaction_diffusion_training import (
    build_noisy_trajectories_coupled as _build_noisy_trajectories_coupled,
)
from runs.helpers.reaction_diffusion_training import (
    noisy_reference_frame_coupled as _noisy_reference_frame_coupled,
)
from runs.helpers.reaction_diffusion_training import (
    noisy_reference_trajectory_coupled as _noisy_reference_trajectory_coupled,
)
from runs.helpers.reaction_diffusion_training import sample_initial_condition as _sample_initial_condition
from runs.helpers.temporal import resolve_temporal_training_config as _resolve_temporal_training_config
from runs.helpers.temporal import window_start_indices as _window_start_indices
from runs.helpers.temporal import window_target_start as _window_target_start
from utils.diagnostics import ReactionDiffusionRSDAnalyzer
from utils.io import save_checkpoint
from utils.progress import progress_iter
def run_single_seed(
    config: GrayScottConfig,
    method: str,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    basis: str = "fourier",
    operator_config: Mapping[str, Any] | None = None,
    baseline_config: Mapping[str, Any] | None = None,
    eval_pair_index: int = 0,
    test_case_index: int = 0,
    test_step_index: int = 0,
    trajectory_case_indices: List[int] | None = None,
    trajectory_step_indices: List[int] | None = None,
    show_data_progress: bool = False,
    show_training_progress: bool = False,
    show_eval_progress: bool = False,
    preloaded_train_data: List[Dict[str, np.ndarray]] | None = None,
    preloaded_test_cases: List[Dict[str, np.ndarray]] | None = None,
    snapshot_dt: float | None = None,
    data_source: str = "generated",
    data_metadata: Mapping[str, Any] | None = None,
    spectral_band_count: int = 8,
    checkpoint_dir: Path | None = None,
    checkpoint_every_epochs: int = 20,
    resume_clean_state: Dict[str, Any] | None = None,
    resume_noisy_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)

    rsd = ReactionDiffusionRSDAnalyzer(config, basis=basis, spectral_band_count=spectral_band_count)

    if preloaded_train_data is not None and preloaded_test_cases is not None:
        train_data = [
            {
                "u": np.asarray(item["u"], dtype=np.float32),
                "v": np.asarray(item["v"], dtype=np.float32),
            }
            for item in preloaded_train_data
        ]
        test_cases = []
        for item in preloaded_test_cases:
            u_true = np.asarray(item["u_true"], dtype=np.float32)
            v_true = np.asarray(item["v_true"], dtype=np.float32)
            u0 = np.asarray(item.get("u0", u_true[0]), dtype=np.float32)
            v0 = np.asarray(item.get("v0", v_true[0]), dtype=np.float32)
            test_cases.append(
                {
                    "u0": u0,
                    "v0": v0,
                    "u_true": u_true,
                    "v_true": v_true,
                }
            )
        if snapshot_dt is None:
            raise ValueError("snapshot_dt is required when preloaded trajectory data is provided.")
        dt = float(snapshot_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Resolved non-positive dt ({dt}) from preloaded data.")
    else:
        solver = GrayScottSolver(config)
        train_data = []
        for idx in progress_iter(
            range(config.n_train_trajectories),
            enabled=show_data_progress,
            desc="Data gen (train)",
            total=config.n_train_trajectories,
        ):
            u0, v0 = _sample_initial_condition(solver, config, seed=seed * 1000 + idx)
            t_save, u_traj, v_traj = solver.solve(u0, v0, config.t_final, config.n_snapshots)
            train_data.append({"u": u_traj, "v": v_traj})
        dt = float(t_save[1] - t_save[0])

        test_cases = []
        for idx in progress_iter(
            range(config.n_test_trajectories),
            enabled=show_data_progress,
            desc="Data gen (test)",
            total=config.n_test_trajectories,
        ):
            u0, v0 = _sample_initial_condition(solver, config, seed=seed * 1000 + 500 + idx)
            _, u_true, v_true = solver.solve(u0, v0, config.t_final, config.n_snapshots)
            test_cases.append({"u0": u0, "v0": v0, "u_true": u_true, "v_true": v_true})
    temporal_cfg = _resolve_temporal_training_config(method, operator_config, baseline_config)
    temporal_enabled = bool(temporal_cfg["enabled"])
    temporal_window = int(temporal_cfg["input_steps"])
    temporal_target_mode = str(temporal_cfg["target_mode"])
    normalized_method = str(method).strip().lower().replace("-", "_")
    is_rno_method = normalized_method in {"rno", "neuralop_rno", "operator_rno"}
    dataloader_workers = _resolve_dataloader_num_workers(config.train_dataloader_num_workers)
    checkpoint_interval = max(1, int(checkpoint_every_epochs))
    resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    best_val_tracker: Dict[str, float] = {
        "clean": float("inf"),
        "noisy": float("inf"),
    }
    latest_training_state: Dict[str, Dict[str, Any] | None] = {
        "clean": None,
        "noisy": None,
    }

    def _build_checkpoint_payload(
        model_tag: str,
        epoch: int,
        val_loss: float,
        training_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "format": "training_state_v1",
            "phase": str(model_tag),
            "epoch": int(epoch),
            "val_loss": float(val_loss) if np.isfinite(val_loss) else float("nan"),
            "training_state": training_state,
        }

    def _save_checkpoint_event(
        model_tag: str,
        epoch: int,
        val_loss: float,
        training_state: Dict[str, Any],
    ) -> None:
        payload = _build_checkpoint_payload(model_tag, epoch, val_loss, training_state)
        latest_training_state[model_tag] = payload
        if resolved_checkpoint_dir is None:
            return
        epoch_idx = int(epoch)
        if epoch_idx % checkpoint_interval != 0:
            pass
        else:
            save_checkpoint(
                resolved_checkpoint_dir / f"model_{model_tag}_epoch_{epoch_idx:04d}.npz",
                payload,
            )
        if np.isfinite(val_loss) and float(val_loss) < best_val_tracker.get(model_tag, float("inf")):
            best_val_tracker[model_tag] = float(val_loss)
            best_payload = dict(payload)
            best_payload["is_best"] = True
            save_checkpoint(
                resolved_checkpoint_dir / f"model_{model_tag}_best.npz",
                best_payload,
            )

    val_fraction = float(np.clip(config.train_validation_fraction, 0.0, 0.95))
    n_train_traj_total = len(train_data)
    n_val_traj = 0
    if n_train_traj_total > 1 and val_fraction > 0.0:
        n_val_traj = int(round(n_train_traj_total * val_fraction))
        n_val_traj = max(1, min(n_train_traj_total - 1, n_val_traj))

    split_rng = np.random.default_rng(seed * 1000 + 907)
    split_perm = list(split_rng.permutation(n_train_traj_total))
    val_idx_set = set(split_perm[:n_val_traj])
    fit_train_data = [traj for idx, traj in enumerate(train_data) if idx not in val_idx_set]
    val_data = [traj for idx, traj in enumerate(train_data) if idx in val_idx_set]
    if not fit_train_data:
        fit_train_data = train_data
        val_data = []

    if trajectory_case_indices:
        requested_cases = [int(idx) for idx in trajectory_case_indices]
    else:
        requested_cases = [0]
    valid_trajectory_case_indices = sorted(
        {
            int(np.clip(idx, 0, len(test_cases) - 1))
            for idx in requested_cases
        }
    )
    if not valid_trajectory_case_indices:
        valid_trajectory_case_indices = [0]
    trajectory_case_set = set(valid_trajectory_case_indices)

    if trajectory_step_indices:
        requested_steps = [int(step) for step in trajectory_step_indices]
    else:
        requested_steps = [0, config.n_snapshots // 4, config.n_snapshots // 2, (3 * config.n_snapshots) // 4, config.n_snapshots - 1]
    valid_trajectory_steps = sorted(
        {
            int(np.clip(step, 0, config.n_snapshots - 1))
            for step in requested_steps
        }
    )
    if not valid_trajectory_steps:
        valid_trajectory_steps = [0, config.n_snapshots - 1]

    train_dataset_clean = _ReactionDiffusionIndexedPairDataset(
        fit_train_data,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    val_dataset_clean = _ReactionDiffusionIndexedPairDataset(
        val_data,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    pair_steps = train_dataset_clean.pair_steps

    if len(train_dataset_clean) == 0:
        raise ValueError(
            "No training pairs were generated. "
            f"temporal_enabled={temporal_enabled}, temporal_window={temporal_window}, "
            f"n_snapshots={config.n_snapshots}"
        )

    eval_pair_index = int(np.clip(eval_pair_index, 0, len(train_dataset_clean) - 1))
    (
        eval_input_u_reference,
        eval_input_v_reference,
        eval_target_u_clean_reference,
        eval_target_v_clean_reference,
    ) = train_dataset_clean.get_pair_arrays(eval_pair_index)

    model_clean = build_model(
        method,
        config.nx,
        config.ny,
        seed=seed,
        device=device,
        loss=loss,
        config=config,
        snapshot_dt=dt,
        operator_config=operator_config,
        baseline_config=baseline_config,
        temporal_config=temporal_cfg,
    )
    def _clean_checkpoint_callback(epoch: int, val_loss: float, training_state: Dict[str, Any]) -> None:
        _save_checkpoint_event("clean", epoch, val_loss, training_state)

    train_trajectory_u = [np.asarray(item["u"], dtype=np.float32) for item in fit_train_data]
    train_trajectory_v = [np.asarray(item["v"], dtype=np.float32) for item in fit_train_data]
    rollout_trajectory_u_clean = (
        train_trajectory_u
        if is_rno_method or (config.train_rollout_weight > 0.0 and config.train_rollout_horizon > 1)
        else None
    )
    rollout_trajectory_v_clean = (
        train_trajectory_v
        if is_rno_method or (config.train_rollout_weight > 0.0 and config.train_rollout_horizon > 1)
        else None
    )

    model_clean.train(
        inputs_u=[],
        inputs_v=[],
        targets_u=[],
        targets_v=[],
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        weight_decay=config.train_weight_decay,
        use_one_cycle_lr=config.train_use_one_cycle_lr,
        one_cycle_pct_start=config.train_one_cycle_pct_start,
        one_cycle_div_factor=config.train_one_cycle_div_factor,
        one_cycle_final_div_factor=config.train_one_cycle_final_div_factor,
        trajectory_u=rollout_trajectory_u_clean,
        trajectory_v=rollout_trajectory_v_clean,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        val_inputs_u=[],
        val_inputs_v=[],
        val_targets_u=[],
        val_targets_v=[],
        pair_steps=pair_steps,
        u_weight=config.train_u_weight,
        v_weight=config.train_v_weight,
        channel_balance_cap=config.train_channel_balance_cap,
        dynamics_weight=config.train_dynamics_weight,
        early_step_bias=config.train_early_step_bias,
        early_step_decay=config.train_early_step_decay,
        checkpoint_callback=_clean_checkpoint_callback,
        early_stopping_patience=config.train_early_stopping_patience,
        resume_state=resume_clean_state,
        train_dataset=train_dataset_clean,
        val_dataset=val_dataset_clean if len(val_dataset_clean) > 0 else None,
        dataloader_num_workers=dataloader_workers,
        show_progress=show_training_progress,
        progress_desc="Training clean model",
    )
    if resolved_checkpoint_dir is not None:
        clean_payload = latest_training_state.get("clean")
        if clean_payload is None:
            clean_payload = {
                "format": "training_state_v1",
                "phase": "clean",
                "epoch": int(config.train_iterations),
                "val_loss": float("nan"),
                "training_state": {"model_state": model_clean.state_dict()},
            }
        else:
            clean_payload = dict(clean_payload)
        clean_payload["is_final"] = True
        save_checkpoint(resolved_checkpoint_dir / "model_clean.npz", clean_payload)
    if str(device).lower() == "cuda":
        _move_model_device(model_clean, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del train_dataset_clean, val_dataset_clean
    del train_trajectory_u, train_trajectory_v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fit_train_data_noisy = _build_noisy_trajectories_coupled(
        fit_train_data,
        config,
        show_progress=show_data_progress,
        progress_desc="Build noisy train trajectories",
    )
    val_data_noisy = _build_noisy_trajectories_coupled(
        val_data,
        config,
        show_progress=False,
        progress_desc="Build noisy validation trajectories",
    )
    train_dataset_noisy = _ReactionDiffusionIndexedPairDataset(
        fit_train_data_noisy,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    val_dataset_noisy = _ReactionDiffusionIndexedPairDataset(
        val_data_noisy,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    if len(train_dataset_noisy) == 0:
        raise ValueError(
            "No noisy training pairs were generated. "
            f"temporal_enabled={temporal_enabled}, temporal_window={temporal_window}, "
            f"n_snapshots={config.n_snapshots}"
        )
    if eval_pair_index >= len(train_dataset_noisy):
        raise ValueError(
            "Noisy training pair count differs from clean training pair count "
            f"(clean_eval_index={eval_pair_index}, noisy_pairs={len(train_dataset_noisy)})."
        )
    (
        _,
        _,
        eval_target_u_noisy_reference,
        eval_target_v_noisy_reference,
    ) = train_dataset_noisy.get_pair_arrays(eval_pair_index)
    n_train_total = len(train_data)
    del fit_train_data, val_data, train_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_noisy = build_model(
        method,
        config.nx,
        config.ny,
        seed=seed + 10000,
        device=device,
        loss=loss,
        config=config,
        snapshot_dt=dt,
        operator_config=operator_config,
        baseline_config=baseline_config,
        temporal_config=temporal_cfg,
    )
    def _noisy_checkpoint_callback(epoch: int, val_loss: float, training_state: Dict[str, Any]) -> None:
        _save_checkpoint_event("noisy", epoch, val_loss, training_state)

    train_trajectory_u_noisy = [np.asarray(item["u"], dtype=np.float32) for item in fit_train_data_noisy]
    train_trajectory_v_noisy = [np.asarray(item["v"], dtype=np.float32) for item in fit_train_data_noisy]
    rollout_trajectory_u_noisy = (
        train_trajectory_u_noisy
        if is_rno_method or (config.train_rollout_weight > 0.0 and config.train_rollout_horizon > 1)
        else None
    )
    rollout_trajectory_v_noisy = (
        train_trajectory_v_noisy
        if is_rno_method or (config.train_rollout_weight > 0.0 and config.train_rollout_horizon > 1)
        else None
    )

    model_noisy.train(
        inputs_u=[],
        inputs_v=[],
        targets_u=[],
        targets_v=[],
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        weight_decay=config.train_weight_decay,
        use_one_cycle_lr=config.train_use_one_cycle_lr,
        one_cycle_pct_start=config.train_one_cycle_pct_start,
        one_cycle_div_factor=config.train_one_cycle_div_factor,
        one_cycle_final_div_factor=config.train_one_cycle_final_div_factor,
        trajectory_u=rollout_trajectory_u_noisy,
        trajectory_v=rollout_trajectory_v_noisy,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        val_inputs_u=[],
        val_inputs_v=[],
        val_targets_u=[],
        val_targets_v=[],
        pair_steps=pair_steps,
        u_weight=config.train_u_weight,
        v_weight=config.train_v_weight,
        channel_balance_cap=config.train_channel_balance_cap,
        dynamics_weight=config.train_dynamics_weight,
        early_step_bias=config.train_early_step_bias,
        early_step_decay=config.train_early_step_decay,
        checkpoint_callback=_noisy_checkpoint_callback,
        early_stopping_patience=config.train_early_stopping_patience,
        resume_state=resume_noisy_state,
        train_dataset=train_dataset_noisy,
        val_dataset=val_dataset_noisy if len(val_dataset_noisy) > 0 else None,
        dataloader_num_workers=dataloader_workers,
        show_progress=show_training_progress,
        progress_desc="Training noisy model",
    )
    if resolved_checkpoint_dir is not None:
        noisy_payload = latest_training_state.get("noisy")
        if noisy_payload is None:
            noisy_payload = {
                "format": "training_state_v1",
                "phase": "noisy",
                "epoch": int(config.train_iterations),
                "val_loss": float("nan"),
                "training_state": {"model_state": model_noisy.state_dict()},
            }
        else:
            noisy_payload = dict(noisy_payload)
        noisy_payload["is_final"] = True
        save_checkpoint(resolved_checkpoint_dir / "model_noisy.npz", noisy_payload)
    if str(device).lower() == "cuda":
        _move_model_device(model_clean, "cuda")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del train_dataset_noisy, val_dataset_noisy
    del train_trajectory_u_noisy, train_trajectory_v_noisy
    del fit_train_data_noisy, val_data_noisy, pair_steps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _load_best_checkpoint_for_eval(model_clean, resolved_checkpoint_dir, "clean")
    _load_best_checkpoint_for_eval(model_noisy, resolved_checkpoint_dir, "noisy")

    metrics = {
        "clean_l2": [],
        "noisy_l2": [],
        "clean_l2_u": [],
        "noisy_l2_u": [],
        "clean_l2_v": [],
        "noisy_l2_v": [],
        "clean_hfv": [],
        "noisy_hfv": [],
        "clean_lfv": [],
        "noisy_lfv": [],
        "clean_pde_residual_st_rms": [],
        "noisy_pde_residual_st_rms": [],
        "clean_pde_residual_st_rms_u": [],
        "noisy_pde_residual_st_rms_u": [],
        "clean_pde_residual_st_rms_v": [],
        "noisy_pde_residual_st_rms_v": [],
        "clean_boundary_error": [],
        "noisy_boundary_error": [],
        "clean_spectral_multiband_error": [],
        "noisy_spectral_multiband_error": [],
        "clean_spectral_low_error": [],
        "noisy_spectral_low_error": [],
        "clean_spectral_mid_error": [],
        "noisy_spectral_mid_error": [],
        "clean_spectral_high_error": [],
        "noisy_spectral_high_error": [],
        "clean_spectral_multiband_error_vs_clean_gt": [],
        "noisy_spectral_multiband_error_vs_clean_gt": [],
        "clean_spectral_low_error_vs_clean_gt": [],
        "noisy_spectral_low_error_vs_clean_gt": [],
        "clean_spectral_mid_error_vs_clean_gt": [],
        "noisy_spectral_mid_error_vs_clean_gt": [],
        "clean_spectral_high_error_vs_clean_gt": [],
        "noisy_spectral_high_error_vs_clean_gt": [],
        "clean_fourier_coeff_mse_multiband_vs_clean_gt": [],
        "noisy_fourier_coeff_mse_multiband_vs_clean_gt": [],
        "clean_fourier_coeff_mse_low_vs_clean_gt": [],
        "noisy_fourier_coeff_mse_low_vs_clean_gt": [],
        "clean_fourier_coeff_mse_mid_vs_clean_gt": [],
        "noisy_fourier_coeff_mse_mid_vs_clean_gt": [],
        "clean_fourier_coeff_mse_high_vs_clean_gt": [],
        "noisy_fourier_coeff_mse_high_vs_clean_gt": [],
    }
    spectral_band_labels = list(rsd.spectral_band_labels)
    for band_label in spectral_band_labels:
        metrics[f"clean_spectral_band_error_{band_label}"] = []
        metrics[f"noisy_spectral_band_error_{band_label}"] = []
        metrics[f"clean_spectral_band_error_vs_clean_gt_{band_label}"] = []
        metrics[f"noisy_spectral_band_error_vs_clean_gt_{band_label}"] = []
        metrics[f"clean_fourier_coeff_mse_band_vs_clean_gt_{band_label}"] = []
        metrics[f"noisy_fourier_coeff_mse_band_vs_clean_gt_{band_label}"] = []

    rollout_context: int | None = temporal_window if temporal_enabled else None

    trajectory_rows = []
    for case_idx, case in enumerate(
        progress_iter(
            test_cases,
            enabled=show_eval_progress,
            desc="Evaluation",
            total=len(test_cases),
        )
    ):
        if rollout_context is not None:
            u_context = np.asarray(case["u_true"][:rollout_context], dtype=np.float32)
            v_context = np.asarray(case["v_true"][:rollout_context], dtype=np.float32)
        else:
            u_context = None
            v_context = None
        u_clean, v_clean = rollout_coupled(
            model_clean,
            case["u0"],
            case["v0"],
            config.n_snapshots,
            context_u=u_context,
            context_v=v_context,
        )
        u_noisy, v_noisy = rollout_coupled(
            model_noisy,
            case["u0"],
            case["v0"],
            config.n_snapshots,
            context_u=u_context,
            context_v=v_context,
        )

        if case_idx in trajectory_case_set:
            trajectory_rows.append(
                {
                    "case_index": case_idx,
                    "model": "clean",
                    "u_pred": u_clean,
                    "v_pred": v_clean,
                    "u_true": case["u_true"],
                    "v_true": case["v_true"],
                }
            )
            trajectory_rows.append(
                {
                    "case_index": case_idx,
                    "model": "noisy",
                    "u_pred": u_noisy,
                    "v_pred": v_noisy,
                    "u_true": case["u_true"],
                    "v_true": case["v_true"],
                }
            )

        clean_stats = rsd.compute_metrics(u_clean, v_clean, case["u_true"], case["v_true"], dt)
        noisy_stats = rsd.compute_metrics(u_noisy, v_noisy, case["u_true"], case["v_true"], dt)

        metrics["clean_l2"].append(clean_stats["l2_error"])
        metrics["noisy_l2"].append(noisy_stats["l2_error"])
        metrics["clean_l2_u"].append(clean_stats["l2_u"])
        metrics["noisy_l2_u"].append(noisy_stats["l2_u"])
        metrics["clean_l2_v"].append(clean_stats["l2_v"])
        metrics["noisy_l2_v"].append(noisy_stats["l2_v"])
        metrics["clean_hfv"].append(clean_stats["hfv"])
        metrics["noisy_hfv"].append(noisy_stats["hfv"])
        metrics["clean_lfv"].append(clean_stats["lfv"])
        metrics["noisy_lfv"].append(noisy_stats["lfv"])
        metrics["clean_pde_residual_st_rms"].append(clean_stats["pde_residual_st_rms"])
        metrics["noisy_pde_residual_st_rms"].append(noisy_stats["pde_residual_st_rms"])
        metrics["clean_pde_residual_st_rms_u"].append(clean_stats["pde_residual_st_rms_u"])
        metrics["noisy_pde_residual_st_rms_u"].append(noisy_stats["pde_residual_st_rms_u"])
        metrics["clean_pde_residual_st_rms_v"].append(clean_stats["pde_residual_st_rms_v"])
        metrics["noisy_pde_residual_st_rms_v"].append(noisy_stats["pde_residual_st_rms_v"])
        metrics["clean_boundary_error"].append(clean_stats["boundary_error"])
        metrics["noisy_boundary_error"].append(noisy_stats["boundary_error"])
        metrics["clean_spectral_multiband_error"].append(clean_stats["spectral_multiband_error"])
        metrics["noisy_spectral_multiband_error"].append(noisy_stats["spectral_multiband_error"])
        metrics["clean_spectral_low_error"].append(clean_stats["spectral_low_error"])
        metrics["noisy_spectral_low_error"].append(noisy_stats["spectral_low_error"])
        metrics["clean_spectral_mid_error"].append(clean_stats["spectral_mid_error"])
        metrics["noisy_spectral_mid_error"].append(noisy_stats["spectral_mid_error"])
        metrics["clean_spectral_high_error"].append(clean_stats["spectral_high_error"])
        metrics["noisy_spectral_high_error"].append(noisy_stats["spectral_high_error"])
        metrics["clean_spectral_multiband_error_vs_clean_gt"].append(
            clean_stats["spectral_multiband_error_vs_clean_gt"]
        )
        metrics["noisy_spectral_multiband_error_vs_clean_gt"].append(
            noisy_stats["spectral_multiband_error_vs_clean_gt"]
        )
        metrics["clean_spectral_low_error_vs_clean_gt"].append(clean_stats["spectral_low_error_vs_clean_gt"])
        metrics["noisy_spectral_low_error_vs_clean_gt"].append(noisy_stats["spectral_low_error_vs_clean_gt"])
        metrics["clean_spectral_mid_error_vs_clean_gt"].append(clean_stats["spectral_mid_error_vs_clean_gt"])
        metrics["noisy_spectral_mid_error_vs_clean_gt"].append(noisy_stats["spectral_mid_error_vs_clean_gt"])
        metrics["clean_spectral_high_error_vs_clean_gt"].append(clean_stats["spectral_high_error_vs_clean_gt"])
        metrics["noisy_spectral_high_error_vs_clean_gt"].append(noisy_stats["spectral_high_error_vs_clean_gt"])
        metrics["clean_fourier_coeff_mse_multiband_vs_clean_gt"].append(
            clean_stats["fourier_coeff_mse_multiband_vs_clean_gt"]
        )
        metrics["noisy_fourier_coeff_mse_multiband_vs_clean_gt"].append(
            noisy_stats["fourier_coeff_mse_multiband_vs_clean_gt"]
        )
        metrics["clean_fourier_coeff_mse_low_vs_clean_gt"].append(
            clean_stats["fourier_coeff_mse_low_vs_clean_gt"]
        )
        metrics["noisy_fourier_coeff_mse_low_vs_clean_gt"].append(
            noisy_stats["fourier_coeff_mse_low_vs_clean_gt"]
        )
        metrics["clean_fourier_coeff_mse_mid_vs_clean_gt"].append(
            clean_stats["fourier_coeff_mse_mid_vs_clean_gt"]
        )
        metrics["noisy_fourier_coeff_mse_mid_vs_clean_gt"].append(
            noisy_stats["fourier_coeff_mse_mid_vs_clean_gt"]
        )
        metrics["clean_fourier_coeff_mse_high_vs_clean_gt"].append(
            clean_stats["fourier_coeff_mse_high_vs_clean_gt"]
        )
        metrics["noisy_fourier_coeff_mse_high_vs_clean_gt"].append(
            noisy_stats["fourier_coeff_mse_high_vs_clean_gt"]
        )

        for band_idx, band_label in enumerate(spectral_band_labels):
            clean_band_fraction = float(clean_stats["spectral_band_error_vs_clean_gt"][band_idx])
            noisy_band_fraction = float(noisy_stats["spectral_band_error_vs_clean_gt"][band_idx])
            metrics[f"clean_spectral_band_error_{band_label}"].append(clean_band_fraction)
            metrics[f"noisy_spectral_band_error_{band_label}"].append(noisy_band_fraction)
            metrics[f"clean_spectral_band_error_vs_clean_gt_{band_label}"].append(clean_band_fraction)
            metrics[f"noisy_spectral_band_error_vs_clean_gt_{band_label}"].append(noisy_band_fraction)

            clean_band_coeff_mse = float(clean_stats["fourier_coeff_mse_band_vs_clean_gt"][band_idx])
            noisy_band_coeff_mse = float(noisy_stats["fourier_coeff_mse_band_vs_clean_gt"][band_idx])
            metrics[f"clean_fourier_coeff_mse_band_vs_clean_gt_{band_label}"].append(clean_band_coeff_mse)
            metrics[f"noisy_fourier_coeff_mse_band_vs_clean_gt_{band_label}"].append(noisy_band_coeff_mse)

    test_case_index = int(np.clip(test_case_index, 0, len(test_cases) - 1))
    test_case = test_cases[test_case_index]
    if temporal_enabled:
        eval_input_u_window = np.asarray(eval_input_u_reference, dtype=np.float32)
        eval_input_v_window = np.asarray(eval_input_v_reference, dtype=np.float32)
        eval_target_u_window = np.asarray(eval_target_u_clean_reference, dtype=np.float32)
        eval_target_v_window = np.asarray(eval_target_v_clean_reference, dtype=np.float32)
        eval_target_u_window_noisy = np.asarray(eval_target_u_noisy_reference, dtype=np.float32)
        eval_target_v_window_noisy = np.asarray(eval_target_v_noisy_reference, dtype=np.float32)
        eval_pred_u_window_clean, eval_pred_v_window_clean = model_clean.predict_window(
            eval_input_u_window,
            eval_input_v_window,
        )
        eval_pred_u_window_noisy, eval_pred_v_window_noisy = model_noisy.predict_window(
            eval_input_u_window,
            eval_input_v_window,
        )
        (
            eval_input_u,
            eval_input_v,
            eval_target_u,
            eval_target_v,
            eval_pred_u_clean,
            eval_pred_v_clean,
        ) = _extract_panel_frames(
            eval_input_u_window,
            eval_input_v_window,
            eval_target_u_window,
            eval_target_v_window,
            np.asarray(eval_pred_u_window_clean, dtype=np.float32),
            np.asarray(eval_pred_v_window_clean, dtype=np.float32),
            temporal_target_mode,
        )
        eval_target_u_noisy = _extract_target_frame(eval_target_u_window_noisy, temporal_target_mode)
        eval_target_v_noisy = _extract_target_frame(eval_target_v_window_noisy, temporal_target_mode)
        (
            _,
            _,
            _,
            _,
            eval_pred_u_noisy,
            eval_pred_v_noisy,
        ) = _extract_panel_frames(
            eval_input_u_window,
            eval_input_v_window,
            eval_target_u_window,
            eval_target_v_window,
            np.asarray(eval_pred_u_window_noisy, dtype=np.float32),
            np.asarray(eval_pred_v_window_noisy, dtype=np.float32),
            temporal_target_mode,
        )

        test_start_candidates = list(
            _window_start_indices(test_case["u_true"].shape[0], temporal_window, temporal_target_mode)
        )
        if not test_start_candidates:
            raise ValueError(
                "Temporal evaluation requested but test trajectory is too short for configured window: "
                f"n_steps={test_case['u_true'].shape[0]}, window={temporal_window}, "
                f"target_mode={temporal_target_mode}"
            )
        max_test_step = max(test_start_candidates)
        test_step_index = int(np.clip(test_step_index, 0, max_test_step))
        target_start = _window_target_start(test_step_index, temporal_window, temporal_target_mode)
        test_input_u_window = np.asarray(
            test_case["u_true"][test_step_index : test_step_index + temporal_window],
            dtype=np.float32,
        )
        test_input_v_window = np.asarray(
            test_case["v_true"][test_step_index : test_step_index + temporal_window],
            dtype=np.float32,
        )
        test_target_u_window = np.asarray(
            test_case["u_true"][target_start : target_start + temporal_window],
            dtype=np.float32,
        )
        test_target_v_window = np.asarray(
            test_case["v_true"][target_start : target_start + temporal_window],
            dtype=np.float32,
        )
        test_target_u_window_noisy, test_target_v_window_noisy = _noisy_reference_trajectory_coupled(
            test_target_u_window,
            test_target_v_window,
            config,
            rng_seed=seed * 1_000_000 + test_case_index * 10_000 + target_start,
        )
        test_pred_u_window_clean, test_pred_v_window_clean = model_clean.predict_window(
            test_input_u_window,
            test_input_v_window,
        )
        test_pred_u_window_noisy, test_pred_v_window_noisy = model_noisy.predict_window(
            test_input_u_window,
            test_input_v_window,
        )
        (
            test_input_u,
            test_input_v,
            test_target_u,
            test_target_v,
            test_pred_u_clean,
            test_pred_v_clean,
        ) = _extract_panel_frames(
            test_input_u_window,
            test_input_v_window,
            test_target_u_window,
            test_target_v_window,
            np.asarray(test_pred_u_window_clean, dtype=np.float32),
            np.asarray(test_pred_v_window_clean, dtype=np.float32),
            temporal_target_mode,
        )
        test_target_u_noisy = _extract_target_frame(test_target_u_window_noisy, temporal_target_mode)
        test_target_v_noisy = _extract_target_frame(test_target_v_window_noisy, temporal_target_mode)
        (
            _,
            _,
            _,
            _,
            test_pred_u_noisy,
            test_pred_v_noisy,
        ) = _extract_panel_frames(
            test_input_u_window,
            test_input_v_window,
            test_target_u_window,
            test_target_v_window,
            np.asarray(test_pred_u_window_noisy, dtype=np.float32),
            np.asarray(test_pred_v_window_noisy, dtype=np.float32),
            temporal_target_mode,
        )
    else:
        max_test_step = test_case["u_true"].shape[0] - 2
        test_step_index = int(np.clip(test_step_index, 0, max_test_step))

        eval_input_u = np.asarray(eval_input_u_reference, dtype=np.float32)
        eval_input_v = np.asarray(eval_input_v_reference, dtype=np.float32)
        eval_target_u = np.asarray(eval_target_u_clean_reference, dtype=np.float32)
        eval_target_v = np.asarray(eval_target_v_clean_reference, dtype=np.float32)
        eval_target_u_noisy = np.asarray(eval_target_u_noisy_reference, dtype=np.float32)
        eval_target_v_noisy = np.asarray(eval_target_v_noisy_reference, dtype=np.float32)
        eval_pred_u_clean, eval_pred_v_clean = model_clean.forward(eval_input_u, eval_input_v)
        eval_pred_u_noisy, eval_pred_v_noisy = model_noisy.forward(eval_input_u, eval_input_v)

        test_input_u = test_case["u_true"][test_step_index]
        test_input_v = test_case["v_true"][test_step_index]
        test_target_u = test_case["u_true"][test_step_index + 1]
        test_target_v = test_case["v_true"][test_step_index + 1]
        test_target_u_noisy, test_target_v_noisy = _noisy_reference_frame_coupled(
            test_target_u,
            test_target_v,
            config,
            rng_seed=seed * 1_000_000 + test_case_index * 10_000 + test_step_index + 1,
        )
        test_pred_u_clean, test_pred_v_clean = model_clean.forward(test_input_u, test_input_v)
        test_pred_u_noisy, test_pred_v_noisy = model_noisy.forward(test_input_u, test_input_v)

    mean_metrics = {key: _safe_mean(value) for key, value in metrics.items()}
    compare_metric_ids = [
        "hfv",
        "lfv",
        "pde_residual_st_rms",
        "boundary_error",
        "spectral_multiband_error",
        "spectral_low_error",
        "spectral_mid_error",
        "spectral_high_error",
        "spectral_multiband_error_vs_clean_gt",
        "fourier_coeff_mse_multiband_vs_clean_gt",
        "fourier_coeff_mse_low_vs_clean_gt",
        "fourier_coeff_mse_mid_vs_clean_gt",
        "fourier_coeff_mse_high_vs_clean_gt",
    ]
    metric_vs_l2 = _build_metric_vs_l2(metrics, compare_metric_ids)
    clean_spectral_band_error_mean = [
        _safe_mean(metrics[f"clean_spectral_band_error_{label}"])
        for label in spectral_band_labels
    ]
    noisy_spectral_band_error_mean = [
        _safe_mean(metrics[f"noisy_spectral_band_error_{label}"])
        for label in spectral_band_labels
    ]
    clean_spectral_band_error_vs_clean_gt_mean = [
        _safe_mean(metrics[f"clean_spectral_band_error_vs_clean_gt_{label}"])
        for label in spectral_band_labels
    ]
    noisy_spectral_band_error_vs_clean_gt_mean = [
        _safe_mean(metrics[f"noisy_spectral_band_error_vs_clean_gt_{label}"])
        for label in spectral_band_labels
    ]
    clean_fourier_coeff_mse_band_vs_clean_gt_mean = [
        _safe_mean(metrics[f"clean_fourier_coeff_mse_band_vs_clean_gt_{label}"])
        for label in spectral_band_labels
    ]
    noisy_fourier_coeff_mse_band_vs_clean_gt_mean = [
        _safe_mean(metrics[f"noisy_fourier_coeff_mse_band_vs_clean_gt_{label}"])
        for label in spectral_band_labels
    ]

    bootstrap_n = 5000
    bootstrap_ci_level = 0.95
    bootstrap_seed_base = int(seed) * 100_000 + int(len(test_cases))
    fraction_multiband_gap_bootstrap = _build_paired_bootstrap_gap(
        metrics["clean_spectral_multiband_error_vs_clean_gt"],
        metrics["noisy_spectral_multiband_error_vs_clean_gt"],
        n_bootstrap=bootstrap_n,
        ci_level=bootstrap_ci_level,
        rng_seed=bootstrap_seed_base + 1,
    )
    coeff_mse_multiband_gap_bootstrap = _build_paired_bootstrap_gap(
        metrics["clean_fourier_coeff_mse_multiband_vs_clean_gt"],
        metrics["noisy_fourier_coeff_mse_multiband_vs_clean_gt"],
        n_bootstrap=bootstrap_n,
        ci_level=bootstrap_ci_level,
        rng_seed=bootstrap_seed_base + 2,
    )
    fraction_band_gap_bootstrap = []
    coeff_mse_band_gap_bootstrap = []
    for band_idx, band_label in enumerate(spectral_band_labels):
        fraction_band_gap_bootstrap.append(
            _build_paired_bootstrap_gap(
                metrics[f"clean_spectral_band_error_vs_clean_gt_{band_label}"],
                metrics[f"noisy_spectral_band_error_vs_clean_gt_{band_label}"],
                n_bootstrap=bootstrap_n,
                ci_level=bootstrap_ci_level,
                rng_seed=bootstrap_seed_base + 100 + band_idx,
            )
        )
        coeff_mse_band_gap_bootstrap.append(
            _build_paired_bootstrap_gap(
                metrics[f"clean_fourier_coeff_mse_band_vs_clean_gt_{band_label}"],
                metrics[f"noisy_fourier_coeff_mse_band_vs_clean_gt_{band_label}"],
                n_bootstrap=bootstrap_n,
                ci_level=bootstrap_ci_level,
                rng_seed=bootstrap_seed_base + 1000 + band_idx,
            )
        )

    return {
        **mean_metrics,
        "metric_vs_l2": metric_vs_l2,
        "spectral_bands": {
            "count": int(len(spectral_band_labels)),
            "labels": spectral_band_labels,
            "centers": [float(v) for v in rsd.spectral_band_centers],
            "clean_band_error_mean": clean_spectral_band_error_mean,
            "noisy_band_error_mean": noisy_spectral_band_error_mean,
            "clean_band_error_vs_clean_gt_mean": clean_spectral_band_error_vs_clean_gt_mean,
            "noisy_band_error_vs_clean_gt_mean": noisy_spectral_band_error_vs_clean_gt_mean,
            "clean_fourier_coeff_mse_band_vs_clean_gt_mean": clean_fourier_coeff_mse_band_vs_clean_gt_mean,
            "noisy_fourier_coeff_mse_band_vs_clean_gt_mean": noisy_fourier_coeff_mse_band_vs_clean_gt_mean,
        },
        "spectral_clean_reference_checks": {
            "bootstrap_config": {
                "n_bootstrap": int(bootstrap_n),
                "ci_level": float(bootstrap_ci_level),
                "gap_definition": "noisy_minus_clean",
            },
            "fractional_band_error": {
                "clean_multiband_mean": mean_metrics["clean_spectral_multiband_error_vs_clean_gt"],
                "noisy_multiband_mean": mean_metrics["noisy_spectral_multiband_error_vs_clean_gt"],
                "clean_low_mean": mean_metrics["clean_spectral_low_error_vs_clean_gt"],
                "noisy_low_mean": mean_metrics["noisy_spectral_low_error_vs_clean_gt"],
                "clean_mid_mean": mean_metrics["clean_spectral_mid_error_vs_clean_gt"],
                "noisy_mid_mean": mean_metrics["noisy_spectral_mid_error_vs_clean_gt"],
                "clean_high_mean": mean_metrics["clean_spectral_high_error_vs_clean_gt"],
                "noisy_high_mean": mean_metrics["noisy_spectral_high_error_vs_clean_gt"],
                "clean_band_error_mean": clean_spectral_band_error_vs_clean_gt_mean,
                "noisy_band_error_mean": noisy_spectral_band_error_vs_clean_gt_mean,
                "multiband_gap_bootstrap": fraction_multiband_gap_bootstrap,
                "band_gap_bootstrap": fraction_band_gap_bootstrap,
            },
            "fourier_coeff_mse_band": {
                "clean_multiband_mean": mean_metrics["clean_fourier_coeff_mse_multiband_vs_clean_gt"],
                "noisy_multiband_mean": mean_metrics["noisy_fourier_coeff_mse_multiband_vs_clean_gt"],
                "clean_low_mean": mean_metrics["clean_fourier_coeff_mse_low_vs_clean_gt"],
                "noisy_low_mean": mean_metrics["noisy_fourier_coeff_mse_low_vs_clean_gt"],
                "clean_mid_mean": mean_metrics["clean_fourier_coeff_mse_mid_vs_clean_gt"],
                "noisy_mid_mean": mean_metrics["noisy_fourier_coeff_mse_mid_vs_clean_gt"],
                "clean_high_mean": mean_metrics["clean_fourier_coeff_mse_high_vs_clean_gt"],
                "noisy_high_mean": mean_metrics["noisy_fourier_coeff_mse_high_vs_clean_gt"],
                "clean_band_mse_mean": clean_fourier_coeff_mse_band_vs_clean_gt_mean,
                "noisy_band_mse_mean": noisy_fourier_coeff_mse_band_vs_clean_gt_mean,
                "multiband_gap_bootstrap": coeff_mse_multiband_gap_bootstrap,
                "band_gap_bootstrap": coeff_mse_band_gap_bootstrap,
            },
        },
        "_viz": {
            "eval": {
                "input_u": eval_input_u,
                "input_v": eval_input_v,
                "target_u": eval_target_u,
                "target_v": eval_target_v,
                "target_u_noisy": eval_target_u_noisy,
                "target_v_noisy": eval_target_v_noisy,
                "pred_u_clean": eval_pred_u_clean,
                "pred_v_clean": eval_pred_v_clean,
                "pred_u_noisy": eval_pred_u_noisy,
                "pred_v_noisy": eval_pred_v_noisy,
            },
            "test": {
                "input_u": test_input_u,
                "input_v": test_input_v,
                "target_u": test_target_u,
                "target_v": test_target_v,
                "target_u_noisy": test_target_u_noisy,
                "target_v_noisy": test_target_v_noisy,
                "pred_u_clean": test_pred_u_clean,
                "pred_v_clean": test_pred_v_clean,
                "pred_u_noisy": test_pred_u_noisy,
                "pred_v_noisy": test_pred_v_noisy,
            },
            "indices": {
                "eval_pair_index": eval_pair_index,
                "test_case_index": test_case_index,
                "test_step_index": test_step_index,
            },
            "trajectory": {
                "case_indices": valid_trajectory_case_indices,
                "step_indices": valid_trajectory_steps,
                "rows": trajectory_rows,
            },
            "diagnostics": {
                "series": {
                    "clean_l2": list(metrics["clean_l2"]),
                    "noisy_l2": list(metrics["noisy_l2"]),
                    "clean_hfv": list(metrics["clean_hfv"]),
                    "noisy_hfv": list(metrics["noisy_hfv"]),
                    "clean_lfv": list(metrics["clean_lfv"]),
                    "noisy_lfv": list(metrics["noisy_lfv"]),
                    "clean_pde_residual_st_rms": list(metrics["clean_pde_residual_st_rms"]),
                    "noisy_pde_residual_st_rms": list(metrics["noisy_pde_residual_st_rms"]),
                    "clean_boundary_error": list(metrics["clean_boundary_error"]),
                    "noisy_boundary_error": list(metrics["noisy_boundary_error"]),
                    "clean_spectral_multiband_error": list(metrics["clean_spectral_multiband_error"]),
                    "noisy_spectral_multiband_error": list(metrics["noisy_spectral_multiband_error"]),
                    "clean_spectral_multiband_error_vs_clean_gt": list(
                        metrics["clean_spectral_multiband_error_vs_clean_gt"]
                    ),
                    "noisy_spectral_multiband_error_vs_clean_gt": list(
                        metrics["noisy_spectral_multiband_error_vs_clean_gt"]
                    ),
                    "clean_fourier_coeff_mse_multiband_vs_clean_gt": list(
                        metrics["clean_fourier_coeff_mse_multiband_vs_clean_gt"]
                    ),
                    "noisy_fourier_coeff_mse_multiband_vs_clean_gt": list(
                        metrics["noisy_fourier_coeff_mse_multiband_vs_clean_gt"]
                    ),
                },
                "spectral_band_labels": spectral_band_labels,
                "spectral_band_centers": [float(v) for v in rsd.spectral_band_centers],
                "clean_spectral_band_error_mean": clean_spectral_band_error_mean,
                "noisy_spectral_band_error_mean": noisy_spectral_band_error_mean,
                "clean_spectral_band_error_vs_clean_gt_mean": clean_spectral_band_error_vs_clean_gt_mean,
                "noisy_spectral_band_error_vs_clean_gt_mean": noisy_spectral_band_error_vs_clean_gt_mean,
                "clean_fourier_coeff_mse_band_vs_clean_gt_mean": clean_fourier_coeff_mse_band_vs_clean_gt_mean,
                "noisy_fourier_coeff_mse_band_vs_clean_gt_mean": noisy_fourier_coeff_mse_band_vs_clean_gt_mean,
                "fraction_multiband_gap_bootstrap_noisy_minus_clean": fraction_multiband_gap_bootstrap,
                "coeff_mse_multiband_gap_bootstrap_noisy_minus_clean": coeff_mse_multiband_gap_bootstrap,
                "fraction_band_gap_bootstrap_noisy_minus_clean": fraction_band_gap_bootstrap,
                "coeff_mse_band_gap_bootstrap_noisy_minus_clean": coeff_mse_band_gap_bootstrap,
            },
        },
        "_resolved_device": str(model_clean.device),
        "_data": {
            "source": str(data_source),
            "dt": float(dt),
            "n_snapshots": int(config.n_snapshots),
            "n_train": int(n_train_total),
            "n_test": int(len(test_cases)),
            "metadata": dict(data_metadata or {}),
        },
    }
