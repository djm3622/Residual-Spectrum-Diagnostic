"""Single-seed training/evaluation orchestration for Navier-Stokes runs."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from data.navier_stokes import NSConfig
from data.navier_stokes.external import (
    ExternalNavierStokesDataConfig,
    load_navier_stokes_trajectory_data,
)
from eval.metrics import build_metric_vs_l2 as _build_metric_vs_l2
from eval.metrics import build_paired_bootstrap_gap as _build_paired_bootstrap_gap
from eval.metrics import safe_mean as _safe_mean
from eval.navier_stokes import block_future_step_indices as _block_future_step_indices
from eval.navier_stokes import extract_panel_frames as _extract_panel_frames
from eval.navier_stokes import future_block_rel_l2 as _future_block_rel_l2
from models.navier_stokes import build_model, rollout_2d
from runs.helpers.common import load_best_checkpoint_for_eval as _load_best_checkpoint_for_eval
from runs.helpers.common import move_model_device as _move_model_device
from runs.helpers.indexed_datasets import (
    NavierStokesIndexedPairDataset as _NavierStokesIndexedPairDataset,
)
from runs.helpers.indexed_datasets import resolve_dataloader_num_workers as _resolve_dataloader_num_workers
from runs.helpers.navier_stokes_training import build_noisy_trajectories as _build_noisy_trajectories
from runs.helpers.navier_stokes_training import noisy_reference_field as _noisy_reference_field
from runs.helpers.navier_stokes_training import noisy_reference_trajectory as _noisy_reference_trajectory
from runs.helpers.temporal import resolve_temporal_training_config as _resolve_temporal_training_config
from runs.helpers.temporal import window_start_indices as _window_start_indices
from runs.helpers.temporal import window_target_start as _window_target_start
from utils.diagnostics import NavierStokesRSDAnalyzer
from utils.io import save_checkpoint
from utils.progress import progress_iter
def run_single_seed(
    config: NSConfig,
    method: str,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    basis: str = "fourier",
    operator_config: Mapping[str, Any] | None = None,
    baseline_config: Mapping[str, Any] | None = None,
    external_data_cfg: ExternalNavierStokesDataConfig | None = None,
    eval_pair_index: int = 0,
    test_case_index: int = 0,
    test_step_index: int = 0,
    trajectory_case_indices: List[int] | None = None,
    trajectory_step_indices: List[int] | None = None,
    show_data_progress: bool = False,
    show_training_progress: bool = False,
    show_eval_progress: bool = False,
    spectral_band_count: int = 8,
    checkpoint_dir: Path | None = None,
    checkpoint_every_epochs: int = 20,
    resume_clean_state: Dict[str, Any] | None = None,
    resume_noisy_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)
    rsd = NavierStokesRSDAnalyzer(config, basis=basis, spectral_band_count=spectral_band_count)
    resolved_external_cfg = external_data_cfg or ExternalNavierStokesDataConfig(source="generated")
    data_bundle = load_navier_stokes_trajectory_data(
        config,
        resolved_external_cfg,
        seed=seed,
        show_data_progress=show_data_progress,
    )
    train_trajectories: List[np.ndarray] = data_bundle.train_trajectories
    test_trajectories: List[np.ndarray] = data_bundle.test_trajectories
    dt = float(data_bundle.dt)
    n_snapshots = int(data_bundle.n_snapshots)
    temporal_cfg = _resolve_temporal_training_config(method, operator_config)
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
    n_train_traj_total = len(train_trajectories)
    n_val_traj = 0
    if n_train_traj_total > 1 and val_fraction > 0.0:
        n_val_traj = int(round(n_train_traj_total * val_fraction))
        n_val_traj = max(1, min(n_train_traj_total - 1, n_val_traj))

    split_rng = np.random.default_rng(seed * 1000 + 707)
    split_perm = list(split_rng.permutation(n_train_traj_total))
    val_idx_set = set(split_perm[:n_val_traj])
    fit_trajectories = [traj for idx, traj in enumerate(train_trajectories) if idx not in val_idx_set]
    val_trajectories = [traj for idx, traj in enumerate(train_trajectories) if idx in val_idx_set]
    if not fit_trajectories:
        fit_trajectories = train_trajectories
        val_trajectories = []
    test_cases = [
        {"omega0": traj[0], "omega_true": traj}
        for traj in test_trajectories
    ]

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
        if temporal_enabled and temporal_target_mode == "next_block":
            requested_steps = _block_future_step_indices(n_snapshots, temporal_window, max_points=6)
        else:
            requested_steps = [0, n_snapshots // 4, n_snapshots // 2, (3 * n_snapshots) // 4, n_snapshots - 1]
    valid_trajectory_steps = sorted(
        {
            int(np.clip(step, 0, n_snapshots - 1))
            for step in requested_steps
        }
    )
    if not valid_trajectory_steps:
        valid_trajectory_steps = [0, n_snapshots - 1]

    train_dataset_clean = _NavierStokesIndexedPairDataset(
        fit_trajectories,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    val_dataset_clean = _NavierStokesIndexedPairDataset(
        val_trajectories,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )

    if len(train_dataset_clean) == 0:
        raise ValueError(
            "No training pairs were generated. "
            f"temporal_enabled={temporal_enabled}, temporal_window={temporal_window}, "
            f"n_snapshots={n_snapshots}"
        )
    eval_pair_index = int(np.clip(eval_pair_index, 0, len(train_dataset_clean) - 1))
    eval_input_reference, eval_target_clean_reference = train_dataset_clean.get_pair_arrays(eval_pair_index)

    model_clean = build_model(
        method,
        config.nx,
        config.ny,
        seed=seed,
        device=device,
        loss=loss,
        operator_config=operator_config,
        baseline_config=baseline_config,
    )
    def _clean_checkpoint_callback(epoch: int, val_loss: float, training_state: Dict[str, Any]) -> None:
        _save_checkpoint_event("clean", epoch, val_loss, training_state)

    rollout_trajectories_clean = (
        fit_trajectories
        if is_rno_method or (config.train_rollout_weight > 0.0 and config.train_rollout_horizon > 1)
        else None
    )

    model_clean.train(
        inputs=[],
        targets=[],
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        weight_decay=config.train_weight_decay,
        use_one_cycle_lr=config.train_use_one_cycle_lr,
        one_cycle_pct_start=config.train_one_cycle_pct_start,
        one_cycle_div_factor=config.train_one_cycle_div_factor,
        one_cycle_final_div_factor=config.train_one_cycle_final_div_factor,
        trajectory=rollout_trajectories_clean,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        val_inputs=[],
        val_targets=[],
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fit_trajectories_noisy = _build_noisy_trajectories(
        fit_trajectories,
        config,
        show_progress=show_data_progress,
        progress_desc="Build noisy train trajectories",
    )
    val_trajectories_noisy = _build_noisy_trajectories(
        val_trajectories,
        config,
        show_progress=False,
        progress_desc="Build noisy validation trajectories",
    )
    train_dataset_noisy = _NavierStokesIndexedPairDataset(
        fit_trajectories_noisy,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    val_dataset_noisy = _NavierStokesIndexedPairDataset(
        val_trajectories_noisy,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    if len(train_dataset_noisy) == 0:
        raise ValueError(
            "No noisy training pairs were generated. "
            f"temporal_enabled={temporal_enabled}, temporal_window={temporal_window}, "
            f"n_snapshots={n_snapshots}"
        )
    if eval_pair_index >= len(train_dataset_noisy):
        raise ValueError(
            "Noisy training pair count differs from clean training pair count "
            f"(clean_eval_index={eval_pair_index}, noisy_pairs={len(train_dataset_noisy)})."
        )
    _, eval_target_noisy_reference = train_dataset_noisy.get_pair_arrays(eval_pair_index)
    del fit_trajectories, val_trajectories, train_trajectories
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
        operator_config=operator_config,
        baseline_config=baseline_config,
    )
    def _noisy_checkpoint_callback(epoch: int, val_loss: float, training_state: Dict[str, Any]) -> None:
        _save_checkpoint_event("noisy", epoch, val_loss, training_state)

    rollout_trajectories_noisy = (
        fit_trajectories_noisy
        if is_rno_method or (config.train_rollout_weight > 0.0 and config.train_rollout_horizon > 1)
        else None
    )

    model_noisy.train(
        inputs=[],
        targets=[],
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        weight_decay=config.train_weight_decay,
        use_one_cycle_lr=config.train_use_one_cycle_lr,
        one_cycle_pct_start=config.train_one_cycle_pct_start,
        one_cycle_div_factor=config.train_one_cycle_div_factor,
        one_cycle_final_div_factor=config.train_one_cycle_final_div_factor,
        trajectory=rollout_trajectories_noisy,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        val_inputs=[],
        val_targets=[],
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
    del train_dataset_noisy, val_dataset_noisy, fit_trajectories_noisy, val_trajectories_noisy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _load_best_checkpoint_for_eval(model_clean, resolved_checkpoint_dir, "clean")
    _load_best_checkpoint_for_eval(model_noisy, resolved_checkpoint_dir, "noisy")

    metrics = {
        "clean_l2": [],
        "noisy_l2": [],
        "clean_l2_future_block": [],
        "noisy_l2_future_block": [],
        "clean_hfv": [],
        "noisy_hfv": [],
        "clean_lfv": [],
        "noisy_lfv": [],
        "clean_pde_residual_st_rms": [],
        "noisy_pde_residual_st_rms": [],
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
    future_block_horizon = temporal_window if (temporal_enabled and temporal_target_mode == "next_block") else 1

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
        # In temporal mode, seed with the first observed window then forecast autonomously.
        if rollout_context is not None:
            omega_context = np.asarray(case["omega_true"][:rollout_context], dtype=np.float32)
        else:
            omega_context = None
        omega_clean = rollout_2d(model_clean, case["omega0"], n_snapshots, context=omega_context)
        omega_noisy = rollout_2d(model_noisy, case["omega0"], n_snapshots, context=omega_context)

        if case_idx in trajectory_case_set:
            trajectory_rows.append(
                {
                    "case_index": case_idx,
                    "model": "clean",
                    "omega_pred": omega_clean,
                    "omega_true": case["omega_true"],
                }
            )
            trajectory_rows.append(
                {
                    "case_index": case_idx,
                    "model": "noisy",
                    "omega_pred": omega_noisy,
                    "omega_true": case["omega_true"],
                }
            )

        clean_stats = rsd.compute_metrics(omega_clean, case["omega_true"], dt)
        noisy_stats = rsd.compute_metrics(omega_noisy, case["omega_true"], dt)

        metrics["clean_l2"].append(clean_stats["l2_error"])
        metrics["noisy_l2"].append(noisy_stats["l2_error"])
        metrics["clean_l2_future_block"].append(
            _future_block_rel_l2(omega_clean, case["omega_true"], future_block_horizon)
        )
        metrics["noisy_l2_future_block"].append(
            _future_block_rel_l2(omega_noisy, case["omega_true"], future_block_horizon)
        )
        metrics["clean_hfv"].append(clean_stats["hfv"])
        metrics["noisy_hfv"].append(noisy_stats["hfv"])
        metrics["clean_lfv"].append(clean_stats["lfv"])
        metrics["noisy_lfv"].append(noisy_stats["lfv"])
        metrics["clean_pde_residual_st_rms"].append(clean_stats["pde_residual_st_rms"])
        metrics["noisy_pde_residual_st_rms"].append(noisy_stats["pde_residual_st_rms"])
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
        eval_input_window = np.asarray(eval_input_reference, dtype=np.float32)
        eval_target_window = np.asarray(eval_target_clean_reference, dtype=np.float32)
        eval_target_window_noisy = np.asarray(eval_target_noisy_reference, dtype=np.float32)
        eval_pred_window_clean = np.asarray(model_clean.predict_window(eval_input_window), dtype=np.float32)
        eval_pred_window_noisy = np.asarray(model_noisy.predict_window(eval_input_window), dtype=np.float32)
        eval_input, eval_target, eval_pred_clean = _extract_panel_frames(
            eval_input_window,
            eval_target_window,
            eval_pred_window_clean,
            temporal_target_mode,
        )
        _, eval_target_noisy, _ = _extract_panel_frames(
            eval_input_window,
            eval_target_window_noisy,
            eval_target_window_noisy,
            temporal_target_mode,
        )
        _, _, eval_pred_noisy = _extract_panel_frames(
            eval_input_window,
            eval_target_window,
            eval_pred_window_noisy,
            temporal_target_mode,
        )

        test_start_candidates = list(
            _window_start_indices(test_case["omega_true"].shape[0], temporal_window, temporal_target_mode)
        )
        if not test_start_candidates:
            raise ValueError(
                "Temporal evaluation requested but test trajectory is too short for configured window: "
                f"n_steps={test_case['omega_true'].shape[0]}, window={temporal_window}, "
                f"target_mode={temporal_target_mode}"
            )
        max_test_step = max(test_start_candidates)
        test_step_index = int(np.clip(test_step_index, 0, max_test_step))
        target_start = _window_target_start(test_step_index, temporal_window, temporal_target_mode)
        test_input_window = np.asarray(
            test_case["omega_true"][test_step_index : test_step_index + temporal_window],
            dtype=np.float32,
        )
        test_target_window = np.asarray(
            test_case["omega_true"][target_start : target_start + temporal_window],
            dtype=np.float32,
        )
        test_target_window_noisy = _noisy_reference_trajectory(
            test_target_window,
            config,
            rng_seed=seed * 1_000_000 + test_case_index * 10_000 + target_start,
        )
        test_pred_window_clean = np.asarray(model_clean.predict_window(test_input_window), dtype=np.float32)
        test_pred_window_noisy = np.asarray(model_noisy.predict_window(test_input_window), dtype=np.float32)
        test_input, test_target, test_pred_clean = _extract_panel_frames(
            test_input_window,
            test_target_window,
            test_pred_window_clean,
            temporal_target_mode,
        )
        _, test_target_noisy, _ = _extract_panel_frames(
            test_input_window,
            test_target_window_noisy,
            test_target_window_noisy,
            temporal_target_mode,
        )
        _, _, test_pred_noisy = _extract_panel_frames(
            test_input_window,
            test_target_window,
            test_pred_window_noisy,
            temporal_target_mode,
        )
    else:
        max_test_step = test_case["omega_true"].shape[0] - 2
        test_step_index = int(np.clip(test_step_index, 0, max_test_step))

        eval_input = np.asarray(eval_input_reference, dtype=np.float32)
        eval_target = np.asarray(eval_target_clean_reference, dtype=np.float32)
        eval_target_noisy = np.asarray(eval_target_noisy_reference, dtype=np.float32)
        eval_pred_clean = model_clean.forward(eval_input)
        eval_pred_noisy = model_noisy.forward(eval_input)

        test_input = test_case["omega_true"][test_step_index]
        test_target = test_case["omega_true"][test_step_index + 1]
        test_target_noisy = _noisy_reference_field(
            test_target,
            config,
            rng_seed=seed * 1_000_000 + test_case_index * 10_000 + test_step_index + 1,
        )
        test_pred_clean = model_clean.forward(test_input)
        test_pred_noisy = model_noisy.forward(test_input)

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
                "input": eval_input,
                "target": eval_target,
                "target_noisy": eval_target_noisy,
                "pred_clean": eval_pred_clean,
                "pred_noisy": eval_pred_noisy,
            },
            "test": {
                "input": test_input,
                "target": test_target,
                "target_noisy": test_target_noisy,
                "pred_clean": test_pred_clean,
                "pred_noisy": test_pred_noisy,
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
        "_data_source": data_bundle.source,
        "_data_metadata": data_bundle.metadata,
        "_dt": dt,
        "_n_snapshots": n_snapshots,
    }
