#!/usr/bin/env python3
"""Entry script for one Navier-Stokes run.

Usage:
    python3 runs/run_navier_stokes.py configs/navier_stokes.yaml tfno 1 --device auto --loss combined --basis fourier
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.navier_stokes import NSConfig
from data.navier_stokes.external import (
    ExternalNavierStokesDataConfig,
    external_data_config_from_yaml,
    load_navier_stokes_trajectory_data,
)
from eval.metrics import build_metric_vs_l2 as _build_metric_vs_l2
from eval.metrics import build_paired_bootstrap_gap as _build_paired_bootstrap_gap
from eval.metrics import safe_mean as _safe_mean
from eval.navier_stokes import block_future_step_indices as _block_future_step_indices
from eval.navier_stokes import extract_panel_frames as _extract_panel_frames
from eval.navier_stokes import future_block_rel_l2 as _future_block_rel_l2
from models.navier_stokes import LOSS_CHOICES, build_model, normalize_loss_name, rollout_2d
from runs.helpers.common import load_best_checkpoint_for_eval as _load_best_checkpoint_for_eval
from runs.helpers.common import move_model_device as _move_model_device
from runs.helpers.indexed_datasets import (
    NavierStokesIndexedPairDataset as _NavierStokesIndexedPairDataset,
)
from runs.helpers.indexed_datasets import resolve_dataloader_num_workers as _resolve_dataloader_num_workers
from runs.helpers.navier_stokes_training import build_noisy_trajectories as _build_noisy_trajectories
from runs.helpers.navier_stokes_training import noisy_reference_field as _noisy_reference_field
from runs.helpers.navier_stokes_training import noisy_reference_trajectory as _noisy_reference_trajectory
from runs.helpers.navier_stokes_reporting import save_fit_visualizations as _save_fit_visualizations
from runs.helpers.navier_stokes_reporting import save_standard_figures as _save_standard_figures
from runs.helpers.navier_stokes_reporting import (
    save_trajectory_visualizations as _save_trajectory_visualizations,
)
from runs.helpers.temporal import resolve_temporal_training_config as _resolve_temporal_training_config
from runs.helpers.temporal import window_start_indices as _window_start_indices
from runs.helpers.temporal import window_target_start as _window_target_start
from utils.config import load_yaml_config
from utils.diagnostics import BASIS_CHOICES, NavierStokesRSDAnalyzer, normalize_basis_name
from utils.io import build_run_dirs, load_checkpoint, save_checkpoint, save_json
from utils.progress import progress_iter
from utils.torch_runtime import DEVICE_CHOICES

from runs.helpers.navier_stokes_seed import run_single_seed as _run_single_seed_impl


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
    return _run_single_seed_impl(
        config=config,
        method=method,
        seed=seed,
        device=device,
        loss=loss,
        basis=basis,
        operator_config=operator_config,
        baseline_config=baseline_config,
        external_data_cfg=external_data_cfg,
        eval_pair_index=eval_pair_index,
        test_case_index=test_case_index,
        test_step_index=test_step_index,
        trajectory_case_indices=trajectory_case_indices,
        trajectory_step_indices=trajectory_step_indices,
        show_data_progress=show_data_progress,
        show_training_progress=show_training_progress,
        show_eval_progress=show_eval_progress,
        spectral_band_count=spectral_band_count,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_epochs=checkpoint_every_epochs,
        resume_clean_state=resume_clean_state,
        resume_noisy_state=resume_noisy_state,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Navier-Stokes experiment with YAML config, method, and seed."
    )
    parser.add_argument("config_yaml", type=str, help="Path to YAML config file")
    parser.add_argument(
        "method",
        type=str,
        help="Model method (tfno, itfno, uno, wno, rno, conv, swin, attn_unet).",
    )
    parser.add_argument("seed", type=int, help="Random seed number")
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default=None,
        help="Compute device: auto, cpu, cuda, or mps (defaults to training.device in YAML, else auto).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=LOSS_CHOICES,
        default=None,
        help="Training objective (defaults to training.loss in YAML, else combined).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        choices=BASIS_CHOICES,
        default=None,
        help="Residual projection basis for HFV/LFV (defaults to rsd.basis in YAML, else fourier).",
    )
    parser.add_argument(
        "--resume-clean-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for resuming clean-model training state.",
    )
    parser.add_argument(
        "--resume-noisy-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for resuming noisy-model training state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_config = load_yaml_config(args.config_yaml)
    config = NSConfig.from_yaml(raw_config)
    external_data_cfg = external_data_config_from_yaml(raw_config)

    paths = raw_config.get("paths", {})
    output_root = paths.get("output_dir", "output")
    checkpoint_root = paths.get("checkpoint_dir", "checkpoints")

    experiment = raw_config.get("experiment", {})
    experiment_name = experiment.get("name", "navier_stokes")
    training = raw_config.get("training", {})
    operator_config = training.get("neural_operator", {})
    baseline_config = training.get("baseline_models", {})
    rsd_cfg = raw_config.get("rsd", {})
    requested_device = args.device if args.device is not None else str(training.get("device", "auto"))
    requested_loss = normalize_loss_name(args.loss if args.loss is not None else str(training.get("loss", "combined")))
    requested_basis = normalize_basis_name(args.basis if args.basis is not None else str(rsd_cfg.get("basis", "fourier")))
    requested_spectral_band_count = max(3, int(rsd_cfg.get("spectral_band_count", 8)))

    run_out_dir, run_ckpt_dir = build_run_dirs(
        output_root,
        checkpoint_root,
        problem_name=experiment_name,
        method=args.method,
        loss=requested_loss,
        basis=requested_basis,
        seed=args.seed,
    )

    artifacts = raw_config.get("artifacts", {})
    eval_pair_index = int(artifacts.get("eval_pair_index", 0))
    test_case_index = int(artifacts.get("test_case_index", 0))
    test_step_index = int(artifacts.get("test_step_index", 0))
    trajectory_viz = artifacts.get("trajectory_visualization", {})
    if isinstance(trajectory_viz, dict):
        trajectory_case_indices = trajectory_viz.get("instance_indices", [0, 1])
        trajectory_step_indices = trajectory_viz.get("step_indices", [0, 4, 8, 12, 16, 19])
    else:
        trajectory_case_indices = [0, 1]
        trajectory_step_indices = [0, 4, 8, 12, 16, 19]
    progress = raw_config.get("progress", {})
    progress_enabled = bool(progress.get("enabled", False))
    data_progress = bool(progress.get("data_generation", progress_enabled))
    training_progress = bool(progress.get("training", progress_enabled))
    eval_progress = bool(progress.get("evaluation", progress_enabled))
    resume_clean_state = None
    resume_noisy_state = None
    if args.resume_clean_checkpoint:
        loaded_clean = load_checkpoint(args.resume_clean_checkpoint)
        if isinstance(loaded_clean, dict):
            resume_clean_state = loaded_clean.get("training_state", loaded_clean)
    if args.resume_noisy_checkpoint:
        loaded_noisy = load_checkpoint(args.resume_noisy_checkpoint)
        if isinstance(loaded_noisy, dict):
            resume_noisy_state = loaded_noisy.get("training_state", loaded_noisy)

    results = run_single_seed(
        config,
        args.method,
        args.seed,
        device=requested_device,
        loss=requested_loss,
        basis=requested_basis,
        operator_config=operator_config,
        baseline_config=baseline_config,
        external_data_cfg=external_data_cfg,
        eval_pair_index=eval_pair_index,
        test_case_index=test_case_index,
        test_step_index=test_step_index,
        trajectory_case_indices=trajectory_case_indices,
        trajectory_step_indices=trajectory_step_indices,
        show_data_progress=data_progress,
        show_training_progress=training_progress,
        show_eval_progress=eval_progress,
        spectral_band_count=requested_spectral_band_count,
        checkpoint_dir=run_ckpt_dir,
        checkpoint_every_epochs=config.train_checkpoint_every_epochs,
        resume_clean_state=resume_clean_state,
        resume_noisy_state=resume_noisy_state,
    )

    viz_payload = results.pop("_viz")
    resolved_device = results.pop("_resolved_device")
    data_source = results.pop("_data_source")
    data_metadata = results.pop("_data_metadata")
    dt = float(results.pop("_dt"))
    n_snapshots = int(results.pop("_n_snapshots"))

    summary = {
        "experiment": experiment_name,
        "config_yaml": str(Path(args.config_yaml).resolve()),
        "method": args.method,
        "seed": args.seed,
        "device_requested": requested_device,
        "device_resolved": resolved_device,
        "loss": requested_loss,
        "basis": requested_basis,
        "spectral_band_count": requested_spectral_band_count,
        "data_source": data_source,
        "data_metadata": data_metadata,
        "dt": dt,
        "n_snapshots": n_snapshots,
        "metrics": results,
        "viz_indices": viz_payload["indices"],
    }
    save_json(run_out_dir / "results.json", summary)

    if artifacts.get("save_figures", True):
        _save_standard_figures(
            results=results,
            viz_payload=viz_payload,
            run_out_dir=run_out_dir,
            method=args.method,
            requested_loss=requested_loss,
            requested_basis=requested_basis,
            seed=args.seed,
        )

    if artifacts.get("save_fit_visualizations", True):
        _save_fit_visualizations(
            artifacts=artifacts,
            viz_payload=viz_payload,
            run_out_dir=run_out_dir,
        )

    if artifacts.get("save_trajectory_visualizations", True):
        _save_trajectory_visualizations(
            viz_payload=viz_payload,
            run_out_dir=run_out_dir,
            config=config,
            seed=args.seed,
        )

    print("Run complete")
    print(f"Device: requested={requested_device} resolved={resolved_device}")
    print(f"Loss: {requested_loss}")
    print(f"Basis: {requested_basis}")
    print(f"Results: {run_out_dir / 'results.json'}")
    print(f"Checkpoints: {run_ckpt_dir}")


if __name__ == "__main__":
    main()
