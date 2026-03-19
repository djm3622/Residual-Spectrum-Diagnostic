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
        help="Model method (tfno, itfno, uno, rno, conv, swin, attn_unet).",
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
        from utils.plotting import (
            save_band_profile_plot,
            save_clean_noisy_metric_bar,
            save_clean_noisy_summary_plot,
            save_dual_band_gap_bootstrap_plot,
            save_metric_vs_l2_grid,
            save_spectral_band_error_plot,
        )

        save_clean_noisy_summary_plot(
            results,
            title=(
                f"Navier-Stokes | method={args.method} | loss={requested_loss} "
                f"| basis={requested_basis} | seed={args.seed}"
            ),
            output_path=run_out_dir / "summary.png",
        )

        save_clean_noisy_metric_bar(
            results["clean_pde_residual_st_rms"],
            results["noisy_pde_residual_st_rms"],
            metric_label="Space-time PDE residual RMS",
            output_path=run_out_dir / "pde_residual_space_time.png",
            title="PDE residual norm (space-time)",
        )
        save_clean_noisy_metric_bar(
            results["clean_boundary_error"],
            results["noisy_boundary_error"],
            metric_label="Boundary-condition error (periodic)",
            output_path=run_out_dir / "boundary_condition_error.png",
            title="Boundary-condition error",
        )
        save_clean_noisy_metric_bar(
            results["clean_spectral_multiband_error"],
            results["noisy_spectral_multiband_error"],
            metric_label="Multi-band spectral error",
            output_path=run_out_dir / "spectral_multiband_error.png",
            title="Multi-band spectral error",
        )
        save_clean_noisy_metric_bar(
            results["clean_fourier_coeff_mse_multiband_vs_clean_gt"],
            results["noisy_fourier_coeff_mse_multiband_vs_clean_gt"],
            metric_label="Multi-band Fourier coeff-MSE",
            output_path=run_out_dir / "fourier_coeff_mse_multiband.png",
            title="Multi-band Fourier coeff-MSE (vs clean GT)",
        )

        diagnostics_viz = viz_payload.get("diagnostics", {})
        save_spectral_band_error_plot(
            clean_band_error=diagnostics_viz.get("clean_spectral_band_error_mean", []),
            noisy_band_error=diagnostics_viz.get("noisy_spectral_band_error_mean", []),
            band_labels=diagnostics_viz.get("spectral_band_labels", []),
            band_centers=diagnostics_viz.get("spectral_band_centers", []),
            output_path=run_out_dir / "spectral_band_error_profile.png",
            title="Per-band spectral error profile",
        )
        save_band_profile_plot(
            clean_band_values=diagnostics_viz.get("clean_fourier_coeff_mse_band_vs_clean_gt_mean", []),
            noisy_band_values=diagnostics_viz.get("noisy_fourier_coeff_mse_band_vs_clean_gt_mean", []),
            band_labels=diagnostics_viz.get("spectral_band_labels", []),
            band_centers=diagnostics_viz.get("spectral_band_centers", []),
            output_path=run_out_dir / "fourier_coeff_mse_band_profile.png",
            title="Per-band Fourier coeff-MSE vs clean GT",
            y_label="Fourier coefficient MSE",
        )
        save_dual_band_gap_bootstrap_plot(
            band_labels=diagnostics_viz.get("spectral_band_labels", []),
            band_centers=diagnostics_viz.get("spectral_band_centers", []),
            fraction_gap_bootstrap=diagnostics_viz.get("fraction_band_gap_bootstrap_noisy_minus_clean", []),
            coeff_mse_gap_bootstrap=diagnostics_viz.get("coeff_mse_band_gap_bootstrap_noisy_minus_clean", []),
            output_path=run_out_dir / "spectral_gap_bootstrap_ci.png",
            title="Bootstrap CI: spectral gap (noisy-clean, vs clean GT)",
        )

        diagnostic_series = diagnostics_viz.get("series", {})
        save_metric_vs_l2_grid(
            l2_clean=diagnostic_series.get("clean_l2", []),
            l2_noisy=diagnostic_series.get("noisy_l2", []),
            metric_series={
                "HFV": {
                    "clean": diagnostic_series.get("clean_hfv", []),
                    "noisy": diagnostic_series.get("noisy_hfv", []),
                },
                "LFV": {
                    "clean": diagnostic_series.get("clean_lfv", []),
                    "noisy": diagnostic_series.get("noisy_lfv", []),
                },
                "PDE Residual RMS": {
                    "clean": diagnostic_series.get("clean_pde_residual_st_rms", []),
                    "noisy": diagnostic_series.get("noisy_pde_residual_st_rms", []),
                },
                "Boundary Error": {
                    "clean": diagnostic_series.get("clean_boundary_error", []),
                    "noisy": diagnostic_series.get("noisy_boundary_error", []),
                },
                "Spectral Multi-band Error": {
                    "clean": diagnostic_series.get("clean_spectral_multiband_error", []),
                    "noisy": diagnostic_series.get("noisy_spectral_multiband_error", []),
                },
                "Fourier Coeff MSE": {
                    "clean": diagnostic_series.get("clean_fourier_coeff_mse_multiband_vs_clean_gt", []),
                    "noisy": diagnostic_series.get("noisy_fourier_coeff_mse_multiband_vs_clean_gt", []),
                },
            },
            output_path=run_out_dir / "metrics_vs_l2.png",
            title="Metric effectiveness vs L2",
        )

    if artifacts.get("save_fit_visualizations", True):
        from utils.plotting import save_scalar_fit_panel

        fit_dir = run_out_dir / "fit_quality"
        fit_viz = artifacts.get("fit_visualization", {})
        input_viz = fit_viz.get("input", {}) if isinstance(fit_viz, dict) else {}
        output_viz = fit_viz.get("output", {}) if isinstance(fit_viz, dict) else {}
        input_cmap = str(input_viz.get("cmap", "cividis"))
        input_border_color = str(input_viz.get("border_color", "#2A9D8F"))
        input_border_width = float(input_viz.get("border_width", 2.0))
        output_cmap = str(output_viz.get("cmap", "RdBu_r"))

        save_scalar_fit_panel(
            viz_payload["eval"]["input"],
            viz_payload["eval"]["target"],
            viz_payload["eval"]["pred_clean"],
            output_path=fit_dir / "eval_clean.png",
            title="Eval split | Clean model",
            cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
            target_field_noisy=viz_payload["eval"]["target_noisy"],
            model_label="Clean model",
        )
        save_scalar_fit_panel(
            viz_payload["eval"]["input"],
            viz_payload["eval"]["target"],
            viz_payload["eval"]["pred_noisy"],
            output_path=fit_dir / "eval_noisy.png",
            title="Eval split | Noisy model",
            cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
            target_field_noisy=viz_payload["eval"]["target_noisy"],
            model_label="Noisy model",
        )
        save_scalar_fit_panel(
            viz_payload["test"]["input"],
            viz_payload["test"]["target"],
            viz_payload["test"]["pred_clean"],
            output_path=fit_dir / "test_clean.png",
            title="Test split | Clean model",
            cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
            target_field_noisy=viz_payload["test"]["target_noisy"],
            model_label="Clean model",
        )
        save_scalar_fit_panel(
            viz_payload["test"]["input"],
            viz_payload["test"]["target"],
            viz_payload["test"]["pred_noisy"],
            output_path=fit_dir / "test_noisy.png",
            title="Test split | Noisy model",
            cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
            target_field_noisy=viz_payload["test"]["target_noisy"],
            model_label="Noisy model",
        )

    if artifacts.get("save_trajectory_visualizations", True):
        from utils.plotting import save_trajectory_error_rows, save_trajectory_field_rows

        fit_dir = run_out_dir / "fit_quality"
        trajectory_payload = viz_payload.get("trajectory", {})
        rows = trajectory_payload.get("rows", [])
        case_indices = trajectory_payload.get("case_indices", [])
        step_indices = trajectory_payload.get("step_indices", [])

        by_case: Dict[int, Dict[str, np.ndarray]] = {}
        for row in rows:
            case_idx = int(row["case_index"])
            case_bucket = by_case.setdefault(case_idx, {})
            case_bucket["omega_true"] = row["omega_true"]
            case_bucket[f"omega_{row['model']}"] = row["omega_pred"]

        ordered_cases = [int(idx) for idx in case_indices if int(idx) in by_case]
        if not ordered_cases:
            ordered_cases = sorted(by_case.keys())

        field_rows = []
        error_rows = []
        for case_idx in ordered_cases:
            bucket = by_case[case_idx]
            omega_true = bucket.get("omega_true")
            omega_clean = bucket.get("omega_clean")
            omega_noisy = bucket.get("omega_noisy")
            if omega_true is None or omega_clean is None or omega_noisy is None:
                continue

            omega_truth_noisy = _noisy_reference_trajectory(
                omega_true,
                config,
                rng_seed=args.seed * 1_000_000 + case_idx * 10_000 + 421,
            )

            field_rows.append({"label": f"case {case_idx} | Clean GT", "traj": omega_true})
            field_rows.append({"label": f"case {case_idx} | Noisy GT", "traj": omega_truth_noisy})
            field_rows.append({"label": f"case {case_idx} | Clean Pred", "traj": omega_clean})
            field_rows.append({"label": f"case {case_idx} | Noisy Pred", "traj": omega_noisy})

            error_rows.append({"label": f"case {case_idx} | Clean Pred vs Clean GT", "pred": omega_clean, "target": omega_true})
            error_rows.append({"label": f"case {case_idx} | Noisy Pred vs Clean GT", "pred": omega_noisy, "target": omega_true})
            error_rows.append({"label": f"case {case_idx} | Clean Pred vs Noisy GT", "pred": omega_clean, "target": omega_truth_noisy})
            error_rows.append({"label": f"case {case_idx} | Noisy Pred vs Noisy GT", "pred": omega_noisy, "target": omega_truth_noisy})

        save_trajectory_field_rows(
            field_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_omega_fields.png",
            title="Trajectory snapshots | Vorticity",
            cmap="RdBu_r",
        )
        save_trajectory_error_rows(
            error_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_omega_error.png",
            title="Trajectory absolute error snapshots | Vorticity",
            cmap="magma",
        )

    print("Run complete")
    print(f"Device: requested={requested_device} resolved={resolved_device}")
    print(f"Loss: {requested_loss}")
    print(f"Basis: {requested_basis}")
    print(f"Results: {run_out_dir / 'results.json'}")
    print(f"Checkpoints: {run_ckpt_dir}")


if __name__ == "__main__":
    main()
