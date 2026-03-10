#!/usr/bin/env python3
"""Entry script for one Navier-Stokes run.

Usage:
    python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1 --device auto --loss combined --basis fourier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.navier_stokes import NSConfig, NavierStokes2D
from models.navier_stokes import LOSS_CHOICES, build_model, normalize_loss_name, rollout_2d
from utils.config import load_yaml_config
from utils.diagnostics import BASIS_CHOICES, NavierStokesRSDAnalyzer, normalize_basis_name
from utils.io import build_run_dirs, save_checkpoint, save_json
from utils.noise import add_hf_noise_2d
from utils.progress import progress_iter
from utils.torch_runtime import DEVICE_CHOICES


def _safe_mean(values: List[float]) -> float:
    """Mean over finite values, preserving NaN only when all values are non-finite."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def run_single_seed(
    config: NSConfig,
    method: str,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    basis: str = "fourier",
    eval_pair_index: int = 0,
    test_case_index: int = 0,
    test_step_index: int = 0,
    trajectory_case_indices: List[int] | None = None,
    trajectory_step_indices: List[int] | None = None,
    show_data_progress: bool = False,
    show_training_progress: bool = False,
    show_eval_progress: bool = False,
) -> Dict[str, float]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)

    solver = NavierStokes2D(config)
    rsd = NavierStokesRSDAnalyzer(config, basis=basis)

    train_trajectories: List[np.ndarray] = []
    for idx in progress_iter(
        range(config.n_train_trajectories),
        enabled=show_data_progress,
        desc="Data gen (train)",
        total=config.n_train_trajectories,
    ):
        omega0 = solver.sample_initial_condition(seed=seed * 1000 + idx, index=idx)
        t_save, omega_traj = solver.solve(omega0, config.t_final, config.n_snapshots)
        train_trajectories.append(omega_traj)
    dt = float(t_save[1] - t_save[0])

    test_cases = []
    for idx in progress_iter(
        range(config.n_test_trajectories),
        enabled=show_data_progress,
        desc="Data gen (test)",
        total=config.n_test_trajectories,
    ):
        omega0 = solver.sample_initial_condition(seed=seed * 1000 + 500 + idx, index=10_000 + idx)
        _, omega_true = solver.solve(omega0, config.t_final, config.n_snapshots)
        test_cases.append({"omega0": omega0, "omega_true": omega_true})

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

    inputs = []
    targets_clean = []
    targets_noisy = []

    for trajectory in progress_iter(
        train_trajectories,
        enabled=show_data_progress,
        desc="Build train pairs",
        total=len(train_trajectories),
    ):
        for step in range(len(trajectory) - 1):
            inputs.append(trajectory[step])
            targets_clean.append(trajectory[step + 1])
            targets_noisy.append(
                add_hf_noise_2d(
                    trajectory[step + 1],
                    config.noise_level,
                    config.nx,
                    config.ny,
                    Lx=config.Lx,
                    Ly=config.Ly,
                )
            )

    model_clean = build_model(
        method,
        config.nx,
        config.ny,
        seed=seed,
        device=device,
        loss=loss,
        model_width=config.train_model_width,
        model_depth=config.train_model_depth,
    )
    model_noisy = build_model(
        method,
        config.nx,
        config.ny,
        seed=seed + 10000,
        device=device,
        loss=loss,
        model_width=config.train_model_width,
        model_depth=config.train_model_depth,
    )

    model_clean.train(
        inputs,
        targets_clean,
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        trajectory=train_trajectories,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        show_progress=show_training_progress,
        progress_desc="Training clean model",
    )
    model_noisy.train(
        inputs,
        targets_noisy,
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        trajectory=train_trajectories,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        show_progress=show_training_progress,
        progress_desc="Training noisy model",
    )

    metrics = {
        "clean_l2": [],
        "noisy_l2": [],
        "clean_hfv": [],
        "noisy_hfv": [],
        "clean_lfv": [],
        "noisy_lfv": [],
    }

    trajectory_rows = []
    for case_idx, case in enumerate(
        progress_iter(
            test_cases,
            enabled=show_eval_progress,
            desc="Evaluation",
            total=len(test_cases),
        )
    ):
        omega_clean = rollout_2d(model_clean, case["omega0"], config.n_snapshots)
        omega_noisy = rollout_2d(model_noisy, case["omega0"], config.n_snapshots)

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
        metrics["clean_hfv"].append(clean_stats["hfv"])
        metrics["noisy_hfv"].append(noisy_stats["hfv"])
        metrics["clean_lfv"].append(clean_stats["lfv"])
        metrics["noisy_lfv"].append(noisy_stats["lfv"])

    eval_pair_index = int(np.clip(eval_pair_index, 0, len(inputs) - 1))
    test_case_index = int(np.clip(test_case_index, 0, len(test_cases) - 1))
    test_case = test_cases[test_case_index]
    max_test_step = test_case["omega_true"].shape[0] - 2
    test_step_index = int(np.clip(test_step_index, 0, max_test_step))

    eval_input = inputs[eval_pair_index]
    eval_target = targets_clean[eval_pair_index]
    eval_pred_clean = model_clean.forward(eval_input)
    eval_pred_noisy = model_noisy.forward(eval_input)

    test_input = test_case["omega_true"][test_step_index]
    test_target = test_case["omega_true"][test_step_index + 1]
    test_pred_clean = model_clean.forward(test_input)
    test_pred_noisy = model_noisy.forward(test_input)

    mean_metrics = {key: _safe_mean(value) for key, value in metrics.items()}
    return {
        **mean_metrics,
        "_viz": {
            "eval": {
                "input": eval_input,
                "target": eval_target,
                "pred_clean": eval_pred_clean,
                "pred_noisy": eval_pred_noisy,
            },
            "test": {
                "input": test_input,
                "target": test_target,
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
        },
        "_checkpoint_clean": model_clean.state_dict(),
        "_checkpoint_noisy": model_noisy.state_dict(),
        "_resolved_device": str(model_clean.device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Navier-Stokes experiment with YAML config, method, and seed."
    )
    parser.add_argument("config_yaml", type=str, help="Path to YAML config file")
    parser.add_argument("method", type=str, help="Model method (e.g., conv)")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_config = load_yaml_config(args.config_yaml)
    config = NSConfig.from_yaml(raw_config)

    paths = raw_config.get("paths", {})
    output_root = paths.get("output_dir", "output")
    checkpoint_root = paths.get("checkpoint_dir", "checkpoints")

    experiment = raw_config.get("experiment", {})
    experiment_name = experiment.get("name", "navier_stokes")
    training = raw_config.get("training", {})
    rsd_cfg = raw_config.get("rsd", {})
    requested_device = args.device if args.device is not None else str(training.get("device", "auto"))
    requested_loss = normalize_loss_name(args.loss if args.loss is not None else str(training.get("loss", "combined")))
    requested_basis = normalize_basis_name(args.basis if args.basis is not None else str(rsd_cfg.get("basis", "fourier")))

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

    results = run_single_seed(
        config,
        args.method,
        args.seed,
        device=requested_device,
        loss=requested_loss,
        basis=requested_basis,
        eval_pair_index=eval_pair_index,
        test_case_index=test_case_index,
        test_step_index=test_step_index,
        trajectory_case_indices=trajectory_case_indices,
        trajectory_step_indices=trajectory_step_indices,
        show_data_progress=data_progress,
        show_training_progress=training_progress,
        show_eval_progress=eval_progress,
    )

    checkpoint_clean = results.pop("_checkpoint_clean")
    checkpoint_noisy = results.pop("_checkpoint_noisy")
    viz_payload = results.pop("_viz")
    resolved_device = results.pop("_resolved_device")

    save_checkpoint(run_ckpt_dir / "model_clean.npz", checkpoint_clean)
    save_checkpoint(run_ckpt_dir / "model_noisy.npz", checkpoint_noisy)

    summary = {
        "experiment": experiment_name,
        "config_yaml": str(Path(args.config_yaml).resolve()),
        "method": args.method,
        "seed": args.seed,
        "device_requested": requested_device,
        "device_resolved": resolved_device,
        "loss": requested_loss,
        "basis": requested_basis,
        "metrics": results,
        "viz_indices": viz_payload["indices"],
    }
    save_json(run_out_dir / "results.json", summary)

    if artifacts.get("save_figures", True):
        from utils.plotting import save_clean_noisy_summary_plot

        save_clean_noisy_summary_plot(
            results,
            title=(
                f"Navier-Stokes | method={args.method} | loss={requested_loss} "
                f"| basis={requested_basis} | seed={args.seed}"
            ),
            output_path=run_out_dir / "summary.png",
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

            field_rows.append({"label": f"case {case_idx} | Truth", "traj": omega_true})
            field_rows.append({"label": f"case {case_idx} | Clean", "traj": omega_clean})
            field_rows.append({"label": f"case {case_idx} | Noisy", "traj": omega_noisy})

            error_rows.append({"label": f"case {case_idx} | Clean", "pred": omega_clean, "target": omega_true})
            error_rows.append({"label": f"case {case_idx} | Noisy", "pred": omega_noisy, "target": omega_true})

        save_trajectory_field_rows(
            field_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_omega_fields.png",
            title="Trajectory snapshots | Vorticity (truth, clean, noisy)",
            cmap="RdYlBu_r",
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
