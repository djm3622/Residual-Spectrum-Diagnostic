#!/usr/bin/env python3
"""Entry script for one Gray-Scott reaction-diffusion run.

Usage:
    python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml conv 1 --device auto --loss combined --basis fourier
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

from data.reaction_diffusion import GrayScottConfig, GrayScottSolver
from models.reaction_diffusion import LOSS_CHOICES, build_model, normalize_loss_name, rollout_coupled
from utils.config import load_yaml_config
from utils.diagnostics import BASIS_CHOICES, ReactionDiffusionRSDAnalyzer, normalize_basis_name
from utils.io import build_run_dirs, save_checkpoint, save_json
from utils.noise import add_hf_noise_coupled
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
    config: GrayScottConfig,
    method: str,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    basis: str = "fourier",
    eval_pair_index: int = 0,
    test_case_index: int = 0,
    test_step_index: int = 0,
    show_data_progress: bool = False,
    show_training_progress: bool = False,
    show_eval_progress: bool = False,
) -> Dict[str, float]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)

    solver = GrayScottSolver(config)
    rsd = ReactionDiffusionRSDAnalyzer(config, basis=basis)

    train_data = []
    for idx in progress_iter(
        range(config.n_train_trajectories),
        enabled=show_data_progress,
        desc="Data gen (train)",
        total=config.n_train_trajectories,
    ):
        u0, v0 = solver.initial_condition_random_seeds(n_seeds=15, seed=seed * 1000 + idx)
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
        u0, v0 = solver.initial_condition_random_seeds(n_seeds=15, seed=seed * 1000 + 500 + idx)
        _, u_true, v_true = solver.solve(u0, v0, config.t_final, config.n_snapshots)
        test_cases.append({"u0": u0, "v0": v0, "u_true": u_true, "v_true": v_true})

    inputs_u: List[np.ndarray] = []
    inputs_v: List[np.ndarray] = []
    targets_u_clean: List[np.ndarray] = []
    targets_v_clean: List[np.ndarray] = []
    targets_u_noisy: List[np.ndarray] = []
    targets_v_noisy: List[np.ndarray] = []

    for data in progress_iter(
        train_data,
        enabled=show_data_progress,
        desc="Build train pairs",
        total=len(train_data),
    ):
        u_traj = data["u"]
        v_traj = data["v"]

        for step in range(len(u_traj) - 1):
            inputs_u.append(u_traj[step])
            inputs_v.append(v_traj[step])
            targets_u_clean.append(u_traj[step + 1])
            targets_v_clean.append(v_traj[step + 1])

            u_noisy, v_noisy = add_hf_noise_coupled(
                u_traj[step + 1],
                v_traj[step + 1],
                config.noise_level,
                config.nx,
                config.ny,
                Lx=config.Lx,
                Ly=config.Ly,
            )
            targets_u_noisy.append(u_noisy)
            targets_v_noisy.append(v_noisy)

    model_clean = build_model(method, config.nx, config.ny, seed=seed, device=device, loss=loss)
    model_noisy = build_model(method, config.nx, config.ny, seed=seed + 10000, device=device, loss=loss)

    model_clean.train(
        inputs_u,
        inputs_v,
        targets_u_clean,
        targets_v_clean,
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        show_progress=show_training_progress,
        progress_desc="Training clean model",
    )
    model_noisy.train(
        inputs_u,
        inputs_v,
        targets_u_noisy,
        targets_v_noisy,
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
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

    for case in progress_iter(
        test_cases,
        enabled=show_eval_progress,
        desc="Evaluation",
        total=len(test_cases),
    ):
        u_clean, v_clean = rollout_coupled(model_clean, case["u0"], case["v0"], config.n_snapshots)
        u_noisy, v_noisy = rollout_coupled(model_noisy, case["u0"], case["v0"], config.n_snapshots)

        clean_stats = rsd.compute_metrics(u_clean, v_clean, case["u_true"], case["v_true"], dt)
        noisy_stats = rsd.compute_metrics(u_noisy, v_noisy, case["u_true"], case["v_true"], dt)

        metrics["clean_l2"].append(clean_stats["l2_error"])
        metrics["noisy_l2"].append(noisy_stats["l2_error"])
        metrics["clean_hfv"].append(clean_stats["hfv"])
        metrics["noisy_hfv"].append(noisy_stats["hfv"])
        metrics["clean_lfv"].append(clean_stats["lfv"])
        metrics["noisy_lfv"].append(noisy_stats["lfv"])

    eval_pair_index = int(np.clip(eval_pair_index, 0, len(inputs_u) - 1))
    test_case_index = int(np.clip(test_case_index, 0, len(test_cases) - 1))
    test_case = test_cases[test_case_index]
    max_test_step = test_case["u_true"].shape[0] - 2
    test_step_index = int(np.clip(test_step_index, 0, max_test_step))

    eval_input_u = inputs_u[eval_pair_index]
    eval_input_v = inputs_v[eval_pair_index]
    eval_target_u = targets_u_clean[eval_pair_index]
    eval_target_v = targets_v_clean[eval_pair_index]
    eval_pred_u_clean, eval_pred_v_clean = model_clean.forward(eval_input_u, eval_input_v)
    eval_pred_u_noisy, eval_pred_v_noisy = model_noisy.forward(eval_input_u, eval_input_v)

    test_input_u = test_case["u_true"][test_step_index]
    test_input_v = test_case["v_true"][test_step_index]
    test_target_u = test_case["u_true"][test_step_index + 1]
    test_target_v = test_case["v_true"][test_step_index + 1]
    test_pred_u_clean, test_pred_v_clean = model_clean.forward(test_input_u, test_input_v)
    test_pred_u_noisy, test_pred_v_noisy = model_noisy.forward(test_input_u, test_input_v)

    mean_metrics = {key: _safe_mean(value) for key, value in metrics.items()}
    return {
        **mean_metrics,
        "_viz": {
            "eval": {
                "input_u": eval_input_u,
                "input_v": eval_input_v,
                "target_u": eval_target_u,
                "target_v": eval_target_v,
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
        },
        "_checkpoint_clean": model_clean.state_dict(),
        "_checkpoint_noisy": model_noisy.state_dict(),
        "_resolved_device": str(model_clean.device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Gray-Scott experiment with YAML config, method, and seed."
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
    config = GrayScottConfig.from_yaml(raw_config)

    paths = raw_config.get("paths", {})
    output_root = paths.get("output_dir", "output")
    checkpoint_root = paths.get("checkpoint_dir", "checkpoints")

    experiment = raw_config.get("experiment", {})
    experiment_name = experiment.get("name", "reaction_diffusion")
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
                f"Reaction-Diffusion | method={args.method} | loss={requested_loss} "
                f"| basis={requested_basis} | seed={args.seed}"
            ),
            output_path=run_out_dir / "summary.png",
        )

    if artifacts.get("save_fit_visualizations", True):
        from utils.plotting import save_coupled_fit_panel

        fit_dir = run_out_dir / "fit_quality"
        fit_viz = artifacts.get("fit_visualization", {})
        input_viz = fit_viz.get("input", {}) if isinstance(fit_viz, dict) else {}
        output_viz = fit_viz.get("output", {}) if isinstance(fit_viz, dict) else {}
        input_cmap = str(input_viz.get("cmap", "cividis"))
        input_border_color = str(input_viz.get("border_color", "#2A9D8F"))
        input_border_width = float(input_viz.get("border_width", 2.0))
        output_cmap = str(output_viz.get("cmap", "viridis"))

        save_coupled_fit_panel(
            viz_payload["eval"]["input_u"],
            viz_payload["eval"]["input_v"],
            viz_payload["eval"]["target_u"],
            viz_payload["eval"]["target_v"],
            viz_payload["eval"]["pred_u_clean"],
            viz_payload["eval"]["pred_v_clean"],
            output_path=fit_dir / "eval_clean.png",
            title="Eval split | Clean model",
            output_cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
        )
        save_coupled_fit_panel(
            viz_payload["eval"]["input_u"],
            viz_payload["eval"]["input_v"],
            viz_payload["eval"]["target_u"],
            viz_payload["eval"]["target_v"],
            viz_payload["eval"]["pred_u_noisy"],
            viz_payload["eval"]["pred_v_noisy"],
            output_path=fit_dir / "eval_noisy.png",
            title="Eval split | Noisy model",
            output_cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
        )
        save_coupled_fit_panel(
            viz_payload["test"]["input_u"],
            viz_payload["test"]["input_v"],
            viz_payload["test"]["target_u"],
            viz_payload["test"]["target_v"],
            viz_payload["test"]["pred_u_clean"],
            viz_payload["test"]["pred_v_clean"],
            output_path=fit_dir / "test_clean.png",
            title="Test split | Clean model",
            output_cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
        )
        save_coupled_fit_panel(
            viz_payload["test"]["input_u"],
            viz_payload["test"]["input_v"],
            viz_payload["test"]["target_u"],
            viz_payload["test"]["target_v"],
            viz_payload["test"]["pred_u_noisy"],
            viz_payload["test"]["pred_v_noisy"],
            output_path=fit_dir / "test_noisy.png",
            title="Test split | Noisy model",
            output_cmap=output_cmap,
            input_cmap=input_cmap,
            input_border_color=input_border_color,
            input_border_width=input_border_width,
        )

    print("Run complete")
    print(f"Device: requested={requested_device} resolved={resolved_device}")
    print(f"Loss: {requested_loss}")
    print(f"Basis: {requested_basis}")
    print(f"Results: {run_out_dir / 'results.json'}")
    print(f"Checkpoints: {run_ckpt_dir}")


if __name__ == "__main__":
    main()
