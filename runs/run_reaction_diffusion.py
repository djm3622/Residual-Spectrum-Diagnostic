#!/usr/bin/env python3
"""Entry script for one Gray-Scott reaction-diffusion run.

Usage:
    python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml conv 1 --device auto --loss combined --basis fourier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.reaction_diffusion import GrayScottConfig, GrayScottSolver
from data.reaction_diffusion_external import (
    load_pdebench_reaction_diffusion_data,
    normalize_external_source,
    pdebench_source_config_from_yaml,
)
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


def _sample_initial_condition(solver: GrayScottSolver, config: GrayScottConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw one configured initial condition for Gray-Scott trajectories."""
    mode = str(config.initial_condition).strip().lower().replace("-", "_")

    if mode in {"random_seeds", "random", "seeds"}:
        return solver.initial_condition_random_seeds(
            n_seeds=max(1, int(config.initial_n_seeds)),
            seed=seed,
        )
    if mode in {"center_square", "square", "center"}:
        return solver.initial_condition_center_square(
            size=max(2, int(config.initial_square_size)),
            noise_amplitude=max(0.0, float(config.initial_noise_amplitude)),
            seed=seed,
        )

    raise ValueError(
        f"Unsupported data.initial_condition '{config.initial_condition}'. "
        "Use one of: random_seeds, center_square."
    )


def run_single_seed(
    config: GrayScottConfig,
    method: str,
    seed: int,
    device: str = "auto",
    loss: str = "combined",
    basis: str = "fourier",
    operator_config: Mapping[str, Any] | None = None,
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
) -> Dict[str, float]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)

    rsd = ReactionDiffusionRSDAnalyzer(config, basis=basis)

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

    inputs_u: List[np.ndarray] = []
    inputs_v: List[np.ndarray] = []
    inputs_u_noisy: List[np.ndarray] = []
    inputs_v_noisy: List[np.ndarray] = []
    targets_u_clean: List[np.ndarray] = []
    targets_v_clean: List[np.ndarray] = []
    targets_u_noisy: List[np.ndarray] = []
    targets_v_noisy: List[np.ndarray] = []
    val_inputs_u: List[np.ndarray] = []
    val_inputs_v: List[np.ndarray] = []
    val_inputs_u_noisy: List[np.ndarray] = []
    val_inputs_v_noisy: List[np.ndarray] = []
    val_targets_u_clean: List[np.ndarray] = []
    val_targets_v_clean: List[np.ndarray] = []
    val_targets_u_noisy: List[np.ndarray] = []
    val_targets_v_noisy: List[np.ndarray] = []
    pair_steps: List[int] = []
    train_trajectory_u: List[np.ndarray] = []
    train_trajectory_v: List[np.ndarray] = []
    train_trajectory_u_noisy: List[np.ndarray] = []
    train_trajectory_v_noisy: List[np.ndarray] = []

    for data in progress_iter(
        fit_train_data,
        enabled=show_data_progress,
        desc="Build train pairs",
        total=len(fit_train_data),
    ):
        u_traj = np.asarray(data["u"], dtype=np.float32)
        v_traj = np.asarray(data["v"], dtype=np.float32)

        u_traj_noisy = np.empty_like(u_traj, dtype=np.float32)
        v_traj_noisy = np.empty_like(v_traj, dtype=np.float32)
        for step in range(len(u_traj)):
            u_noisy_step, v_noisy_step = add_hf_noise_coupled(
                u_traj[step],
                v_traj[step],
                config.noise_level,
                config.nx,
                config.ny,
                Lx=config.Lx,
                Ly=config.Ly,
            )
            u_traj_noisy[step] = np.asarray(u_noisy_step, dtype=np.float32)
            v_traj_noisy[step] = np.asarray(v_noisy_step, dtype=np.float32)

        train_trajectory_u.append(u_traj)
        train_trajectory_v.append(v_traj)
        train_trajectory_u_noisy.append(u_traj_noisy)
        train_trajectory_v_noisy.append(v_traj_noisy)

        for step in range(len(u_traj) - 1):
            inputs_u.append(u_traj[step])
            inputs_v.append(v_traj[step])
            inputs_u_noisy.append(u_traj_noisy[step])
            inputs_v_noisy.append(v_traj_noisy[step])
            targets_u_clean.append(u_traj[step + 1])
            targets_v_clean.append(v_traj[step + 1])
            targets_u_noisy.append(u_traj_noisy[step + 1])
            targets_v_noisy.append(v_traj_noisy[step + 1])
            pair_steps.append(step)

    for data in val_data:
        u_traj = np.asarray(data["u"], dtype=np.float32)
        v_traj = np.asarray(data["v"], dtype=np.float32)

        u_traj_noisy = np.empty_like(u_traj, dtype=np.float32)
        v_traj_noisy = np.empty_like(v_traj, dtype=np.float32)
        for step in range(len(u_traj)):
            u_noisy_step, v_noisy_step = add_hf_noise_coupled(
                u_traj[step],
                v_traj[step],
                config.noise_level,
                config.nx,
                config.ny,
                Lx=config.Lx,
                Ly=config.Ly,
            )
            u_traj_noisy[step] = np.asarray(u_noisy_step, dtype=np.float32)
            v_traj_noisy[step] = np.asarray(v_noisy_step, dtype=np.float32)

        for step in range(len(u_traj) - 1):
            val_inputs_u.append(u_traj[step])
            val_inputs_v.append(v_traj[step])
            val_inputs_u_noisy.append(u_traj_noisy[step])
            val_inputs_v_noisy.append(v_traj_noisy[step])
            val_targets_u_clean.append(u_traj[step + 1])
            val_targets_v_clean.append(v_traj[step + 1])
            val_targets_u_noisy.append(u_traj_noisy[step + 1])
            val_targets_v_noisy.append(v_traj_noisy[step + 1])

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
    )
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
    )

    model_clean.train(
        inputs_u,
        inputs_v,
        targets_u_clean,
        targets_v_clean,
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        trajectory_u=train_trajectory_u,
        trajectory_v=train_trajectory_v,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        val_inputs_u=val_inputs_u,
        val_inputs_v=val_inputs_v,
        val_targets_u=val_targets_u_clean,
        val_targets_v=val_targets_v_clean,
        pair_steps=pair_steps,
        u_weight=config.train_u_weight,
        v_weight=config.train_v_weight,
        channel_balance_cap=config.train_channel_balance_cap,
        dynamics_weight=config.train_dynamics_weight,
        early_step_bias=config.train_early_step_bias,
        early_step_decay=config.train_early_step_decay,
        show_progress=show_training_progress,
        progress_desc="Training clean model",
    )
    model_noisy.train(
        inputs_u_noisy,
        inputs_v_noisy,
        targets_u_noisy,
        targets_v_noisy,
        lr=config.train_lr,
        n_iter=config.train_iterations,
        batch_size=config.train_batch_size,
        grad_clip=config.train_grad_clip,
        trajectory_u=train_trajectory_u_noisy,
        trajectory_v=train_trajectory_v_noisy,
        rollout_horizon=config.train_rollout_horizon,
        rollout_weight=config.train_rollout_weight,
        val_inputs_u=val_inputs_u_noisy,
        val_inputs_v=val_inputs_v_noisy,
        val_targets_u=val_targets_u_noisy,
        val_targets_v=val_targets_v_noisy,
        pair_steps=pair_steps,
        u_weight=config.train_u_weight,
        v_weight=config.train_v_weight,
        channel_balance_cap=config.train_channel_balance_cap,
        dynamics_weight=config.train_dynamics_weight,
        early_step_bias=config.train_early_step_bias,
        early_step_decay=config.train_early_step_decay,
        show_progress=show_training_progress,
        progress_desc="Training noisy model",
    )

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
        u_clean, v_clean = rollout_coupled(model_clean, case["u0"], case["v0"], config.n_snapshots)
        u_noisy, v_noisy = rollout_coupled(model_noisy, case["u0"], case["v0"], config.n_snapshots)

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
            "trajectory": {
                "case_indices": valid_trajectory_case_indices,
                "step_indices": valid_trajectory_steps,
                "rows": trajectory_rows,
            },
        },
        "_checkpoint_clean": model_clean.state_dict(),
        "_checkpoint_noisy": model_noisy.state_dict(),
        "_resolved_device": str(model_clean.device),
        "_data": {
            "source": str(data_source),
            "dt": float(dt),
            "n_snapshots": int(config.n_snapshots),
            "n_train": int(len(train_data)),
            "n_test": int(len(test_cases)),
            "metadata": dict(data_metadata or {}),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Gray-Scott experiment with YAML config, method, and seed."
    )
    parser.add_argument("config_yaml", type=str, help="Path to YAML config file")
    parser.add_argument(
        "method",
        type=str,
        help="Model method (conv, fno, tfno, uno, physics).",
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
    operator_config = training.get("neural_operator", {})
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
        trajectory_step_indices = trajectory_viz.get("step_indices", [0, 20, 40, 60, 80, 100, 120])
    else:
        trajectory_case_indices = [0, 1]
        trajectory_step_indices = [0, 20, 40, 60, 80, 100, 120]
    progress = raw_config.get("progress", {})
    progress_enabled = bool(progress.get("enabled", False))
    data_progress = bool(progress.get("data_generation", progress_enabled))
    training_progress = bool(progress.get("training", progress_enabled))
    eval_progress = bool(progress.get("evaluation", progress_enabled))

    external_cfg = ((raw_config.get("data", {}) or {}).get("external", {}) or {})
    source_name = normalize_external_source(str(external_cfg.get("source", "generated")))
    preloaded_train_data = None
    preloaded_test_cases = None
    snapshot_dt = None
    data_metadata: Dict[str, Any] = {}
    if source_name == "pdebench":
        pde_cfg = pdebench_source_config_from_yaml(raw_config)
        loaded = load_pdebench_reaction_diffusion_data(config, pde_cfg, seed=args.seed)
        preloaded_train_data = loaded.train_data
        preloaded_test_cases = loaded.test_cases
        snapshot_dt = float(loaded.dt)
        config.n_snapshots = int(loaded.n_snapshots)
        config.t_final = float(snapshot_dt) * float(max(config.n_snapshots - 1, 1))
        config.n_train_trajectories = int(len(preloaded_train_data))
        config.n_test_trajectories = int(len(preloaded_test_cases))
        data_metadata = dict(loaded.metadata)

    results = run_single_seed(
        config,
        args.method,
        args.seed,
        device=requested_device,
        loss=requested_loss,
        basis=requested_basis,
        operator_config=operator_config,
        eval_pair_index=eval_pair_index,
        test_case_index=test_case_index,
        test_step_index=test_step_index,
        trajectory_case_indices=trajectory_case_indices,
        trajectory_step_indices=trajectory_step_indices,
        show_data_progress=data_progress,
        show_training_progress=training_progress,
        show_eval_progress=eval_progress,
        preloaded_train_data=preloaded_train_data,
        preloaded_test_cases=preloaded_test_cases,
        snapshot_dt=snapshot_dt,
        data_source=source_name,
        data_metadata=data_metadata,
    )

    checkpoint_clean = results.pop("_checkpoint_clean")
    checkpoint_noisy = results.pop("_checkpoint_noisy")
    viz_payload = results.pop("_viz")
    resolved_device = results.pop("_resolved_device")
    data_info = results.pop("_data")

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
        "data": data_info,
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
            case_bucket["u_true"] = row["u_true"]
            case_bucket["v_true"] = row["v_true"]
            case_bucket[f"u_{row['model']}"] = row["u_pred"]
            case_bucket[f"v_{row['model']}"] = row["v_pred"]

        ordered_cases = [int(idx) for idx in case_indices if int(idx) in by_case]
        if not ordered_cases:
            ordered_cases = sorted(by_case.keys())

        u_field_rows = []
        v_field_rows = []
        u_rows = []
        v_rows = []
        for case_idx in ordered_cases:
            bucket = by_case[case_idx]
            u_true = bucket.get("u_true")
            v_true = bucket.get("v_true")
            u_clean = bucket.get("u_clean")
            v_clean = bucket.get("v_clean")
            u_noisy = bucket.get("u_noisy")
            v_noisy = bucket.get("v_noisy")

            if u_true is None or v_true is None or u_clean is None or v_clean is None or u_noisy is None or v_noisy is None:
                continue

            u_field_rows.append({"label": f"case {case_idx} | Truth", "traj": u_true})
            u_field_rows.append({"label": f"case {case_idx} | Clean", "traj": u_clean})
            u_field_rows.append({"label": f"case {case_idx} | Noisy", "traj": u_noisy})

            v_field_rows.append({"label": f"case {case_idx} | Truth", "traj": v_true})
            v_field_rows.append({"label": f"case {case_idx} | Clean", "traj": v_clean})
            v_field_rows.append({"label": f"case {case_idx} | Noisy", "traj": v_noisy})

            u_rows.append({"label": f"case {case_idx} | Clean", "pred": u_clean, "target": u_true})
            u_rows.append({"label": f"case {case_idx} | Noisy", "pred": u_noisy, "target": u_true})
            v_rows.append({"label": f"case {case_idx} | Clean", "pred": v_clean, "target": v_true})
            v_rows.append({"label": f"case {case_idx} | Noisy", "pred": v_noisy, "target": v_true})

        save_trajectory_field_rows(
            u_field_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_u_fields.png",
            title="Trajectory snapshots | Species u (truth, clean, noisy)",
            cmap="turbo",
        )
        save_trajectory_field_rows(
            v_field_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_v_fields.png",
            title="Trajectory snapshots | Species v (truth, clean, noisy)",
            cmap="turbo",
        )

        save_trajectory_error_rows(
            u_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_u_error.png",
            title="Trajectory absolute error snapshots | Species u",
        )
        save_trajectory_error_rows(
            v_rows,
            step_indices=step_indices,
            output_path=fit_dir / "trajectory_v_error.png",
            title="Trajectory absolute error snapshots | Species v",
        )
    print("Run complete")
    print(f"Device: requested={requested_device} resolved={resolved_device}")
    print(f"Loss: {requested_loss}")
    print(f"Basis: {requested_basis}")
    print(f"Data source: {data_info['source']}")
    print(f"Results: {run_out_dir / 'results.json'}")
    print(f"Checkpoints: {run_ckpt_dir}")


if __name__ == "__main__":
    main()
