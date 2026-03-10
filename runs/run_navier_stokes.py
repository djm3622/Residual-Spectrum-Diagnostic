#!/usr/bin/env python3
"""Entry script for one Navier-Stokes run.

Usage:
    python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1
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
from models.navier_stokes import build_model, rollout_2d
from utils.config import load_yaml_config
from utils.diagnostics import NavierStokesRSDAnalyzer
from utils.io import build_run_dirs, save_checkpoint, save_json
from utils.noise import add_hf_noise_2d


def run_single_seed(config: NSConfig, method: str, seed: int) -> Dict[str, float]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)

    solver = NavierStokes2D(config)
    rsd = NavierStokesRSDAnalyzer(config)

    train_trajectories: List[np.ndarray] = []
    for idx in range(config.n_train_trajectories):
        omega0 = solver.random_initial_condition(seed=seed * 1000 + idx)
        t_save, omega_traj = solver.solve(omega0, config.t_final, config.n_snapshots)
        train_trajectories.append(omega_traj)
    dt = float(t_save[1] - t_save[0])

    test_cases = []
    for idx in range(config.n_test_trajectories):
        omega0 = solver.random_initial_condition(seed=seed * 1000 + 500 + idx)
        _, omega_true = solver.solve(omega0, config.t_final, config.n_snapshots)
        test_cases.append({"omega0": omega0, "omega_true": omega_true})

    inputs = []
    targets_clean = []
    targets_noisy = []

    for trajectory in train_trajectories:
        for step in range(len(trajectory) - 1):
            inputs.append(trajectory[step])
            targets_clean.append(trajectory[step + 1])
            targets_noisy.append(
                add_hf_noise_2d(
                    trajectory[step + 1],
                    config.noise_level,
                    config.nx,
                    config.ny,
                )
            )

    model_clean = build_model(method, config.nx, config.ny, seed=seed)
    model_noisy = build_model(method, config.nx, config.ny, seed=seed + 10000)

    model_clean.train(inputs, targets_clean, lr=config.train_lr, n_iter=config.train_iterations)
    model_noisy.train(inputs, targets_noisy, lr=config.train_lr, n_iter=config.train_iterations)

    metrics = {
        "clean_l2": [],
        "noisy_l2": [],
        "clean_hfv": [],
        "noisy_hfv": [],
        "clean_lfv": [],
        "noisy_lfv": [],
    }

    for case in test_cases:
        omega_clean = rollout_2d(model_clean, case["omega0"], config.n_snapshots)
        omega_noisy = rollout_2d(model_noisy, case["omega0"], config.n_snapshots)

        clean_stats = rsd.compute_metrics(omega_clean, case["omega_true"], dt)
        noisy_stats = rsd.compute_metrics(omega_noisy, case["omega_true"], dt)

        metrics["clean_l2"].append(clean_stats["l2_error"])
        metrics["noisy_l2"].append(noisy_stats["l2_error"])
        metrics["clean_hfv"].append(clean_stats["hfv"])
        metrics["noisy_hfv"].append(noisy_stats["hfv"])
        metrics["clean_lfv"].append(clean_stats["lfv"])
        metrics["noisy_lfv"].append(noisy_stats["lfv"])

    mean_metrics = {key: float(np.mean(value)) for key, value in metrics.items()}
    return {
        **mean_metrics,
        "_checkpoint_clean": model_clean.state_dict(),
        "_checkpoint_noisy": model_noisy.state_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Navier-Stokes experiment with YAML config, method, and seed."
    )
    parser.add_argument("config_yaml", type=str, help="Path to YAML config file")
    parser.add_argument("method", type=str, help="Model method (e.g., conv or linear)")
    parser.add_argument("seed", type=int, help="Random seed number")
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

    run_out_dir, run_ckpt_dir = build_run_dirs(
        output_root,
        checkpoint_root,
        problem_name=experiment_name,
        method=args.method,
        seed=args.seed,
    )

    results = run_single_seed(config, args.method, args.seed)

    checkpoint_clean = results.pop("_checkpoint_clean")
    checkpoint_noisy = results.pop("_checkpoint_noisy")

    save_checkpoint(run_ckpt_dir / "model_clean.npz", checkpoint_clean)
    save_checkpoint(run_ckpt_dir / "model_noisy.npz", checkpoint_noisy)

    summary = {
        "experiment": experiment_name,
        "config_yaml": str(Path(args.config_yaml).resolve()),
        "method": args.method,
        "seed": args.seed,
        "metrics": results,
    }
    save_json(run_out_dir / "results.json", summary)

    artifacts = raw_config.get("artifacts", {})
    if artifacts.get("save_figures", True):
        from utils.plotting import save_clean_noisy_summary_plot

        save_clean_noisy_summary_plot(
            results,
            title=f"Navier-Stokes | method={args.method} | seed={args.seed}",
            output_path=run_out_dir / "summary.png",
        )

    print("Run complete")
    print(f"Results: {run_out_dir / 'results.json'}")
    print(f"Checkpoints: {run_ckpt_dir}")


if __name__ == "__main__":
    main()
