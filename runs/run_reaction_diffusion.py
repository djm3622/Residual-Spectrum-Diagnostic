#!/usr/bin/env python3
"""Entry script for one Gray-Scott reaction-diffusion run.

Usage:
    python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml conv 1
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
from models.reaction_diffusion import build_model, rollout_coupled
from utils.config import load_yaml_config
from utils.diagnostics import ReactionDiffusionRSDAnalyzer
from utils.io import build_run_dirs, save_checkpoint, save_json
from utils.noise import add_hf_noise_coupled


def run_single_seed(config: GrayScottConfig, method: str, seed: int) -> Dict[str, float]:
    """Train/evaluate clean and noisy models for one seed."""
    np.random.seed(seed * 1000)

    solver = GrayScottSolver(config)
    rsd = ReactionDiffusionRSDAnalyzer(config)

    train_data = []
    for idx in range(config.n_train_trajectories):
        u0, v0 = solver.initial_condition_random_seeds(n_seeds=15, seed=seed * 1000 + idx)
        t_save, u_traj, v_traj = solver.solve(u0, v0, config.t_final, config.n_snapshots)
        train_data.append({"u": u_traj, "v": v_traj})
    dt = float(t_save[1] - t_save[0])

    test_cases = []
    for idx in range(config.n_test_trajectories):
        u0, v0 = solver.initial_condition_random_seeds(n_seeds=15, seed=seed * 1000 + 500 + idx)
        _, u_true, v_true = solver.solve(u0, v0, config.t_final, config.n_snapshots)
        test_cases.append({"u0": u0, "v0": v0, "u_true": u_true, "v_true": v_true})

    inputs_u: List[np.ndarray] = []
    inputs_v: List[np.ndarray] = []
    targets_u_clean: List[np.ndarray] = []
    targets_v_clean: List[np.ndarray] = []
    targets_u_noisy: List[np.ndarray] = []
    targets_v_noisy: List[np.ndarray] = []

    for data in train_data:
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
            )
            targets_u_noisy.append(u_noisy)
            targets_v_noisy.append(v_noisy)

    model_clean = build_model(method, config.nx, config.ny, seed=seed)
    model_noisy = build_model(method, config.nx, config.ny, seed=seed + 10000)

    model_clean.train(
        inputs_u,
        inputs_v,
        targets_u_clean,
        targets_v_clean,
        lr=config.train_lr,
        n_iter=config.train_iterations,
    )
    model_noisy.train(
        inputs_u,
        inputs_v,
        targets_u_noisy,
        targets_v_noisy,
        lr=config.train_lr,
        n_iter=config.train_iterations,
    )

    metrics = {
        "clean_l2": [],
        "noisy_l2": [],
        "clean_hfv": [],
        "noisy_hfv": [],
        "clean_lfv": [],
        "noisy_lfv": [],
    }

    for case in test_cases:
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

    mean_metrics = {key: float(np.mean(value)) for key, value in metrics.items()}
    return {
        **mean_metrics,
        "_checkpoint_clean": model_clean.state_dict(),
        "_checkpoint_noisy": model_noisy.state_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Gray-Scott experiment with YAML config, method, and seed."
    )
    parser.add_argument("config_yaml", type=str, help="Path to YAML config file")
    parser.add_argument("method", type=str, help="Model method (e.g., conv or linear)")
    parser.add_argument("seed", type=int, help="Random seed number")
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
            title=f"Reaction-Diffusion | method={args.method} | seed={args.seed}",
            output_path=run_out_dir / "summary.png",
        )

    print("Run complete")
    print(f"Results: {run_out_dir / 'results.json'}")
    print(f"Checkpoints: {run_ckpt_dir}")


if __name__ == "__main__":
    main()
