#!/usr/bin/env python3
"""Load a trained NS checkpoint and simulate validation-set predictions.

Example:
    python3 runs/predict_navier_stokes_validation.py \
      /home/djm3622/Residual-Spectrum-Diagnostic/checkpoints/unsteady_ns/tfno/loss_l2/basis_fourier/seed_0/model_clean_epoch_0020.npz \
      --config-yaml configs/unsteady_ns.yaml \
      --store-rollouts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.navier_stokes import NSConfig
from data.navier_stokes.external import external_data_config_from_yaml, load_navier_stokes_trajectory_data
from models.navier_stokes import LOSS_CHOICES, build_model, normalize_loss_name, rollout_2d
from runs.helpers.indexed_datasets import NavierStokesIndexedPairDataset, resolve_dataloader_num_workers
from runs.helpers.temporal import resolve_temporal_training_config
from utils.config import load_yaml_config
from utils.io import load_checkpoint, save_json
from utils.progress import progress_iter
from utils.torch_runtime import DEVICE_CHOICES


def _infer_run_fields_from_checkpoint_path(checkpoint_path: Path) -> Dict[str, Any]:
    parts = list(checkpoint_path.parts)
    inferred: Dict[str, Any] = {
        "method": None,
        "loss": None,
        "basis": None,
        "seed": None,
    }

    for idx, part in enumerate(parts):
        if part.startswith("loss_"):
            inferred["loss"] = part[len("loss_") :] or None
            if idx > 0:
                inferred["method"] = parts[idx - 1]
        elif part.startswith("basis_"):
            inferred["basis"] = part[len("basis_") :] or None
        elif part.startswith("seed_"):
            match = re.match(r"^seed_(\d+)$", part)
            if match is not None:
                inferred["seed"] = int(match.group(1))
    return inferred


def _resolve_model_state_dict(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload is not a dictionary.")

    candidate_paths = [
        ("training_state", "model_state"),
        ("training_state", "best_model_state"),
        ("model_state",),
        ("best_model_state",),
    ]
    for path in candidate_paths:
        node: Any = payload
        valid_path = True
        for key in path:
            if not isinstance(node, dict):
                valid_path = False
                break
            node = node.get(key)
        if valid_path and isinstance(node, dict):
            return node

    if payload and all(isinstance(key, str) for key in payload.keys()):
        if any(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload

    raise ValueError(
        "Could not find model weights in checkpoint. "
        "Expected training_state.model_state (or equivalent)."
    )


def _load_weights_into_model(model: Any, state_dict: Dict[str, Any]) -> None:
    net = getattr(model, "net", None)
    if net is not None and hasattr(net, "load_state_dict"):
        try:
            net.load_state_dict(state_dict, strict=True)
            if hasattr(net, "eval"):
                net.eval()
            return
        except RuntimeError:
            net.load_state_dict(state_dict, strict=False)
            if hasattr(net, "eval"):
                net.eval()
            return

    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state_dict)
        if hasattr(model, "eval"):
            model.eval()
        return

    raise ValueError("Model instance does not expose a load_state_dict-compatible API.")


def _split_fit_and_validation_trajectories(
    train_trajectories: list[np.ndarray],
    validation_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[tuple[int, np.ndarray]]]:
    n_train_total = len(train_trajectories)
    n_val = 0
    val_fraction = float(np.clip(validation_fraction, 0.0, 0.95))
    if n_train_total > 1 and val_fraction > 0.0:
        n_val = int(round(n_train_total * val_fraction))
        n_val = max(1, min(n_train_total - 1, n_val))

    split_rng = np.random.default_rng(seed * 1000 + 707)
    split_perm = list(split_rng.permutation(n_train_total))
    val_idx_set = set(split_perm[:n_val])

    fit_trajectories: list[np.ndarray] = []
    val_items: list[tuple[int, np.ndarray]] = []
    for idx, traj in enumerate(train_trajectories):
        if idx in val_idx_set:
            val_items.append((idx, traj))
        else:
            fit_trajectories.append(traj)

    if not fit_trajectories:
        # Mirror training behavior fallback.
        fit_trajectories = list(train_trajectories)
        val_items = []

    return fit_trajectories, val_items


def _maybe_fit_normalizers(
    model: Any,
    fit_trajectories: list[np.ndarray],
    temporal_enabled: bool,
    temporal_window: int,
    temporal_target_mode: str,
    train_batch_size: int,
    dataloader_workers: int,
) -> None:
    fit_fn = getattr(model, "_fit_normalizers_from_loader", None)
    if not callable(fit_fn):
        return

    train_dataset = NavierStokesIndexedPairDataset(
        fit_trajectories,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
    )
    if len(train_dataset) == 0:
        raise ValueError("Fit split produced zero supervised pairs; cannot fit normalizers.")

    batch = max(1, min(int(train_batch_size), len(train_dataset)))
    num_workers = max(0, int(dataloader_workers))
    pin_memory = str(getattr(model, "device", "")).startswith("cuda")
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    stats_loader = DataLoader(train_dataset, **loader_kwargs)
    fit_fn(stats_loader)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one Navier-Stokes checkpoint and simulate predictions on the validation split "
            "defined by the current YAML config."
        )
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint .npz file.")
    parser.add_argument(
        "--config-yaml",
        type=str,
        default="configs/unsteady_ns.yaml",
        help="Path to run config YAML (default: configs/unsteady_ns.yaml).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Model method (conv, fno, tfno, uno). Inferred from checkpoint path when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed used for train/validation split. Inferred from checkpoint path when omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default=None,
        help="Compute device override (auto/cpu/cuda/mps). Defaults to training.device in YAML.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=LOSS_CHOICES,
        default=None,
        help="Loss name used to build the model wrapper. Defaults to YAML, then checkpoint path loss tag.",
    )
    parser.add_argument(
        "--max-validation-trajectories",
        type=int,
        default=0,
        help="If >0, evaluate only the first N validation trajectories.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional summary JSON path. Defaults next to checkpoint.",
    )
    parser.add_argument(
        "--store-rollouts",
        action="store_true",
        help="If set, save validation predicted/target rollouts into NPZ.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Optional rollout NPZ path when --store-rollouts is set. Defaults next to checkpoint.",
    )
    parser.add_argument(
        "--rollout-dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Storage dtype for saved rollouts.",
    )
    parser.add_argument(
        "--show-data-progress",
        action="store_true",
        help="Show progress bars while loading/building data trajectories.",
    )
    parser.add_argument(
        "--show-eval-progress",
        action="store_true",
        help="Show progress bar during validation rollout simulation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inferred = _infer_run_fields_from_checkpoint_path(checkpoint_path)
    raw_config = load_yaml_config(args.config_yaml)
    config = NSConfig.from_yaml(raw_config)
    external_data_cfg = external_data_config_from_yaml(raw_config)

    training_cfg = raw_config.get("training", {})
    operator_config = training_cfg.get("neural_operator", {})

    method = str(args.method or inferred["method"] or "").strip()
    if not method:
        raise ValueError(
            "Unable to resolve method. Provide --method or use a checkpoint path containing loss_<...> folders."
        )

    if args.seed is not None:
        seed = int(args.seed)
    elif inferred["seed"] is not None:
        seed = int(inferred["seed"])
    else:
        raise ValueError("Unable to resolve seed. Provide --seed or use a checkpoint path containing seed_<N>.")

    config_loss = normalize_loss_name(str(training_cfg.get("loss", "combined")))
    inferred_loss = inferred["loss"]
    if args.loss is not None:
        requested_loss = normalize_loss_name(args.loss)
    elif "loss" in training_cfg:
        requested_loss = config_loss
    elif inferred_loss:
        requested_loss = normalize_loss_name(str(inferred_loss))
    else:
        requested_loss = config_loss

    requested_device = str(args.device if args.device is not None else training_cfg.get("device", "auto"))

    temporal_cfg = resolve_temporal_training_config(method, operator_config)
    temporal_enabled = bool(temporal_cfg["enabled"])
    temporal_window = int(temporal_cfg["input_steps"])
    temporal_target_mode = str(temporal_cfg["target_mode"])

    data_bundle = load_navier_stokes_trajectory_data(
        config,
        external_data_cfg,
        seed=seed,
        show_data_progress=bool(args.show_data_progress),
    )

    fit_trajectories, val_items = _split_fit_and_validation_trajectories(
        data_bundle.train_trajectories,
        validation_fraction=config.train_validation_fraction,
        seed=seed,
    )
    if not val_items:
        raise ValueError(
            "Validation split is empty. Increase training.validation_fraction in YAML or use more train trajectories."
        )

    max_val = int(args.max_validation_trajectories)
    n_validation_total = len(val_items)
    if max_val > 0:
        val_items = val_items[:max_val]

    model = build_model(
        method,
        config.nx,
        config.ny,
        seed=seed,
        device=requested_device,
        loss=requested_loss,
        model_width=config.train_model_width,
        model_depth=config.train_model_depth,
        operator_config=operator_config,
    )

    dataloader_workers = resolve_dataloader_num_workers(config.train_dataloader_num_workers)
    _maybe_fit_normalizers(
        model=model,
        fit_trajectories=fit_trajectories,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
        train_batch_size=config.train_batch_size,
        dataloader_workers=dataloader_workers,
    )

    checkpoint_payload = load_checkpoint(checkpoint_path)
    state_dict = _resolve_model_state_dict(checkpoint_payload)
    _load_weights_into_model(model, state_dict)

    if temporal_enabled:
        rollout_context = temporal_window
    else:
        rollout_context = None

    rel_l2_values: list[float] = []
    rel_l2_final_values: list[float] = []
    rel_l2_step1_values: list[float] = []
    mse_values: list[float] = []
    val_indices: list[int] = []
    per_step_rel_l2_sum: np.ndarray | None = None
    per_step_rel_l2_sq_sum: np.ndarray | None = None

    save_rollouts = bool(args.store_rollouts)
    pred_rollouts: list[np.ndarray] = []
    target_rollouts: list[np.ndarray] = []

    for val_idx, omega_true in progress_iter(
        val_items,
        enabled=bool(args.show_eval_progress),
        desc="Validation rollout",
        total=len(val_items),
    ):
        omega_true_arr = np.asarray(omega_true, dtype=np.float32)
        n_steps = int(omega_true_arr.shape[0])
        if rollout_context is not None:
            context = np.asarray(omega_true_arr[:rollout_context], dtype=np.float32)
        else:
            context = None

        omega_pred = np.asarray(
            rollout_2d(model, omega_true_arr[0], n_steps, context=context),
            dtype=np.float32,
        )
        diff = omega_pred - omega_true_arr

        numer_full = float(np.linalg.norm(diff))
        denom_full = float(np.linalg.norm(omega_true_arr) + 1e-12)
        rel_l2_values.append(numer_full / denom_full)

        numer_final = float(np.linalg.norm(diff[-1]))
        denom_final = float(np.linalg.norm(omega_true_arr[-1]) + 1e-12)
        rel_l2_final_values.append(numer_final / denom_final)

        if n_steps > 1:
            numer_step1 = float(np.linalg.norm(diff[1]))
            denom_step1 = float(np.linalg.norm(omega_true_arr[1]) + 1e-12)
            rel_l2_step1_values.append(numer_step1 / denom_step1)

        mse_values.append(float(np.mean(diff * diff)))
        val_indices.append(int(val_idx))

        flat_diff = diff.reshape(n_steps, -1)
        flat_true = omega_true_arr.reshape(n_steps, -1)
        per_step_rel = np.linalg.norm(flat_diff, axis=1) / (np.linalg.norm(flat_true, axis=1) + 1e-12)

        if per_step_rel_l2_sum is None:
            per_step_rel_l2_sum = np.zeros_like(per_step_rel, dtype=np.float64)
            per_step_rel_l2_sq_sum = np.zeros_like(per_step_rel, dtype=np.float64)
        per_step_rel_l2_sum += per_step_rel
        per_step_rel_l2_sq_sum += per_step_rel * per_step_rel

        if save_rollouts:
            pred_rollouts.append(omega_pred)
            target_rollouts.append(omega_true_arr)

    if not rel_l2_values:
        raise ValueError("No validation trajectories were evaluated.")

    n_eval = len(rel_l2_values)
    assert per_step_rel_l2_sum is not None
    assert per_step_rel_l2_sq_sum is not None
    per_step_mean = per_step_rel_l2_sum / float(n_eval)
    per_step_var = np.maximum(0.0, per_step_rel_l2_sq_sum / float(n_eval) - per_step_mean * per_step_mean)
    per_step_std = np.sqrt(per_step_var)

    summary = {
        "config_yaml": str(Path(args.config_yaml).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "method": method,
        "seed": int(seed),
        "loss": requested_loss,
        "device_requested": requested_device,
        "device_resolved": str(getattr(model, "device", requested_device)),
        "data_source": data_bundle.source,
        "data_metadata": data_bundle.metadata,
        "n_train_trajectories": int(len(data_bundle.train_trajectories)),
        "n_fit_trajectories": int(len(fit_trajectories)),
        "n_validation_trajectories_total": int(n_validation_total),
        "n_validation_trajectories_used": int(n_eval),
        "validation_train_indices_used": val_indices,
        "n_snapshots": int(data_bundle.n_snapshots),
        "temporal": {
            "enabled": temporal_enabled,
            "input_steps": int(temporal_window),
            "target_mode": temporal_target_mode,
        },
        "metrics": {
            "rel_l2_mean": float(np.mean(rel_l2_values)),
            "rel_l2_std": float(np.std(rel_l2_values)),
            "rel_l2_final_mean": float(np.mean(rel_l2_final_values)),
            "rel_l2_final_std": float(np.std(rel_l2_final_values)),
            "rel_l2_step1_mean": float(np.mean(rel_l2_step1_values)) if rel_l2_step1_values else float("nan"),
            "rel_l2_step1_std": float(np.std(rel_l2_step1_values)) if rel_l2_step1_values else float("nan"),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "rel_l2_per_step_mean": per_step_mean.tolist(),
            "rel_l2_per_step_std": per_step_std.tolist(),
        },
    }

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
    else:
        output_json = checkpoint_path.with_name(f"{checkpoint_path.stem}_validation_summary.json")
    save_json(output_json, summary)

    if save_rollouts:
        if args.output_npz:
            output_npz = Path(args.output_npz).expanduser()
        else:
            output_npz = checkpoint_path.with_name(f"{checkpoint_path.stem}_validation_rollouts.npz")
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        storage_dtype = np.float16 if args.rollout_dtype == "float16" else np.float32
        np.savez_compressed(
            output_npz,
            predictions=np.asarray(pred_rollouts, dtype=storage_dtype),
            targets=np.asarray(target_rollouts, dtype=storage_dtype),
            validation_train_indices=np.asarray(val_indices, dtype=np.int64),
        )
        print(f"Saved rollouts: {output_npz}")

    print("Validation prediction run complete")
    print(f"Summary: {output_json}")
    print(f"Validation trajectories used: {n_eval}")
    print(f"Mean rel-L2: {summary['metrics']['rel_l2_mean']:.6e}")
    print(f"Mean final-step rel-L2: {summary['metrics']['rel_l2_final_mean']:.6e}")


if __name__ == "__main__":
    main()
