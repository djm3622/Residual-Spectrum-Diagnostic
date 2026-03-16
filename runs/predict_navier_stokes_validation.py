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


def _infer_method_from_state_dict(state_dict: Dict[str, Any]) -> str | None:
    if not state_dict:
        return None
    keys = list(state_dict.keys())

    # TFNO commonly stores factorized spectral tensors as *.weight.core / *.weight.factors.*
    if any(".weight.core" in key for key in keys):
        return "tfno"

    # FNO-like operators expose fno_blocks keys.
    if any("fno_blocks" in key for key in keys):
        return "fno"

    # UNO checkpoints typically carry explicit UNO/integral-operator naming.
    if any(("uno" in key.lower()) or ("integral_operator" in key.lower()) for key in keys):
        return "uno"

    # Fallback for convolutional surrogate weights.
    return "conv"


def _load_weights_into_model(model: Any, state_dict: Dict[str, Any]) -> None:
    net = getattr(model, "net", None)
    if net is not None and hasattr(net, "load_state_dict"):
        try:
            net.load_state_dict(state_dict, strict=True)
            if hasattr(net, "eval"):
                net.eval()
            return
        except TypeError as exc:
            if "assign" in str(exc):
                # Older torch.Module.load_state_dict does not accept `assign`;
                # some neuralop wrappers forward it unconditionally.
                torch.nn.Module.load_state_dict(net, state_dict, strict=True)
                if hasattr(net, "eval"):
                    net.eval()
                return
            raise
        except RuntimeError as exc:
            raise RuntimeError(
                "Strict checkpoint load failed. This usually means model/config/package "
                "mismatch (for example neuraloperator version drift)."
            ) from exc

    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(state_dict)
            if hasattr(model, "eval"):
                model.eval()
            return
        except TypeError as exc:
            if "assign" in str(exc):
                torch.nn.Module.load_state_dict(model, state_dict, strict=True)
                if hasattr(model, "eval"):
                    model.eval()
                return
            raise

    raise ValueError("Model instance does not expose a load_state_dict-compatible API.")


def _split_fit_and_validation_trajectories(
    train_trajectories: list[np.ndarray],
    validation_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
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
    fit_items: list[tuple[int, np.ndarray]] = []
    val_items: list[tuple[int, np.ndarray]] = []
    for idx, traj in enumerate(train_trajectories):
        if idx in val_idx_set:
            val_items.append((idx, traj))
        else:
            fit_trajectories.append(traj)
            fit_items.append((idx, traj))

    if not fit_trajectories:
        # Mirror training behavior fallback.
        fit_trajectories = list(train_trajectories)
        fit_items = [(idx, traj) for idx, traj in enumerate(train_trajectories)]
        val_items = []

    return fit_trajectories, val_items, fit_items


def _evaluate_rollout_items(
    model: Any,
    items: list[tuple[int, np.ndarray]],
    rollout_context: int | None,
    autonomous_start: int,
    autonomous_block_size: int,
    show_progress: bool,
    progress_desc: str,
    collect_rollouts: bool,
) -> Dict[str, Any]:
    rel_l2_values: list[float] = []
    rel_l2_final_values: list[float] = []
    rel_l2_step1_values: list[float] = []
    rel_l2_autonomous_start_values: list[float] = []
    rel_l2_autonomous_block_values: list[float] = []
    mse_values: list[float] = []
    source_indices: list[int] = []
    per_step_rel_l2_sum: np.ndarray | None = None
    per_step_rel_l2_sq_sum: np.ndarray | None = None
    pred_rollouts: list[np.ndarray] = []
    target_rollouts: list[np.ndarray] = []

    for source_idx, omega_true in progress_iter(
        items,
        enabled=show_progress,
        desc=progress_desc,
        total=len(items),
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
        if n_steps > autonomous_start:
            numer_auto = float(np.linalg.norm(diff[autonomous_start]))
            denom_auto = float(np.linalg.norm(omega_true_arr[autonomous_start]) + 1e-12)
            rel_l2_autonomous_start_values.append(numer_auto / denom_auto)
            block_end = min(n_steps, autonomous_start + autonomous_block_size)
            block_diff = diff[autonomous_start:block_end].reshape(-1)
            block_true = omega_true_arr[autonomous_start:block_end].reshape(-1)
            numer_block = float(np.linalg.norm(block_diff))
            denom_block = float(np.linalg.norm(block_true) + 1e-12)
            rel_l2_autonomous_block_values.append(numer_block / denom_block)

        mse_values.append(float(np.mean(diff * diff)))
        source_indices.append(int(source_idx))

        flat_diff = diff.reshape(n_steps, -1)
        flat_true = omega_true_arr.reshape(n_steps, -1)
        per_step_rel = np.linalg.norm(flat_diff, axis=1) / (np.linalg.norm(flat_true, axis=1) + 1e-12)
        if per_step_rel_l2_sum is None:
            per_step_rel_l2_sum = np.zeros_like(per_step_rel, dtype=np.float64)
            per_step_rel_l2_sq_sum = np.zeros_like(per_step_rel, dtype=np.float64)
        per_step_rel_l2_sum += per_step_rel
        per_step_rel_l2_sq_sum += per_step_rel * per_step_rel

        if collect_rollouts:
            pred_rollouts.append(omega_pred)
            target_rollouts.append(omega_true_arr)

    if not rel_l2_values:
        raise ValueError(f"No trajectories were evaluated for split '{progress_desc}'.")

    n_eval = len(rel_l2_values)
    assert per_step_rel_l2_sum is not None
    assert per_step_rel_l2_sq_sum is not None
    per_step_mean = per_step_rel_l2_sum / float(n_eval)
    per_step_var = np.maximum(0.0, per_step_rel_l2_sq_sum / float(n_eval) - per_step_mean * per_step_mean)
    per_step_std = np.sqrt(per_step_var)

    return {
        "n_used": int(n_eval),
        "source_indices": source_indices,
        "metrics": {
            "rel_l2_mean": float(np.mean(rel_l2_values)),
            "rel_l2_std": float(np.std(rel_l2_values)),
            "rel_l2_final_mean": float(np.mean(rel_l2_final_values)),
            "rel_l2_final_std": float(np.std(rel_l2_final_values)),
            "rel_l2_step1_mean": float(np.mean(rel_l2_step1_values)) if rel_l2_step1_values else float("nan"),
            "rel_l2_step1_std": float(np.std(rel_l2_step1_values)) if rel_l2_step1_values else float("nan"),
            "autonomous_start_index": int(autonomous_start),
            "autonomous_block_size": int(autonomous_block_size),
            "rel_l2_autonomous_start_mean": (
                float(np.mean(rel_l2_autonomous_start_values))
                if rel_l2_autonomous_start_values
                else float("nan")
            ),
            "rel_l2_autonomous_start_std": (
                float(np.std(rel_l2_autonomous_start_values))
                if rel_l2_autonomous_start_values
                else float("nan")
            ),
            "rel_l2_autonomous_block_mean": (
                float(np.mean(rel_l2_autonomous_block_values))
                if rel_l2_autonomous_block_values
                else float("nan")
            ),
            "rel_l2_autonomous_block_std": (
                float(np.std(rel_l2_autonomous_block_values))
                if rel_l2_autonomous_block_values
                else float("nan")
            ),
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "rel_l2_per_step_mean": per_step_mean.tolist(),
            "rel_l2_per_step_std": per_step_std.tolist(),
        },
        "pred_rollouts": pred_rollouts,
        "target_rollouts": target_rollouts,
    }


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
        "--max-train-trajectories",
        type=int,
        default=0,
        help="If >0, also evaluate first N fit-train trajectories (for overfitting checks).",
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
        "--store-train-rollouts",
        action="store_true",
        help="If set, save train-subset predicted/target rollouts into NPZ.",
    )
    parser.add_argument(
        "--output-train-npz",
        type=str,
        default=None,
        help="Optional train-subset rollout NPZ path. Defaults next to checkpoint.",
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
        "--dataloader-workers",
        type=int,
        default=None,
        help=(
            "Override dataloader worker count used for normalizer-fit loader. "
            "Defaults to YAML training.dataloader_num_workers."
        ),
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
    checkpoint_payload = load_checkpoint(checkpoint_path)
    state_dict = _resolve_model_state_dict(checkpoint_payload)
    inferred_method_from_state = _infer_method_from_state_dict(state_dict)
    raw_config = load_yaml_config(args.config_yaml)
    config = NSConfig.from_yaml(raw_config)
    external_data_cfg = external_data_config_from_yaml(raw_config)

    training_cfg = raw_config.get("training", {})
    operator_config = training_cfg.get("neural_operator", {})

    method = str(args.method or inferred["method"] or inferred_method_from_state or "").strip()
    if not method:
        raise ValueError(
            "Unable to resolve method. Provide --method explicitly."
        )

    if args.seed is not None:
        seed = int(args.seed)
    elif inferred["seed"] is not None:
        seed = int(inferred["seed"])
    else:
        seed = 0
        print("Seed was not inferable from checkpoint path; defaulting to --seed 0.")

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

    fit_trajectories, val_items, fit_items = _split_fit_and_validation_trajectories(
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

    if args.dataloader_workers is None:
        dataloader_workers = resolve_dataloader_num_workers(config.train_dataloader_num_workers)
    else:
        dataloader_workers = max(0, int(args.dataloader_workers))
    _maybe_fit_normalizers(
        model=model,
        fit_trajectories=fit_trajectories,
        temporal_enabled=temporal_enabled,
        temporal_window=temporal_window,
        temporal_target_mode=temporal_target_mode,
        train_batch_size=config.train_batch_size,
        dataloader_workers=dataloader_workers,
    )

    _load_weights_into_model(model, state_dict)

    if temporal_enabled:
        rollout_context = temporal_window
    else:
        rollout_context = None
    autonomous_start = int(rollout_context) if rollout_context is not None else 1
    autonomous_block_size = int(temporal_window) if temporal_enabled else 1

    save_rollouts = bool(args.store_rollouts)
    val_eval = _evaluate_rollout_items(
        model=model,
        items=val_items,
        rollout_context=rollout_context,
        autonomous_start=autonomous_start,
        autonomous_block_size=autonomous_block_size,
        show_progress=bool(args.show_eval_progress),
        progress_desc="Validation rollout",
        collect_rollouts=save_rollouts,
    )

    max_train = int(args.max_train_trajectories)
    train_items_selected = fit_items[:max_train] if max_train > 0 else []
    train_eval = None
    if train_items_selected:
        train_eval = _evaluate_rollout_items(
            model=model,
            items=train_items_selected,
            rollout_context=rollout_context,
            autonomous_start=autonomous_start,
            autonomous_block_size=autonomous_block_size,
            show_progress=bool(args.show_eval_progress),
            progress_desc="Train-subset rollout",
            collect_rollouts=bool(args.store_train_rollouts),
        )

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
        "n_validation_trajectories_used": int(val_eval["n_used"]),
        "validation_train_indices_used": val_eval["source_indices"],
        "n_train_subset_trajectories_used": int(train_eval["n_used"]) if train_eval is not None else 0,
        "train_subset_indices_used": train_eval["source_indices"] if train_eval is not None else [],
        "n_snapshots": int(data_bundle.n_snapshots),
        "temporal": {
            "enabled": temporal_enabled,
            "input_steps": int(temporal_window),
            "target_mode": temporal_target_mode,
        },
        "metrics": val_eval["metrics"],
        "validation_metrics": val_eval["metrics"],
        "train_subset_metrics": train_eval["metrics"] if train_eval is not None else None,
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
            predictions=np.asarray(val_eval["pred_rollouts"], dtype=storage_dtype),
            targets=np.asarray(val_eval["target_rollouts"], dtype=storage_dtype),
            autonomous_start=np.asarray(autonomous_start, dtype=np.int64),
            autonomous_block_size=np.asarray(autonomous_block_size, dtype=np.int64),
            validation_train_indices=np.asarray(val_eval["source_indices"], dtype=np.int64),
            source_indices=np.asarray(val_eval["source_indices"], dtype=np.int64),
        )
        print(f"Saved rollouts: {output_npz}")

    if train_eval is not None and bool(args.store_train_rollouts):
        if args.output_train_npz:
            output_train_npz = Path(args.output_train_npz).expanduser()
        else:
            output_train_npz = checkpoint_path.with_name(f"{checkpoint_path.stem}_train_subset_rollouts.npz")
        output_train_npz.parent.mkdir(parents=True, exist_ok=True)
        storage_dtype = np.float16 if args.rollout_dtype == "float16" else np.float32
        np.savez_compressed(
            output_train_npz,
            predictions=np.asarray(train_eval["pred_rollouts"], dtype=storage_dtype),
            targets=np.asarray(train_eval["target_rollouts"], dtype=storage_dtype),
            autonomous_start=np.asarray(autonomous_start, dtype=np.int64),
            autonomous_block_size=np.asarray(autonomous_block_size, dtype=np.int64),
            train_indices=np.asarray(train_eval["source_indices"], dtype=np.int64),
            source_indices=np.asarray(train_eval["source_indices"], dtype=np.int64),
        )
        print(f"Saved train-subset rollouts: {output_train_npz}")

    print("Validation prediction run complete")
    print(f"Summary: {output_json}")
    print(f"Validation trajectories used: {val_eval['n_used']}")
    print(f"Validation mean rel-L2: {summary['validation_metrics']['rel_l2_mean']:.6e}")
    print(f"Validation mean final-step rel-L2: {summary['validation_metrics']['rel_l2_final_mean']:.6e}")
    if train_eval is not None:
        print(f"Train-subset trajectories used: {train_eval['n_used']}")
        print(f"Train-subset mean rel-L2: {summary['train_subset_metrics']['rel_l2_mean']:.6e}")


if __name__ == "__main__":
    main()
