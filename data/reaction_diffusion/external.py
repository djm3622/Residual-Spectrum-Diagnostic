"""Reaction-diffusion trajectory adapters for generated and PDEBench datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .helper.external_io import _as_optional_float, _is_grouped_samples
from .helper.pdebench_loader import (
    load_grouped_pdebench_split,
    load_root_pdebench_split,
)
from .solver import GrayScottConfig


@dataclass
class PDEBenchReactionDiffusionSourceConfig:
    """Settings for loading reaction-diffusion trajectories from PDEBench HDF5."""

    file_path: str = ""
    dataset_key: str = "data"
    layout: str = "AUTO"
    sample_grouped: bool = False
    u_channel_index: int = 0
    v_channel_index: int = 1
    time_stride: int = 1
    spatial_stride: int = 1
    n_train: int = 0
    n_test: int = 0
    shuffle: bool = True
    split_seed_offset: int = 1207
    dt: float | None = None


@dataclass
class ReactionDiffusionTrajectoryData:
    """Unified trajectory payload consumed by the RD run script."""

    train_data: List[Dict[str, np.ndarray]]
    test_cases: List[Dict[str, np.ndarray]]
    dt: float
    n_snapshots: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_external_source(source: str) -> str:
    normalized = str(source).strip().lower().replace("-", "_")
    alias_map = {
        "generated": "generated",
        "solver": "generated",
        "synthetic": "generated",
        "pdebench": "pdebench",
        "pde_bench": "pdebench",
    }
    if normalized not in alias_map:
        supported = ", ".join(sorted(set(alias_map.values())))
        raise ValueError(f"Unsupported external data source '{source}'. Use one of: {supported}")
    return alias_map[normalized]


def pdebench_source_config_from_yaml(raw_config: Mapping[str, Any]) -> PDEBenchReactionDiffusionSourceConfig:
    data_cfg = raw_config.get("data", {})
    ext_cfg = data_cfg.get("external", {})
    if not isinstance(ext_cfg, Mapping):
        ext_cfg = {}
    pde_cfg_raw = ext_cfg.get("pdebench", {})
    if not isinstance(pde_cfg_raw, Mapping):
        pde_cfg_raw = {}

    return PDEBenchReactionDiffusionSourceConfig(
        file_path=str(pde_cfg_raw.get("file_path", "")),
        dataset_key=str(pde_cfg_raw.get("dataset_key", "data")),
        layout=str(pde_cfg_raw.get("layout", "AUTO")),
        sample_grouped=bool(pde_cfg_raw.get("sample_grouped", False)),
        u_channel_index=int(pde_cfg_raw.get("u_channel_index", 0)),
        v_channel_index=int(pde_cfg_raw.get("v_channel_index", 1)),
        time_stride=max(1, int(pde_cfg_raw.get("time_stride", 1))),
        spatial_stride=max(1, int(pde_cfg_raw.get("spatial_stride", 1))),
        n_train=int(pde_cfg_raw.get("n_train", 0)),
        n_test=int(pde_cfg_raw.get("n_test", 0)),
        shuffle=bool(pde_cfg_raw.get("shuffle", True)),
        split_seed_offset=int(pde_cfg_raw.get("split_seed_offset", 1207)),
        dt=_as_optional_float(pde_cfg_raw.get("dt")),
    )


def load_pdebench_reaction_diffusion_data(
    config: GrayScottConfig,
    source_cfg: PDEBenchReactionDiffusionSourceConfig,
    seed: int,
) -> ReactionDiffusionTrajectoryData:
    try:
        import h5py
    except Exception as exc:
        raise ImportError(
            "PDEBench source requires `h5py`. Install it with: python3 -m pip install h5py"
        ) from exc

    dataset_path = Path(source_cfg.file_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"PDEBench file not found: {dataset_path}. "
            "Set data.external.pdebench.file_path to a local HDF5 file."
        )

    requested_train = int(source_cfg.n_train) if int(source_cfg.n_train) > 0 else int(config.n_train_trajectories)
    requested_test = int(source_cfg.n_test) if int(source_cfg.n_test) > 0 else int(config.n_test_trajectories)
    if requested_train <= 0 or requested_test <= 0:
        raise ValueError(
            f"Resolved split counts must be positive (train={requested_train}, test={requested_test})."
        )

    with h5py.File(dataset_path, "r") as handle:
        grouped = bool(source_cfg.sample_grouped) or _is_grouped_samples(handle)
        if grouped:
            (
                train_u,
                train_v,
                test_u,
                test_v,
                dt_from_file,
                raw_shape,
                resolved_layout,
            ) = load_grouped_pdebench_split(
                handle=handle,
                config=config,
                source_cfg=source_cfg,
                requested_train=requested_train,
                requested_test=requested_test,
                seed=seed,
            )
        else:
            (
                train_u,
                train_v,
                test_u,
                test_v,
                dt_from_file,
                raw_shape,
                resolved_layout,
            ) = load_root_pdebench_split(
                handle=handle,
                config=config,
                source_cfg=source_cfg,
                requested_train=requested_train,
                requested_test=requested_test,
                seed=seed,
            )

    min_steps = min(
        min(traj.shape[0] for traj in train_u),
        min(traj.shape[0] for traj in test_u),
    )
    n_snapshots = min(max(2, int(config.n_snapshots)), int(min_steps))
    if n_snapshots < 2:
        raise ValueError("PDEBench trajectories must contain at least 2 time steps.")

    train_data = [
        {
            "u": np.asarray(u[:n_snapshots], dtype=np.float32),
            "v": np.asarray(v[:n_snapshots], dtype=np.float32),
        }
        for u, v in zip(train_u, train_v)
    ]
    test_cases = []
    for u, v in zip(test_u, test_v):
        u_trim = np.asarray(u[:n_snapshots], dtype=np.float32)
        v_trim = np.asarray(v[:n_snapshots], dtype=np.float32)
        test_cases.append(
            {
                "u0": np.asarray(u_trim[0], dtype=np.float32),
                "v0": np.asarray(v_trim[0], dtype=np.float32),
                "u_true": u_trim,
                "v_true": v_trim,
            }
        )

    if source_cfg.dt is not None:
        dt = float(source_cfg.dt) * float(source_cfg.time_stride)
    elif dt_from_file is not None:
        dt = float(dt_from_file) * float(source_cfg.time_stride)
    else:
        dt = float(config.t_final / max(config.n_snapshots - 1, 1))

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Resolved non-positive dt ({dt}) for PDEBench source.")

    metadata = {
        "dataset": "pdebench_hdf5",
        "file_path": str(dataset_path.resolve()),
        "layout": resolved_layout,
        "raw_shape": raw_shape,
        "n_train_loaded": len(train_data),
        "n_test_loaded": len(test_cases),
        "sample_grouped": grouped,
        "time_stride": int(source_cfg.time_stride),
        "spatial_stride": int(source_cfg.spatial_stride),
    }

    return ReactionDiffusionTrajectoryData(
        train_data=train_data,
        test_cases=test_cases,
        dt=float(dt),
        n_snapshots=int(n_snapshots),
        source="pdebench",
        metadata=metadata,
    )
