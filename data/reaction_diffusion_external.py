"""Reaction-diffusion trajectory adapters for generated and PDEBench datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from data.reaction_diffusion import GrayScottConfig


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
            sample_keys = _sample_group_keys(handle)
            available = len(sample_keys)
            if available < requested_train + requested_test:
                raise ValueError(
                    f"PDEBench file has {available} grouped samples but "
                    f"{requested_train + requested_test} are required."
                )

            indices = np.arange(available, dtype=int)
            if source_cfg.shuffle:
                rng = np.random.default_rng(seed * 1000 + int(source_cfg.split_seed_offset))
                rng.shuffle(indices)
            train_indices = indices[:requested_train]
            test_indices = indices[requested_train : requested_train + requested_test]

            layout_hint = str(source_cfg.layout)
            train_u, train_v = _load_grouped_samples(
                handle=handle,
                sample_keys=sample_keys,
                selected_indices=train_indices.tolist(),
                source_cfg=source_cfg,
                layout_hint=layout_hint,
                target_nx=int(config.nx),
                target_ny=int(config.ny),
            )
            test_u, test_v = _load_grouped_samples(
                handle=handle,
                sample_keys=sample_keys,
                selected_indices=test_indices.tolist(),
                source_cfg=source_cfg,
                layout_hint=layout_hint,
                target_nx=int(config.nx),
                target_ny=int(config.ny),
            )
            dt_from_file = _infer_dt_from_group(handle, sample_keys[0])
            first_group = handle[sample_keys[0]]
            first_dataset_key = str(source_cfg.dataset_key).strip() or "data"
            if first_dataset_key not in first_group:
                if "data" in first_group:
                    first_dataset_key = "data"
                else:
                    first_dataset_key = next(
                        (
                            key
                            for key in first_group.keys()
                            if isinstance(first_group.get(key), h5py.Dataset)
                        ),
                        "",
                    )
            if not first_dataset_key:
                raise ValueError(f"Could not infer dataset key from sample group '{sample_keys[0]}'.")
            raw_shape = tuple(int(dim) for dim in first_group[first_dataset_key].shape)
            resolved_layout = _resolve_layout(len(raw_shape), source_cfg.layout, grouped_hint=True)
        else:
            root_keys = list(handle.keys())
            dataset_key = str(source_cfg.dataset_key).strip() or "data"
            if dataset_key not in handle:
                for candidate in ("data", "tensor", "fields"):
                    if candidate in handle:
                        dataset_key = candidate
                        break
                else:
                    raise KeyError(
                        f"Dataset key '{source_cfg.dataset_key}' not found in {dataset_path}. "
                        f"Available keys: {root_keys}"
                    )

            dataset = handle[dataset_key]
            raw_shape = tuple(int(dim) for dim in dataset.shape)
            resolved_layout = _resolve_layout(dataset.ndim, source_cfg.layout, grouped_hint=False)
            axis_map = {label: idx for idx, label in enumerate(resolved_layout)}
            if "N" not in axis_map:
                raise ValueError(
                    f"Root dataset layout must include N for non-grouped data. Got '{resolved_layout}'."
                )
            available = int(dataset.shape[axis_map["N"]])
            if available < requested_train + requested_test:
                raise ValueError(
                    f"PDEBench dataset has {available} samples but "
                    f"{requested_train + requested_test} are required."
                )
            indices = np.arange(available, dtype=int)
            if source_cfg.shuffle:
                rng = np.random.default_rng(seed * 1000 + int(source_cfg.split_seed_offset))
                rng.shuffle(indices)
            train_indices = indices[:requested_train]
            test_indices = indices[requested_train : requested_train + requested_test]

            train_u, train_v = _load_root_samples(
                dataset=dataset,
                selected_indices=train_indices.tolist(),
                layout=resolved_layout,
                source_cfg=source_cfg,
                target_nx=int(config.nx),
                target_ny=int(config.ny),
            )
            test_u, test_v = _load_root_samples(
                dataset=dataset,
                selected_indices=test_indices.tolist(),
                layout=resolved_layout,
                source_cfg=source_cfg,
                target_nx=int(config.nx),
                target_ny=int(config.ny),
            )
            dt_from_file = _infer_dt_from_root(handle)

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


def _is_grouped_samples(handle: Any) -> bool:
    try:
        import h5py
    except Exception:
        return False
    keys = _sample_group_keys(handle)
    if not keys:
        return False
    first = handle.get(keys[0])
    if first is None:
        return False
    return isinstance(first, h5py.Group) and "data" in first


def _sample_group_keys(handle: Any) -> List[str]:
    try:
        import h5py
    except Exception:
        return []
    return sorted([key for key in handle.keys() if isinstance(handle.get(key), h5py.Group)])


def _load_grouped_samples(
    handle: Any,
    sample_keys: List[str],
    selected_indices: List[int],
    source_cfg: PDEBenchReactionDiffusionSourceConfig,
    layout_hint: str,
    target_nx: int,
    target_ny: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    u_trajectories: List[np.ndarray] = []
    v_trajectories: List[np.ndarray] = []
    for sample_idx in selected_indices:
        key = sample_keys[int(sample_idx)]
        group = handle[key]
        dataset_key = str(source_cfg.dataset_key).strip() or "data"
        if dataset_key not in group:
            if "data" in group:
                dataset_key = "data"
            else:
                raise KeyError(
                    f"Dataset key '{source_cfg.dataset_key}' not found in sample group '{key}'. "
                    f"Available keys: {list(group.keys())}"
                )
        raw = np.asarray(group[dataset_key], dtype=np.float32)
        layout = _resolve_layout(raw.ndim, layout_hint, grouped_hint=True)
        u_traj, v_traj = _array_to_uv_trajectory(
            raw,
            layout=layout,
            u_channel_index=int(source_cfg.u_channel_index),
            v_channel_index=int(source_cfg.v_channel_index),
        )
        u_traj = u_traj[:: source_cfg.time_stride, :: source_cfg.spatial_stride, :: source_cfg.spatial_stride]
        v_traj = v_traj[:: source_cfg.time_stride, :: source_cfg.spatial_stride, :: source_cfg.spatial_stride]
        u_trajectories.append(_match_trajectory_resolution(u_traj, target_nx, target_ny))
        v_trajectories.append(_match_trajectory_resolution(v_traj, target_nx, target_ny))
    return u_trajectories, v_trajectories


def _load_root_samples(
    dataset: Any,
    selected_indices: List[int],
    layout: str,
    source_cfg: PDEBenchReactionDiffusionSourceConfig,
    target_nx: int,
    target_ny: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    axis_map = {label: idx for idx, label in enumerate(layout)}
    n_axis = axis_map["N"]
    layout_wo_n = layout.replace("N", "")

    u_trajectories: List[np.ndarray] = []
    v_trajectories: List[np.ndarray] = []
    for sample_idx in selected_indices:
        selector = [slice(None)] * dataset.ndim
        selector[n_axis] = int(sample_idx)
        raw = np.asarray(dataset[tuple(selector)], dtype=np.float32)
        u_traj, v_traj = _array_to_uv_trajectory(
            raw,
            layout=layout_wo_n,
            u_channel_index=int(source_cfg.u_channel_index),
            v_channel_index=int(source_cfg.v_channel_index),
        )
        u_traj = u_traj[:: source_cfg.time_stride, :: source_cfg.spatial_stride, :: source_cfg.spatial_stride]
        v_traj = v_traj[:: source_cfg.time_stride, :: source_cfg.spatial_stride, :: source_cfg.spatial_stride]
        u_trajectories.append(_match_trajectory_resolution(u_traj, target_nx, target_ny))
        v_trajectories.append(_match_trajectory_resolution(v_traj, target_nx, target_ny))
    return u_trajectories, v_trajectories


def _array_to_uv_trajectory(
    array: np.ndarray,
    layout: str,
    u_channel_index: int,
    v_channel_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(array, dtype=np.float32)
    resolved = _resolve_layout(arr.ndim, layout, grouped_hint=True)
    axis_map = {label: idx for idx, label in enumerate(resolved)}

    required = {"T", "H", "W"}
    if not required.issubset(axis_map):
        raise ValueError(f"PDEBench layout must include T,H,W. Got '{resolved}'.")
    if "C" not in axis_map:
        raise ValueError(
            f"PDEBench reaction-diffusion data requires a channel axis C for u/v. Got '{resolved}'."
        )

    c_axis = axis_map["C"]
    n_channels = int(arr.shape[c_axis])
    if n_channels < 2:
        raise ValueError(f"Expected at least 2 channels for u/v, got C={n_channels}.")

    u_idx = min(max(0, int(u_channel_index)), n_channels - 1)
    v_idx = min(max(0, int(v_channel_index)), n_channels - 1)

    u = np.take(arr, indices=u_idx, axis=c_axis)
    v = np.take(arr, indices=v_idx, axis=c_axis)
    layout_wo_c = resolved.replace("C", "")
    axis_wo_c = {label: idx for idx, label in enumerate(layout_wo_c)}
    perm = [axis_wo_c["T"], axis_wo_c["H"], axis_wo_c["W"]]
    u_traj = np.asarray(np.transpose(u, axes=perm), dtype=np.float32)
    v_traj = np.asarray(np.transpose(v, axes=perm), dtype=np.float32)
    return u_traj, v_traj


def _match_trajectory_resolution(trajectory: np.ndarray, target_nx: int, target_ny: int) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.float32)
    matched = [_match_resolution(frame, target_nx, target_ny) for frame in traj]
    return np.asarray(matched, dtype=np.float32)


def _match_resolution(field: np.ndarray, target_nx: int, target_ny: int) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    if arr.shape == (target_nx, target_ny):
        return arr

    src_nx, src_ny = int(arr.shape[0]), int(arr.shape[1])
    if src_nx % target_nx == 0 and src_ny % target_ny == 0:
        stride_x = src_nx // target_nx
        stride_y = src_ny // target_ny
        return np.asarray(arr[::stride_x, ::stride_y], dtype=np.float32)

    raise ValueError(
        f"Cannot map field shape {arr.shape} to target {(target_nx, target_ny)} "
        "using integer-stride downsampling."
    )


def _resolve_layout(ndim: int, layout: str, grouped_hint: bool) -> str:
    normalized = str(layout).strip().upper()
    if normalized in {"", "AUTO"}:
        if ndim == 5:
            return "NTHWC"
        if ndim == 4:
            return "THWC" if grouped_hint else "NTHW"
        if ndim == 3:
            return "THW"
        raise ValueError(f"AUTO layout supports 3D/4D/5D arrays, got ndim={ndim}.")

    labels = tuple(normalized)
    if len(labels) != ndim:
        raise ValueError(f"Layout '{normalized}' has len={len(labels)} but array ndim={ndim}.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Layout '{normalized}' contains duplicate axes.")
    for label in labels:
        if label not in {"N", "T", "H", "W", "C"}:
            raise ValueError(f"Unsupported layout axis '{label}' in '{normalized}'.")
    return normalized


def _infer_dt_from_group(handle: Any, sample_key: str) -> float | None:
    for key in (f"{sample_key}/grid/t", f"{sample_key}/t"):
        if key not in handle:
            continue
        values = np.asarray(handle[key], dtype=np.float64)
        dt = _dt_from_time_values(values)
        if dt is not None:
            return dt
    return None


def _infer_dt_from_root(handle: Any) -> float | None:
    for key in ("t-coordinate", "time", "t"):
        if key not in handle:
            continue
        values = np.asarray(handle[key], dtype=np.float64)
        dt = _dt_from_time_values(values)
        if dt is not None:
            return dt
    return None


def _dt_from_time_values(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[0]
    diffs = np.diff(arr)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return None
    dt = float(np.median(np.abs(finite)))
    if dt <= 0.0:
        return None
    return dt


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)
