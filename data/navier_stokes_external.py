"""Navier-Stokes trajectory data adapters for generated and external datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from data.navier_stokes import NSConfig, NavierStokes2D
from utils.progress import progress_iter


@dataclass
class NeuralOperatorSourceConfig:
    """Settings for loading Navier-Stokes samples via neuraloperator."""

    data_path: str = "external_data/neuraloperator"
    train_resolution: int = 64
    test_resolution: int = 64
    n_train: int = 0
    n_test: int = 0
    batch_size: int = 32
    test_batch_size: int = 32
    positional_encoding: bool = True
    encode_input: bool = False
    encode_output: bool = False
    channel_dim: int = 1
    channel_index: int = 0
    dt: float | None = None


@dataclass
class PDEBenchSourceConfig:
    """Settings for loading Navier-Stokes-like trajectories from PDEBench HDF5."""

    file_path: str = ""
    dataset_key: str = "tensor"
    layout: str = "AUTO"
    channel_index: int = 0
    time_stride: int = 1
    spatial_stride: int = 1
    n_train: int = 0
    n_test: int = 0
    shuffle: bool = True
    split_seed_offset: int = 1207
    dt: float | None = None


@dataclass
class ExternalNavierStokesDataConfig:
    """Top-level external data configuration."""

    source: str = "generated"
    neuraloperator: NeuralOperatorSourceConfig = field(default_factory=NeuralOperatorSourceConfig)
    pdebench: PDEBenchSourceConfig = field(default_factory=PDEBenchSourceConfig)


@dataclass
class NavierStokesTrajectoryData:
    """Unified trajectory payload consumed by the NS run script."""

    train_trajectories: List[np.ndarray]
    test_trajectories: List[np.ndarray]
    dt: float
    n_snapshots: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def external_data_config_from_yaml(raw_config: Mapping[str, Any]) -> ExternalNavierStokesDataConfig:
    """Parse `data.external` config block with safe defaults."""
    data_cfg = raw_config.get("data", {})
    ext_cfg = data_cfg.get("external", {})
    if not isinstance(ext_cfg, Mapping):
        ext_cfg = {}

    source = _normalize_source(str(ext_cfg.get("source", "generated")))
    neural_raw = ext_cfg.get("neuraloperator", {})
    pde_raw = ext_cfg.get("pdebench", {})
    if not isinstance(neural_raw, Mapping):
        neural_raw = {}
    if not isinstance(pde_raw, Mapping):
        pde_raw = {}

    neural_cfg = NeuralOperatorSourceConfig(
        data_path=str(neural_raw.get("data_path", "external_data/neuraloperator")),
        train_resolution=int(neural_raw.get("train_resolution", 64)),
        test_resolution=int(neural_raw.get("test_resolution", neural_raw.get("train_resolution", 64))),
        n_train=int(neural_raw.get("n_train", 0)),
        n_test=int(neural_raw.get("n_test", 0)),
        batch_size=int(neural_raw.get("batch_size", 32)),
        test_batch_size=int(neural_raw.get("test_batch_size", neural_raw.get("batch_size", 32))),
        positional_encoding=bool(neural_raw.get("positional_encoding", True)),
        encode_input=bool(neural_raw.get("encode_input", False)),
        encode_output=bool(neural_raw.get("encode_output", False)),
        channel_dim=int(neural_raw.get("channel_dim", 1)),
        channel_index=int(neural_raw.get("channel_index", 0)),
        dt=_as_optional_float(neural_raw.get("dt")),
    )
    pde_cfg = PDEBenchSourceConfig(
        file_path=str(pde_raw.get("file_path", "")),
        dataset_key=str(pde_raw.get("dataset_key", "tensor")),
        layout=str(pde_raw.get("layout", "AUTO")),
        channel_index=int(pde_raw.get("channel_index", 0)),
        time_stride=max(1, int(pde_raw.get("time_stride", 1))),
        spatial_stride=max(1, int(pde_raw.get("spatial_stride", 1))),
        n_train=int(pde_raw.get("n_train", 0)),
        n_test=int(pde_raw.get("n_test", 0)),
        shuffle=bool(pde_raw.get("shuffle", True)),
        split_seed_offset=int(pde_raw.get("split_seed_offset", 1207)),
        dt=_as_optional_float(pde_raw.get("dt")),
    )
    return ExternalNavierStokesDataConfig(
        source=source,
        neuraloperator=neural_cfg,
        pdebench=pde_cfg,
    )


def load_navier_stokes_trajectory_data(
    config: NSConfig,
    external_cfg: ExternalNavierStokesDataConfig,
    seed: int,
    show_data_progress: bool = False,
) -> NavierStokesTrajectoryData:
    """Load Navier-Stokes train/test trajectories from requested source."""
    source = _normalize_source(external_cfg.source)

    if source == "generated":
        train, test, dt, metadata = _load_generated(config, seed, show_data_progress)
    elif source == "neuraloperator":
        train, test, dt, metadata = _load_neuraloperator(config, external_cfg.neuraloperator, seed, show_data_progress)
    elif source == "pdebench":
        train, test, dt, metadata = _load_pdebench(config, external_cfg.pdebench, seed)
    else:
        raise ValueError(f"Unsupported data.external.source '{external_cfg.source}'.")

    if not train:
        raise ValueError("Loaded train trajectory set is empty.")
    if not test:
        raise ValueError("Loaded test trajectory set is empty.")

    requested_steps = max(2, int(config.n_snapshots))
    min_steps = min(
        min(traj.shape[0] for traj in train),
        min(traj.shape[0] for traj in test),
    )
    n_snapshots = min(requested_steps, int(min_steps))
    if n_snapshots < 2:
        raise ValueError("Need at least 2 snapshots per trajectory.")

    train_trim = [np.asarray(traj[:n_snapshots], dtype=np.float32) for traj in train]
    test_trim = [np.asarray(traj[:n_snapshots], dtype=np.float32) for traj in test]
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError(f"Resolved non-positive dt ({dt_value}) for source '{source}'.")

    metadata = dict(metadata)
    metadata["requested_n_snapshots"] = requested_steps
    metadata["resolved_n_snapshots"] = n_snapshots

    return NavierStokesTrajectoryData(
        train_trajectories=train_trim,
        test_trajectories=test_trim,
        dt=dt_value,
        n_snapshots=n_snapshots,
        source=source,
        metadata=metadata,
    )


def _load_generated(
    config: NSConfig,
    seed: int,
    show_data_progress: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, Dict[str, Any]]:
    solver = NavierStokes2D(config)
    train_trajectories: List[np.ndarray] = []
    t_save: np.ndarray | None = None
    for idx in progress_iter(
        range(config.n_train_trajectories),
        enabled=show_data_progress,
        desc="Data gen (train)",
        total=config.n_train_trajectories,
    ):
        omega0 = solver.sample_initial_condition(seed=seed * 1000 + idx, index=idx)
        t_save, omega_traj = solver.solve(omega0, config.t_final, config.n_snapshots)
        train_trajectories.append(np.asarray(omega_traj, dtype=np.float32))

    test_trajectories: List[np.ndarray] = []
    for idx in progress_iter(
        range(config.n_test_trajectories),
        enabled=show_data_progress,
        desc="Data gen (test)",
        total=config.n_test_trajectories,
    ):
        omega0 = solver.sample_initial_condition(seed=seed * 1000 + 500 + idx, index=10_000 + idx)
        _, omega_true = solver.solve(omega0, config.t_final, config.n_snapshots)
        test_trajectories.append(np.asarray(omega_true, dtype=np.float32))

    if t_save is None or t_save.size < 2:
        raise ValueError("Generated trajectory times are invalid.")
    dt = float(t_save[1] - t_save[0])
    metadata = {
        "generator": "pseudo_spectral_solver",
        "n_train_requested": int(config.n_train_trajectories),
        "n_test_requested": int(config.n_test_trajectories),
    }
    return train_trajectories, test_trajectories, dt, metadata


def _load_neuraloperator(
    config: NSConfig,
    source_cfg: NeuralOperatorSourceConfig,
    seed: int,
    show_data_progress: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, Dict[str, Any]]:
    try:
        from neuralop.data.datasets import load_navier_stokes_pt
    except Exception as exc:
        raise ImportError(
            "NeuralOperator source requires `neuraloperator`. "
            "Install it with: python3 -m pip install neuraloperator"
        ) from exc

    n_train = int(source_cfg.n_train) if int(source_cfg.n_train) > 0 else int(config.n_train_trajectories)
    n_test = int(source_cfg.n_test) if int(source_cfg.n_test) > 0 else int(config.n_test_trajectories)
    if n_train <= 0 or n_test <= 0:
        raise ValueError("Resolved neuraloperator sample counts must be positive.")

    train_loader, test_loaders, _ = load_navier_stokes_pt(
        data_path=str(source_cfg.data_path),
        train_resolution=int(source_cfg.train_resolution),
        n_train=int(n_train),
        batch_size=max(1, int(source_cfg.batch_size)),
        positional_encoding=bool(source_cfg.positional_encoding),
        test_resolutions=[int(source_cfg.test_resolution)],
        n_tests=[int(n_test)],
        test_batch_sizes=[max(1, int(source_cfg.test_batch_size))],
        encode_input=bool(source_cfg.encode_input),
        encode_output=bool(source_cfg.encode_output),
        channel_dim=int(source_cfg.channel_dim),
    )

    train_trajectories = _collect_neuraloperator_trajectories(
        train_loader,
        expected_count=n_train,
        channel_index=int(source_cfg.channel_index),
        target_nx=int(config.nx),
        target_ny=int(config.ny),
        show_data_progress=show_data_progress,
        progress_desc="Load NO train",
    )

    if not test_loaders:
        raise ValueError("NeuralOperator loader returned no test loaders.")
    test_loader = test_loaders.get(int(source_cfg.test_resolution))
    if test_loader is None:
        test_loader = next(iter(test_loaders.values()))
    test_trajectories = _collect_neuraloperator_trajectories(
        test_loader,
        expected_count=n_test,
        channel_index=int(source_cfg.channel_index),
        target_nx=int(config.nx),
        target_ny=int(config.ny),
        show_data_progress=show_data_progress,
        progress_desc="Load NO test",
    )

    default_dt = float(config.t_final / max(config.n_snapshots - 1, 1))
    dt = source_cfg.dt if source_cfg.dt is not None else default_dt
    metadata = {
        "dataset": "neuraloperator_navier_stokes_pt",
        "data_path": str(Path(source_cfg.data_path).resolve()),
        "train_resolution": int(source_cfg.train_resolution),
        "test_resolution": int(source_cfg.test_resolution),
        "n_train_loaded": len(train_trajectories),
        "n_test_loaded": len(test_trajectories),
        "note": "PT loader is one-step oriented; trajectories are constructed from loader sample x/y pairs.",
    }
    return train_trajectories, test_trajectories, float(dt), metadata


def _collect_neuraloperator_trajectories(
    loader: Any,
    expected_count: int,
    channel_index: int,
    target_nx: int,
    target_ny: int,
    show_data_progress: bool,
    progress_desc: str,
) -> List[np.ndarray]:
    trajectories: List[np.ndarray] = []
    iterator = progress_iter(loader, enabled=show_data_progress, desc=progress_desc, total=None)
    for batch in iterator:
        x_batch, y_batch = _extract_xy_batch(batch)
        batch_trajectories = _batch_to_trajectories(
            x_batch,
            y_batch,
            channel_index=channel_index,
        )
        for traj in batch_trajectories:
            fixed = np.asarray(
                [_match_resolution(frame, target_nx, target_ny) for frame in traj],
                dtype=np.float32,
            )
            trajectories.append(fixed)
            if len(trajectories) >= expected_count:
                return trajectories

    if len(trajectories) < expected_count:
        raise ValueError(
            f"NeuralOperator loader returned {len(trajectories)} samples, expected {expected_count}."
        )
    return trajectories[:expected_count]


def _load_pdebench(
    config: NSConfig,
    source_cfg: PDEBenchSourceConfig,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, Dict[str, Any]]:
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

    with h5py.File(dataset_path, "r") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"PDEBench file has no datasets: {dataset_path}")

        dataset_key = str(source_cfg.dataset_key).strip()
        if not dataset_key:
            dataset_key = "tensor"
        if dataset_key not in handle:
            if dataset_key == "tensor" and "density" in handle:
                dataset_key = "density"
            else:
                raise KeyError(
                    f"Dataset key '{dataset_key}' not found in {dataset_path}. Available keys: {keys}"
                )

        raw = np.asarray(handle[dataset_key], dtype=np.float32)
        dt_from_file = _infer_dt_from_h5(handle)

    trajectories = _pdebench_array_to_trajectories(
        raw,
        layout=str(source_cfg.layout),
        channel_index=int(source_cfg.channel_index),
    )
    trajectories = trajectories[
        :,
        :: max(1, int(source_cfg.time_stride)),
        :: max(1, int(source_cfg.spatial_stride)),
        :: max(1, int(source_cfg.spatial_stride)),
    ]
    if trajectories.shape[1] < 2:
        raise ValueError("PDEBench trajectories must contain at least 2 time steps.")

    requested_train = int(source_cfg.n_train) if int(source_cfg.n_train) > 0 else int(config.n_train_trajectories)
    requested_test = int(source_cfg.n_test) if int(source_cfg.n_test) > 0 else int(config.n_test_trajectories)
    requested_total = requested_train + requested_test
    if trajectories.shape[0] < requested_total:
        raise ValueError(
            f"PDEBench file has {trajectories.shape[0]} samples but "
            f"{requested_total} are required (train={requested_train}, test={requested_test})."
        )

    indices = np.arange(trajectories.shape[0], dtype=int)
    if source_cfg.shuffle:
        rng = np.random.default_rng(seed * 1000 + int(source_cfg.split_seed_offset))
        rng.shuffle(indices)

    train_idx = indices[:requested_train]
    test_idx = indices[requested_train : requested_train + requested_test]

    train_trajectories = [
        np.asarray(
            [_match_resolution(frame, config.nx, config.ny) for frame in trajectories[idx]],
            dtype=np.float32,
        )
        for idx in train_idx
    ]
    test_trajectories = [
        np.asarray(
            [_match_resolution(frame, config.nx, config.ny) for frame in trajectories[idx]],
            dtype=np.float32,
        )
        for idx in test_idx
    ]

    dt_fallback = float(config.t_final / max(config.n_snapshots - 1, 1))
    if source_cfg.dt is not None:
        dt = float(source_cfg.dt) * float(max(1, int(source_cfg.time_stride)))
    elif dt_from_file is not None:
        dt = float(dt_from_file) * float(max(1, int(source_cfg.time_stride)))
    else:
        dt = dt_fallback

    metadata = {
        "dataset": "pdebench_hdf5",
        "file_path": str(dataset_path.resolve()),
        "dataset_key": dataset_key,
        "layout": _resolve_layout(raw.ndim, source_cfg.layout),
        "raw_shape": tuple(int(dim) for dim in raw.shape),
        "loaded_shape": tuple(int(dim) for dim in trajectories.shape),
        "n_train_loaded": len(train_trajectories),
        "n_test_loaded": len(test_trajectories),
    }
    return train_trajectories, test_trajectories, dt, metadata


def _extract_xy_batch(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, Mapping):
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Unsupported loader batch format. Expected mapping with keys x/y or tuple/list.")


def _batch_to_trajectories(x_batch: Any, y_batch: Any, channel_index: int) -> List[np.ndarray]:
    x = _strip_channel_axes(_to_numpy(x_batch), channel_index=channel_index)
    y = _strip_channel_axes(_to_numpy(y_batch), channel_index=channel_index)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched batch size between x and y: {x.shape} vs {y.shape}")

    trajectories: List[np.ndarray] = []
    if x.ndim == 3 and y.ndim == 3:
        for idx in range(x.shape[0]):
            trajectories.append(np.stack([x[idx], y[idx]], axis=0).astype(np.float32))
        return trajectories

    if x.ndim == 4 and y.ndim == 4:
        for idx in range(x.shape[0]):
            trajectories.append(np.concatenate([x[idx], y[idx]], axis=0).astype(np.float32))
        return trajectories

    if x.ndim == 4 and y.ndim == 3:
        for idx in range(x.shape[0]):
            trajectories.append(np.concatenate([x[idx], y[idx][None, ...]], axis=0).astype(np.float32))
        return trajectories

    if x.ndim == 3 and y.ndim == 4:
        for idx in range(x.shape[0]):
            trajectories.append(np.concatenate([x[idx][None, ...], y[idx]], axis=0).astype(np.float32))
        return trajectories

    raise ValueError(
        "Unsupported x/y tensor rank combination after channel stripping: "
        f"x.ndim={x.ndim}, y.ndim={y.ndim}"
    )


def _strip_channel_axes(array: np.ndarray, channel_index: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim <= 3:
        return arr

    idx = max(0, int(channel_index))
    if arr.ndim == 4:
        # Common cases: BCHW or BHWC.
        if arr.shape[1] <= 8 and arr.shape[2] > 8 and arr.shape[3] > 8:
            ch = min(idx, arr.shape[1] - 1)
            return arr[:, ch, :, :]
        if arr.shape[-1] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            ch = min(idx, arr.shape[-1] - 1)
            return arr[..., ch]
        return arr

    if arr.ndim == 5:
        # Common cases: BTHWC, BTCHW, BCTHW.
        if arr.shape[-1] <= 8:
            ch = min(idx, arr.shape[-1] - 1)
            return arr[..., ch]
        if arr.shape[2] <= 8 and arr.shape[3] > 8 and arr.shape[4] > 8:
            ch = min(idx, arr.shape[2] - 1)
            return arr[:, :, ch, :, :]
        if arr.shape[1] <= 8 and arr.shape[3] > 8 and arr.shape[4] > 8:
            ch = min(idx, arr.shape[1] - 1)
            return arr[:, ch, :, :, :]
        return arr

    return arr


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


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


def _pdebench_array_to_trajectories(array: np.ndarray, layout: str, channel_index: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    resolved_layout = _resolve_layout(arr.ndim, layout)
    axis_map = {label: idx for idx, label in enumerate(resolved_layout)}

    required = {"N", "T", "H", "W"}
    if not required.issubset(axis_map):
        raise ValueError(f"PDEBench layout must include N,T,H,W. Got '{resolved_layout}'.")

    if "C" in axis_map:
        ch = min(max(0, int(channel_index)), arr.shape[axis_map["C"]] - 1)
        arr = np.take(arr, indices=ch, axis=axis_map["C"])
        resolved_layout = resolved_layout.replace("C", "")
        axis_map = {label: idx for idx, label in enumerate(resolved_layout)}

    perm = [axis_map["N"], axis_map["T"], axis_map["H"], axis_map["W"]]
    arr = np.transpose(arr, axes=perm)
    if arr.ndim != 4:
        raise ValueError(f"Expected NTHW after transpose, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def _resolve_layout(ndim: int, layout: str) -> str:
    normalized = str(layout).strip().upper()
    if normalized in {"", "AUTO"}:
        if ndim == 5:
            return "NTHWC"
        if ndim == 4:
            return "NTHW"
        raise ValueError(f"AUTO layout is only supported for 4D/5D arrays, got ndim={ndim}.")

    labels = tuple(normalized)
    if len(labels) != ndim:
        raise ValueError(f"Layout '{normalized}' has len={len(labels)} but array ndim={ndim}.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Layout '{normalized}' contains duplicate axes.")
    for label in labels:
        if label not in {"N", "T", "H", "W", "C"}:
            raise ValueError(f"Unsupported layout axis '{label}' in '{normalized}'.")
    return normalized


def _infer_dt_from_h5(handle: Any) -> float | None:
    for key in ("t-coordinate", "time", "t"):
        if key not in handle:
            continue
        values = np.asarray(handle[key], dtype=np.float64)
        if values.size < 2:
            continue
        if values.ndim > 1:
            values = values.reshape(-1, values.shape[-1])[0]
        diffs = np.diff(values)
        finite = diffs[np.isfinite(diffs)]
        if finite.size == 0:
            continue
        dt = float(np.median(np.abs(finite)))
        if dt > 0.0:
            return dt
    return None


def _normalize_source(source: str) -> str:
    normalized = str(source).strip().lower().replace("-", "_")
    alias_map = {
        "generated": "generated",
        "solver": "generated",
        "synthetic": "generated",
        "neuraloperator": "neuraloperator",
        "neural_operator": "neuraloperator",
        "neuralop": "neuraloperator",
        "pdebench": "pdebench",
        "pde_bench": "pdebench",
    }
    if normalized not in alias_map:
        supported = ", ".join(sorted(set(alias_map.values())))
        raise ValueError(f"Unsupported external data source '{source}'. Use one of: {supported}")
    return alias_map[normalized]


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)
