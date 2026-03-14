"""Navier-Stokes trajectory data adapters for generated and external datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from utils.progress import progress_iter

from .helper.external_io import (
    _as_optional_float,
    _as_string_list,
    _collect_neuraloperator_trajectories,
    _infer_dt_from_h5,
    _match_resolution,
    _normalize_source,
    _pdebench_array_to_trajectories,
    _resolve_layout,
    _resolve_pdebench_input_paths,
    _velocity_to_vorticity_trajectories,
)
from .solver import NSConfig, NavierStokes2D


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
    file_paths: List[str] = field(default_factory=list)
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
class FNOMATSourceConfig:
    """Settings for loading full-trajectory FNO MAT Navier-Stokes data."""

    file_path: str = "external_data/fno/NavierStokes_V1e-3_N5000_T50.mat"
    dataset_key: str = "u"
    time_key: str = "t"
    layout: str = "AUTO"
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
    fno_mat: FNOMATSourceConfig = field(default_factory=FNOMATSourceConfig)


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
    fno_mat_raw = ext_cfg.get("fno_mat", {})
    if not isinstance(neural_raw, Mapping):
        neural_raw = {}
    if not isinstance(pde_raw, Mapping):
        pde_raw = {}
    if not isinstance(fno_mat_raw, Mapping):
        fno_mat_raw = {}

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
        file_paths=_as_string_list(pde_raw.get("file_paths")),
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
    fno_mat_cfg = FNOMATSourceConfig(
        file_path=str(
            fno_mat_raw.get(
                "file_path",
                "external_data/fno/NavierStokes_V1e-3_N5000_T50.mat",
            )
        ),
        dataset_key=str(fno_mat_raw.get("dataset_key", "u")),
        time_key=str(fno_mat_raw.get("time_key", "t")),
        layout=str(fno_mat_raw.get("layout", "AUTO")),
        time_stride=max(1, int(fno_mat_raw.get("time_stride", 1))),
        spatial_stride=max(1, int(fno_mat_raw.get("spatial_stride", 1))),
        n_train=int(fno_mat_raw.get("n_train", 0)),
        n_test=int(fno_mat_raw.get("n_test", 0)),
        shuffle=bool(fno_mat_raw.get("shuffle", True)),
        split_seed_offset=int(fno_mat_raw.get("split_seed_offset", 1207)),
        dt=_as_optional_float(fno_mat_raw.get("dt")),
    )
    return ExternalNavierStokesDataConfig(
        source=source,
        neuraloperator=neural_cfg,
        pdebench=pde_cfg,
        fno_mat=fno_mat_cfg,
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
    elif source == "fno_mat":
        train, test, dt, metadata = _load_fno_mat(config, external_cfg.fno_mat, seed)
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

    time_stride = max(1, int(source_cfg.time_stride))
    spatial_stride = max(1, int(source_cfg.spatial_stride))
    dataset_paths = _resolve_pdebench_input_paths(source_cfg.file_path, source_cfg.file_paths)

    all_trajectories: List[np.ndarray] = []
    dataset_keys_used: List[str] = []
    layouts_used: List[str] = []
    converted_count = 0
    dt_from_files: List[float] = []
    per_file_shapes: Dict[str, Dict[str, Any]] = {}

    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError(f"PDEBench file has no datasets: {dataset_path}")

            dataset_key = str(source_cfg.dataset_key).strip()
            if not dataset_key:
                dataset_key = "tensor"
            if dataset_key not in handle:
                if dataset_key == "tensor":
                    for candidate in ("velocity", "density", "particles"):
                        if candidate in handle:
                            dataset_key = candidate
                            break
                    else:
                        raise KeyError(
                            f"Dataset key '{dataset_key}' not found in {dataset_path}. Available keys: {keys}"
                        )
                else:
                    raise KeyError(
                        f"Dataset key '{dataset_key}' not found in {dataset_path}. Available keys: {keys}"
                    )

            raw = np.array(handle[dataset_key], dtype=np.float32, copy=True)
            dt_from_file = _infer_dt_from_h5(handle)

        keep_channels = str(dataset_key).strip().lower() == "velocity"
        trajectories = _pdebench_array_to_trajectories(
            raw,
            layout=str(source_cfg.layout),
            channel_index=int(source_cfg.channel_index),
            keep_channels=keep_channels,
        )
        trajectories = trajectories[:, ::time_stride, ::spatial_stride, ::spatial_stride, ...]
        if trajectories.ndim == 5:
            trajectories = _velocity_to_vorticity_trajectories(
                trajectories,
                Lx=float(config.Lx),
                Ly=float(config.Ly),
            )
            converted_count += 1
        if trajectories.shape[1] < 2:
            raise ValueError(f"PDEBench trajectories in {dataset_path} must contain at least 2 time steps.")

        for idx in range(trajectories.shape[0]):
            all_trajectories.append(
                np.asarray(
                    [_match_resolution(frame, config.nx, config.ny) for frame in trajectories[idx]],
                    dtype=np.float32,
                )
            )

        dataset_keys_used.append(str(dataset_key))
        layouts_used.append(_resolve_layout(raw.ndim, source_cfg.layout))
        if dt_from_file is not None:
            dt_from_files.append(float(dt_from_file) * float(time_stride))
        per_file_shapes[str(dataset_path.resolve())] = {
            "raw_shape": tuple(int(dim) for dim in raw.shape),
            "loaded_shape": tuple(int(dim) for dim in trajectories.shape),
        }

    requested_train = int(source_cfg.n_train) if int(source_cfg.n_train) > 0 else int(config.n_train_trajectories)
    requested_test = int(source_cfg.n_test) if int(source_cfg.n_test) > 0 else int(config.n_test_trajectories)
    requested_total = requested_train + requested_test
    total_samples = len(all_trajectories)
    if total_samples < requested_total:
        raise ValueError(
            f"PDEBench sources provide {total_samples} samples but "
            f"{requested_total} are required (train={requested_train}, test={requested_test})."
        )

    indices = np.arange(total_samples, dtype=int)
    if source_cfg.shuffle:
        rng = np.random.default_rng(seed * 1000 + int(source_cfg.split_seed_offset))
        rng.shuffle(indices)

    train_idx = indices[:requested_train]
    test_idx = indices[requested_train : requested_train + requested_test]

    train_trajectories = [all_trajectories[int(idx)] for idx in train_idx]
    test_trajectories = [all_trajectories[int(idx)] for idx in test_idx]

    dt_fallback = float(config.t_final / max(config.n_snapshots - 1, 1))
    if source_cfg.dt is not None:
        dt = float(source_cfg.dt) * float(time_stride)
    elif dt_from_files:
        dt = float(np.median(np.asarray(dt_from_files, dtype=np.float64)))
    else:
        dt = dt_fallback

    unique_dataset_keys = sorted(set(dataset_keys_used))
    unique_layouts = sorted(set(layouts_used))
    if converted_count == len(dataset_paths):
        field_representation = "vorticity_from_velocity"
    elif converted_count == 0:
        field_representation = "scalar_from_dataset"
    else:
        field_representation = "mixed"

    metadata = {
        "dataset": "pdebench_hdf5",
        "file_path": str(dataset_paths[0].resolve()) if dataset_paths else "",
        "file_paths": [str(path.resolve()) for path in dataset_paths],
        "n_files": len(dataset_paths),
        "dataset_key": unique_dataset_keys[0] if len(unique_dataset_keys) == 1 else unique_dataset_keys,
        "layout": unique_layouts[0] if len(unique_layouts) == 1 else unique_layouts,
        "per_file_shapes": per_file_shapes,
        "field_representation": field_representation,
        "converted_from_velocity": bool(converted_count > 0),
        "vorticity_formula": "omega = dv/dx - du/dy" if converted_count > 0 else None,
        "velocity_channels_used": [0, 1] if converted_count > 0 else None,
        "vorticity_domain": {"Lx": float(config.Lx), "Ly": float(config.Ly)} if converted_count > 0 else None,
        "storage": "eager_host_memory",
        "n_total_loaded": total_samples,
        "n_train_loaded": len(train_trajectories),
        "n_test_loaded": len(test_trajectories),
    }
    return train_trajectories, test_trajectories, dt, metadata


def _load_fno_mat(
    config: NSConfig,
    source_cfg: FNOMATSourceConfig,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, Dict[str, Any]]:
    dataset_path = Path(source_cfg.file_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"FNO MAT file not found: {dataset_path}. "
            "Set data.external.fno_mat.file_path to a local MAT file."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"FNO MAT path is not a file: {dataset_path}")

    requested_train = int(source_cfg.n_train) if int(source_cfg.n_train) > 0 else int(config.n_train_trajectories)
    requested_test = int(source_cfg.n_test) if int(source_cfg.n_test) > 0 else int(config.n_test_trajectories)
    if requested_train <= 0 or requested_test <= 0:
        raise ValueError(
            f"Resolved split counts must be positive (train={requested_train}, test={requested_test})."
        )

    h5_error: Exception | None = None
    try:
        train_trajectories, test_trajectories, dt_from_file, metadata = _load_fno_mat_h5(
            dataset_path,
            source_cfg,
            requested_train=requested_train,
            requested_test=requested_test,
            seed=seed,
            target_nx=int(config.nx),
            target_ny=int(config.ny),
        )
    except (OSError, ImportError) as exc:
        h5_error = exc
        train_trajectories, test_trajectories, dt_from_file, metadata = _load_fno_mat_scipy(
            dataset_path,
            source_cfg,
            requested_train=requested_train,
            requested_test=requested_test,
            seed=seed,
            target_nx=int(config.nx),
            target_ny=int(config.ny),
        )
        metadata["h5py_open_error"] = str(h5_error)

    dt_fallback = float(config.t_final / max(config.n_snapshots - 1, 1))
    if source_cfg.dt is not None:
        dt = float(source_cfg.dt) * float(source_cfg.time_stride)
    elif dt_from_file is not None:
        dt = float(dt_from_file)
    else:
        dt = dt_fallback

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Resolved non-positive dt ({dt}) for FNO MAT source.")
    return train_trajectories, test_trajectories, float(dt), metadata


def _load_fno_mat_h5(
    dataset_path: Path,
    source_cfg: FNOMATSourceConfig,
    requested_train: int,
    requested_test: int,
    seed: int,
    target_nx: int,
    target_ny: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], float | None, Dict[str, Any]]:
    try:
        import h5py
    except Exception as exc:
        raise ImportError(
            "FNO MAT HDF5 loading requires `h5py`. Install it with: python3 -m pip install h5py"
        ) from exc

    dataset_key = str(source_cfg.dataset_key).strip() or "u"
    time_key = str(source_cfg.time_key).strip() or "t"
    time_stride = max(1, int(source_cfg.time_stride))
    spatial_stride = max(1, int(source_cfg.spatial_stride))
    requested_total = requested_train + requested_test

    with h5py.File(dataset_path, "r") as handle:
        keys = list(handle.keys())
        if dataset_key not in handle:
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in {dataset_path}. Available keys: {keys}"
            )

        u_ds = handle[dataset_key]
        if not isinstance(u_ds, h5py.Dataset):
            raise ValueError(f"MAT key '{dataset_key}' in {dataset_path} is not a dataset.")
        if u_ds.ndim != 4:
            raise ValueError(
                f"FNO MAT dataset '{dataset_key}' must have rank 4, got shape {tuple(u_ds.shape)}."
            )

        time_values = handle[time_key] if time_key in handle else None
        time_length_hint = _extract_time_length_hint(time_values)
        resolved_layout = _resolve_fno_mat_layout(
            tuple(int(dim) for dim in u_ds.shape),
            source_cfg.layout,
            time_length_hint=time_length_hint,
        )
        axis_map = {label: idx for idx, label in enumerate(resolved_layout)}
        n_available = int(u_ds.shape[axis_map["N"]])
        if n_available < requested_total:
            raise ValueError(
                f"FNO MAT file provides {n_available} samples but "
                f"{requested_total} are required (train={requested_train}, test={requested_test})."
            )

        train_idx, test_idx = _split_sample_indices(
            n_available=n_available,
            requested_train=requested_train,
            requested_test=requested_test,
            shuffle=bool(source_cfg.shuffle),
            seed=seed,
            split_seed_offset=int(source_cfg.split_seed_offset),
        )

        train_trajectories = [
            _extract_fno_mat_trajectory(
                u_ds,
                layout=resolved_layout,
                sample_index=int(idx),
                time_stride=time_stride,
                spatial_stride=spatial_stride,
                target_nx=target_nx,
                target_ny=target_ny,
            )
            for idx in train_idx
        ]
        test_trajectories = [
            _extract_fno_mat_trajectory(
                u_ds,
                layout=resolved_layout,
                sample_index=int(idx),
                time_stride=time_stride,
                spatial_stride=spatial_stride,
                target_nx=target_nx,
                target_ny=target_ny,
            )
            for idx in test_idx
        ]

        n_steps_raw = int(u_ds.shape[axis_map["T"]])
        dt_from_file = _infer_dt_from_mat_time(
            time_values,
            n_steps_raw=n_steps_raw,
            time_stride=time_stride,
        )

        metadata = {
            "dataset": "fno_mat",
            "file_path": str(dataset_path.resolve()),
            "dataset_key": dataset_key,
            "time_key": time_key if time_key else None,
            "layout": resolved_layout,
            "raw_shape": tuple(int(dim) for dim in u_ds.shape),
            "time_stride": int(time_stride),
            "spatial_stride": int(spatial_stride),
            "n_total_available": int(n_available),
            "n_train_loaded": len(train_trajectories),
            "n_test_loaded": len(test_trajectories),
            "backend": "h5py",
        }

    return train_trajectories, test_trajectories, dt_from_file, metadata


def _load_fno_mat_scipy(
    dataset_path: Path,
    source_cfg: FNOMATSourceConfig,
    requested_train: int,
    requested_test: int,
    seed: int,
    target_nx: int,
    target_ny: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], float | None, Dict[str, Any]]:
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise ImportError(
            "FNO MAT fallback loader requires `scipy`. Install it with: python3 -m pip install scipy"
        ) from exc

    dataset_key = str(source_cfg.dataset_key).strip() or "u"
    time_key = str(source_cfg.time_key).strip() or "t"
    time_stride = max(1, int(source_cfg.time_stride))
    spatial_stride = max(1, int(source_cfg.spatial_stride))
    requested_total = requested_train + requested_test

    variable_names = [dataset_key]
    if time_key and time_key != dataset_key:
        variable_names.append(time_key)
    payload = loadmat(dataset_path, variable_names=variable_names)
    if dataset_key not in payload:
        keys = sorted(k for k in payload.keys() if not k.startswith("__"))
        raise KeyError(
            f"Dataset key '{dataset_key}' not found in {dataset_path}. Available keys: {keys}"
        )

    u_arr = np.asarray(payload[dataset_key], dtype=np.float32)
    if u_arr.ndim != 4:
        raise ValueError(
            f"FNO MAT array '{dataset_key}' must have rank 4, got shape {tuple(u_arr.shape)}."
        )

    time_values = payload.get(time_key)
    time_length_hint = _extract_time_length_hint(time_values)
    resolved_layout = _resolve_fno_mat_layout(
        tuple(int(dim) for dim in u_arr.shape),
        source_cfg.layout,
        time_length_hint=time_length_hint,
    )
    axis_map = {label: idx for idx, label in enumerate(resolved_layout)}
    n_available = int(u_arr.shape[axis_map["N"]])
    if n_available < requested_total:
        raise ValueError(
            f"FNO MAT file provides {n_available} samples but "
            f"{requested_total} are required (train={requested_train}, test={requested_test})."
        )

    train_idx, test_idx = _split_sample_indices(
        n_available=n_available,
        requested_train=requested_train,
        requested_test=requested_test,
        shuffle=bool(source_cfg.shuffle),
        seed=seed,
        split_seed_offset=int(source_cfg.split_seed_offset),
    )

    train_trajectories = [
        _extract_fno_mat_trajectory(
            u_arr,
            layout=resolved_layout,
            sample_index=int(idx),
            time_stride=time_stride,
            spatial_stride=spatial_stride,
            target_nx=target_nx,
            target_ny=target_ny,
        )
        for idx in train_idx
    ]
    test_trajectories = [
        _extract_fno_mat_trajectory(
            u_arr,
            layout=resolved_layout,
            sample_index=int(idx),
            time_stride=time_stride,
            spatial_stride=spatial_stride,
            target_nx=target_nx,
            target_ny=target_ny,
        )
        for idx in test_idx
    ]

    n_steps_raw = int(u_arr.shape[axis_map["T"]])
    dt_from_file = _infer_dt_from_mat_time(
        time_values,
        n_steps_raw=n_steps_raw,
        time_stride=time_stride,
    )

    metadata = {
        "dataset": "fno_mat",
        "file_path": str(dataset_path.resolve()),
        "dataset_key": dataset_key,
        "time_key": time_key if time_key else None,
        "layout": resolved_layout,
        "raw_shape": tuple(int(dim) for dim in u_arr.shape),
        "time_stride": int(time_stride),
        "spatial_stride": int(spatial_stride),
        "n_total_available": int(n_available),
        "n_train_loaded": len(train_trajectories),
        "n_test_loaded": len(test_trajectories),
        "backend": "scipy.io.loadmat",
    }
    return train_trajectories, test_trajectories, dt_from_file, metadata


def _split_sample_indices(
    n_available: int,
    requested_train: int,
    requested_test: int,
    shuffle: bool,
    seed: int,
    split_seed_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    requested_total = int(requested_train) + int(requested_test)
    if int(n_available) < requested_total:
        raise ValueError(
            f"Requested train/test split ({requested_total}) exceeds available samples ({n_available})."
        )

    indices = np.arange(int(n_available), dtype=int)
    if bool(shuffle):
        rng = np.random.default_rng(int(seed) * 1000 + int(split_seed_offset))
        rng.shuffle(indices)
    train_idx = indices[: int(requested_train)]
    test_idx = indices[int(requested_train) : requested_total]
    return train_idx, test_idx


def _extract_fno_mat_trajectory(
    dataset: Any,
    layout: str,
    sample_index: int,
    time_stride: int,
    spatial_stride: int,
    target_nx: int,
    target_ny: int,
) -> np.ndarray:
    axis_map = {label: idx for idx, label in enumerate(layout)}
    if set(axis_map.keys()) != {"N", "T", "H", "W"}:
        raise ValueError(f"FNO MAT layout must be a permutation of N,T,H,W. Got '{layout}'.")

    slicer = [slice(None)] * 4
    slicer[axis_map["N"]] = int(sample_index)
    sample = np.asarray(dataset[tuple(slicer)], dtype=np.float32)
    if sample.ndim != 3:
        raise ValueError(
            f"Expected one-sample slice to be rank 3, got shape {tuple(sample.shape)} for sample {sample_index}."
        )

    remaining_labels = [label for label in layout if label != "N"]
    transpose_order = (
        remaining_labels.index("T"),
        remaining_labels.index("H"),
        remaining_labels.index("W"),
    )
    traj = np.transpose(sample, axes=transpose_order)
    traj = np.asarray(traj[:: int(time_stride), :: int(spatial_stride), :: int(spatial_stride)], dtype=np.float32)
    if traj.shape[0] < 2:
        raise ValueError(
            "Resolved FNO MAT trajectory has fewer than 2 time steps "
            f"(shape={tuple(traj.shape)}, time_stride={time_stride})."
        )

    if traj.shape[1:] != (int(target_nx), int(target_ny)):
        traj = np.asarray(
            [_match_resolution(frame, int(target_nx), int(target_ny)) for frame in traj],
            dtype=np.float32,
        )
    return traj


def _resolve_fno_mat_layout(
    shape: Tuple[int, int, int, int],
    layout: str,
    time_length_hint: int | None = None,
) -> str:
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 4:
        raise ValueError(f"FNO MAT layout resolver expects rank-4 shape, got {dims}.")

    normalized = str(layout).strip().upper()
    required_axes = {"N", "T", "H", "W"}
    if normalized not in {"", "AUTO"}:
        resolved = _resolve_layout(4, normalized)
        if set(resolved) != required_axes:
            raise ValueError(
                f"FNO MAT layout must be a permutation of N,T,H,W. Got '{resolved}'."
            )
        return resolved

    spatial_pair: Tuple[int, int] | None = None
    best_pair_dim = -1
    all_axes = list(range(4))
    for left_idx in range(len(all_axes)):
        for right_idx in range(left_idx + 1, len(all_axes)):
            axis_left = all_axes[left_idx]
            axis_right = all_axes[right_idx]
            dim_left = dims[axis_left]
            dim_right = dims[axis_right]
            if dim_left == dim_right and dim_left > best_pair_dim:
                spatial_pair = (axis_left, axis_right)
                best_pair_dim = dim_left

    if spatial_pair is not None:
        h_axis, w_axis = sorted((int(spatial_pair[0]), int(spatial_pair[1])))
        remaining = [axis for axis in range(4) if axis not in spatial_pair]
        if len(remaining) != 2:
            raise ValueError(f"Could not infer FNO MAT layout from shape {dims}.")
        axis_a, axis_b = int(remaining[0]), int(remaining[1])
        dim_a, dim_b = int(dims[axis_a]), int(dims[axis_b])

        t_axis: int
        if time_length_hint is not None and dim_a == int(time_length_hint) and dim_b != int(time_length_hint):
            t_axis = axis_a
        elif time_length_hint is not None and dim_b == int(time_length_hint) and dim_a != int(time_length_hint):
            t_axis = axis_b
        elif dim_a > 512 >= dim_b:
            t_axis = axis_b
        elif dim_b > 512 >= dim_a:
            t_axis = axis_a
        else:
            t_axis = axis_a if dim_a >= dim_b else axis_b
        n_axis = axis_b if t_axis == axis_a else axis_a
    else:
        max_dim = max(dims)
        max_axes = [axis for axis, dim in enumerate(dims) if dim == max_dim]
        n_axis = int(max_axes[0])
        remaining = [axis for axis in range(4) if axis != n_axis]
        ranked = sorted(remaining, key=lambda axis: int(dims[axis]), reverse=True)
        h_axis, w_axis = sorted((int(ranked[0]), int(ranked[1])))
        t_candidates = [axis for axis in remaining if axis not in {h_axis, w_axis}]
        if len(t_candidates) != 1:
            raise ValueError(f"Could not infer FNO MAT layout from shape {dims}.")
        t_axis = int(t_candidates[0])

    labels = [""] * 4
    labels[n_axis] = "N"
    labels[t_axis] = "T"
    labels[h_axis] = "H"
    labels[w_axis] = "W"
    resolved = "".join(labels)
    if set(resolved) != required_axes:
        raise ValueError(f"Failed to infer valid FNO MAT layout from shape {dims}: '{resolved}'.")
    return resolved


def _extract_time_length_hint(time_values: Any) -> int | None:
    if time_values is None:
        return None
    arr = np.asarray(time_values, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        return None
    if arr.size < 2:
        return None
    return int(arr.shape[0])


def _infer_dt_from_mat_time(
    time_values: Any,
    n_steps_raw: int,
    time_stride: int,
) -> float | None:
    if time_values is None:
        return None

    arr = np.asarray(time_values, dtype=np.float64)
    if arr.size < 2:
        return None

    candidates: List[np.ndarray] = []
    if arr.ndim == 1:
        candidates.append(arr.reshape(-1))
    else:
        for axis, axis_size in enumerate(arr.shape):
            if int(axis_size) != int(n_steps_raw):
                continue
            index = [0] * arr.ndim
            index[axis] = slice(None)
            seq = np.asarray(arr[tuple(index)], dtype=np.float64).reshape(-1)
            if seq.size >= 2:
                candidates.append(seq)
    if not candidates and arr.size == int(n_steps_raw):
        candidates.append(arr.reshape(-1))

    for sequence in candidates:
        dt = _infer_dt_from_time_sequence(sequence, time_stride=time_stride)
        if dt is not None:
            return dt
    return None


def _infer_dt_from_time_sequence(sequence: np.ndarray, time_stride: int) -> float | None:
    values = np.asarray(sequence, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return None

    diffs = np.diff(values)
    finite = diffs[np.isfinite(diffs)]
    finite = finite[np.abs(finite) > 0.0]
    if finite.size == 0:
        return None

    dt = float(np.median(np.abs(finite))) * float(max(1, int(time_stride)))
    if np.isfinite(dt) and dt > 0.0:
        return dt
    return None
