#!/usr/bin/env python3
"""Process PDEBench Navier-Stokes HDF5 into a run-ready omega dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import h5py
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a processed PDEBench NS file (downsampled + optional velocity->omega conversion) "
            "and auto-patch a Navier-Stokes YAML config."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Source HDF5 path.")
    parser.add_argument("--output", type=str, required=True, help="Destination HDF5 path.")
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="AUTO",
        help="Dataset to process (AUTO picks velocity/tensor/particles).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="AUTO",
        help="Input axis layout (AUTO supports 4D/5D: NTHW/NTHWC).",
    )
    parser.add_argument("--nx", type=int, required=True, help="Target x-resolution.")
    parser.add_argument("--ny", type=int, required=True, help="Target y-resolution.")
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Temporal stride applied before writing output (default: 1).",
    )
    parser.add_argument(
        "--min-timestep",
        type=int,
        default=0,
        help="Inclusive minimum source timestep index to keep before stride (default: 0).",
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=None,
        help=(
            "Optional inclusive max source timestep index to keep before stride "
            "(for example, 500 keeps t=0..500)."
        ),
    )
    parser.add_argument(
        "--convert-to-omega",
        type=str,
        default="auto",
        choices=("auto", "always", "never"),
        help=(
            "Whether to convert velocity to vorticity. "
            "`auto` converts when dataset_key resolves to velocity."
        ),
    )
    parser.add_argument(
        "--output-key",
        type=str,
        default="AUTO",
        help="Output dataset key (AUTO => omega when converted, else input key).",
    )
    parser.add_argument(
        "--lx",
        type=float,
        default=None,
        help="Optional domain length Lx override used for velocity->omega conversion.",
    )
    parser.add_argument(
        "--ly",
        type=float,
        default=None,
        help="Optional domain length Ly override used for velocity->omega conversion.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Fraction of samples assigned to test split when patching config (default: 0.25).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=("gzip", "lzf", "none"),
        help="Output HDF5 compression.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level (ignored unless --compression gzip).",
    )
    parser.add_argument(
        "--copy-time-key",
        type=str,
        default="t",
        help="Time-coordinate dataset key to copy if present (empty string disables).",
    )
    parser.add_argument(
        "--patch-config",
        type=str,
        default="configs/navier_stokes.yaml",
        help=(
            "Navier-Stokes YAML config to patch with processed metadata "
            "(default: configs/navier_stokes.yaml)."
        ),
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        default=False,
        help="Do not patch any YAML config.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        default=False,
        help="Delete source HDF5 after successful write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting output path.",
    )
    return parser.parse_args()


def _resolve_layout(ndim: int, layout: str) -> str:
    normalized = str(layout).strip().upper()
    if normalized in {"", "AUTO"}:
        if ndim == 5:
            return "NTHWC"
        if ndim == 4:
            return "NTHW"
        raise ValueError(f"AUTO layout only supports 4D/5D arrays, got ndim={ndim}.")

    labels = tuple(normalized)
    if len(labels) != ndim:
        raise ValueError(f"Layout '{normalized}' has len={len(labels)} but array ndim={ndim}.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Layout '{normalized}' contains duplicate axes.")
    for label in labels:
        if label not in {"N", "T", "H", "W", "C"}:
            raise ValueError(f"Unsupported layout axis '{label}' in '{normalized}'.")
    if "T" not in labels or "H" not in labels or "W" not in labels:
        raise ValueError(f"Layout must include T/H/W axes. Got '{normalized}'.")
    return normalized


def _pick_dataset_key(handle: h5py.File, requested: str) -> str:
    requested_norm = str(requested).strip()
    keys = list(handle.keys())
    if requested_norm and requested_norm.upper() != "AUTO":
        if requested_norm not in handle:
            raise KeyError(f"Dataset key '{requested_norm}' not found. Available keys: {keys}")
        return requested_norm

    for candidate in ("velocity", "omega", "tensor", "particles", "density"):
        if candidate in handle:
            return candidate
    raise KeyError(f"Could not infer dataset key. Available keys: {keys}")


def _build_compression_kwargs(method: str, level: int) -> Dict[str, Any]:
    if method == "none":
        return {}
    if method == "gzip":
        return {"compression": "gzip", "compression_opts": int(np.clip(level, 0, 9))}
    return {"compression": method}


def _copy_file_attrs(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_dataset_attrs(src_ds: h5py.Dataset, dst_ds: h5py.Dataset) -> None:
    for key, value in src_ds.attrs.items():
        dst_ds.attrs[key] = value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if not np.isfinite(result):
        return None
    return result


def _as_yaml_mapping(text: str) -> Dict[str, Any]:
    parsed = yaml.safe_load(text)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {}


def _read_source_config_blob(handle: h5py.File) -> Dict[str, Any]:
    raw = handle.attrs.get("config")
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    return _as_yaml_mapping(text)


def _infer_domain_lengths(
    source_cfg_blob: Mapping[str, Any],
    override_lx: float | None,
    override_ly: float | None,
) -> tuple[float, float]:
    if override_lx is not None and override_ly is not None:
        return float(override_lx), float(override_ly)

    domain = source_cfg_blob.get("domain_size")
    if isinstance(domain, Sequence) and len(domain) >= 2:
        x_val = _safe_float(domain[0])
        y_val = _safe_float(domain[1])
        if x_val is not None and y_val is not None and x_val > 0.0 and y_val > 0.0:
            lx = float(override_lx) if override_lx is not None else float(x_val)
            ly = float(override_ly) if override_ly is not None else float(y_val)
            return lx, ly

    lx = float(override_lx) if override_lx is not None else float(2.0 * np.pi)
    ly = float(override_ly) if override_ly is not None else float(2.0 * np.pi)
    return lx, ly


def _infer_nu(source_cfg_blob: Mapping[str, Any], src_ds: h5py.Dataset, src_file: h5py.File) -> float | None:
    for key in ("NU", "nu", "viscosity", "kinematic_viscosity"):
        value = _safe_float(source_cfg_blob.get(key))
        if value is not None and value > 0.0:
            return float(value)

    for attrs in (src_ds.attrs, src_file.attrs):
        for key in attrs.keys():
            normalized = str(key).strip().lower()
            if normalized not in {"nu", "viscosity", "kinematic_viscosity"}:
                continue
            value = _safe_float(attrs.get(key))
            if value is not None and value > 0.0:
                return float(value)
    return None


def _infer_dt(
    src_file: h5py.File,
    source_cfg_blob: Mapping[str, Any],
    time_key: str,
    time_stride: int,
) -> float | None:
    if time_key and time_key in src_file:
        values = np.asarray(src_file[time_key], dtype=np.float64)
        if values.ndim > 1:
            values = values.reshape(-1, values.shape[-1])[0]
        if values.size >= 2:
            diffs = np.diff(values)
            finite = diffs[np.isfinite(diffs)]
            if finite.size > 0:
                dt = float(np.median(np.abs(finite)))
                if dt > 0.0:
                    return dt * float(max(1, time_stride))

    dt_raw = _safe_float(source_cfg_blob.get("DT"))
    frame_int = _safe_float(source_cfg_blob.get("frame_int"))
    if dt_raw is not None and dt_raw > 0.0:
        multiplier = frame_int if frame_int is not None and frame_int > 0.0 else 1.0
        return float(dt_raw) * float(multiplier) * float(max(1, time_stride))
    return None


def _velocity_to_vorticity(u: np.ndarray, v: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Convert velocity block [T, X, Y] into vorticity block [T, X, Y]."""
    u_block = np.asarray(u, dtype=np.float64)
    v_block = np.asarray(v, dtype=np.float64)
    if u_block.shape != v_block.shape:
        raise ValueError(f"Velocity component shape mismatch: u={u_block.shape}, v={v_block.shape}")
    if u_block.ndim != 3:
        raise ValueError(f"Expected velocity block with shape [T, X, Y], got {u_block.shape}")

    nx = int(u_block.shape[-2])
    ny = int(u_block.shape[-1])
    dx = float(Lx) / float(nx)
    dy = float(Ly) / float(ny)

    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    u_hat = np.fft.fft2(u_block, axes=(-2, -1))
    v_hat = np.fft.fft2(v_block, axes=(-2, -1))
    dv_dx = np.fft.ifft2((1j * kx_grid) * v_hat, axes=(-2, -1)).real
    du_dy = np.fft.ifft2((1j * ky_grid) * u_hat, axes=(-2, -1)).real
    omega = dv_dx - du_dy
    return np.asarray(omega, dtype=np.float32)


def _target_len(size: int, stride: int) -> int:
    return int(math.ceil(float(size) / float(max(1, stride))))


def _resolve_time_bounds(
    src_nt: int,
    min_timestep: int,
    max_timestep: int | None,
) -> tuple[int, int]:
    min_idx = int(min_timestep)
    if min_idx < 0:
        raise ValueError(f"min_timestep must be >= 0, got {min_idx}")

    if max_timestep is None:
        max_idx = int(src_nt - 1)
    else:
        max_idx = int(max_timestep)
        if max_idx < 0:
            raise ValueError(f"max_timestep must be >= 0, got {max_idx}")

    if min_idx > max_idx:
        raise ValueError(
            f"min_timestep ({min_idx}) must be <= max_timestep ({max_idx})."
        )

    start = int(min(min_idx, max(src_nt - 1, 0)))
    stop = int(min(src_nt, max_idx + 1))
    if stop <= start:
        stop = min(src_nt, start + 1)
    return start, stop


def _resolve_output_key(output_key: str, src_key: str, convert_to_omega: bool) -> str:
    normalized = str(output_key).strip()
    if normalized and normalized.upper() != "AUTO":
        return normalized
    if convert_to_omega:
        return "omega"
    return str(src_key)


def _transpose_to_canonical(block: np.ndarray, block_layout: str) -> np.ndarray:
    axis = {name: idx for idx, name in enumerate(block_layout)}
    if "C" in axis:
        perm = [axis["T"], axis["H"], axis["W"], axis["C"]]
    else:
        perm = [axis["T"], axis["H"], axis["W"]]
    return np.transpose(block, axes=perm)


def _process_dataset(
    src_ds: h5py.Dataset,
    dst_file: h5py.File,
    out_key: str,
    layout: str,
    target_nx: int,
    target_ny: int,
    time_stride: int,
    min_timestep: int,
    max_timestep: int | None,
    convert_to_omega: bool,
    Lx: float,
    Ly: float,
    compression_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    axis = {name: idx for idx, name in enumerate(layout)}
    has_n_axis = "N" in axis
    has_c_axis = "C" in axis
    n_axis = axis["N"] if has_n_axis else -1
    t_axis = axis["T"]
    h_axis = axis["H"]
    w_axis = axis["W"]
    c_axis = axis["C"] if has_c_axis else -1

    src_nx = int(src_ds.shape[h_axis])
    src_ny = int(src_ds.shape[w_axis])
    src_nt = int(src_ds.shape[t_axis])
    n_samples = int(src_ds.shape[n_axis]) if has_n_axis else 1

    if src_nx % target_nx != 0 or src_ny % target_ny != 0:
        raise ValueError(
            f"Source shape HxW={src_nx}x{src_ny} is not divisible by target {target_nx}x{target_ny}."
        )

    stride_x = src_nx // target_nx
    stride_y = src_ny // target_ny
    time_start, time_stop = _resolve_time_bounds(src_nt, min_timestep, max_timestep)
    target_nt = _target_len(max(0, time_stop - time_start), time_stride)

    if convert_to_omega:
        if not has_c_axis or int(src_ds.shape[c_axis]) < 2:
            raise ValueError(
                "velocity->omega conversion requested, but source dataset has no usable channel axis."
            )
        out_shape = (n_samples, target_nt, target_nx, target_ny) if has_n_axis else (target_nt, target_nx, target_ny)
        out_layout = "NTHW" if has_n_axis else "THW"
    else:
        if has_c_axis:
            out_channels = int(src_ds.shape[c_axis])
            out_shape = (
                (n_samples, target_nt, target_nx, target_ny, out_channels)
                if has_n_axis
                else (target_nt, target_nx, target_ny, out_channels)
            )
            out_layout = "NTHWC" if has_n_axis else "THWC"
        else:
            out_shape = (n_samples, target_nt, target_nx, target_ny) if has_n_axis else (target_nt, target_nx, target_ny)
            out_layout = "NTHW" if has_n_axis else "THW"

    dst_ds = dst_file.create_dataset(
        out_key,
        shape=out_shape,
        dtype=np.float32,
        **compression_kwargs,
    )
    _copy_dataset_attrs(src_ds, dst_ds)

    n_count = n_samples if has_n_axis else 1
    for n_idx in range(n_count):
        src_sel: list[object] = [slice(None)] * src_ds.ndim
        if has_n_axis:
            src_sel[n_axis] = n_idx
        src_sel[t_axis] = slice(time_start, time_stop, time_stride)
        src_sel[h_axis] = slice(None, None, stride_x)
        src_sel[w_axis] = slice(None, None, stride_y)

        block = np.asarray(src_ds[tuple(src_sel)], dtype=np.float32)
        block_layout = layout.replace("N", "") if has_n_axis else layout
        canonical = _transpose_to_canonical(block, block_layout)

        if convert_to_omega:
            u = canonical[..., 0]
            v = canonical[..., 1]
            payload = _velocity_to_vorticity(u, v, Lx=Lx, Ly=Ly)
        else:
            payload = np.asarray(canonical, dtype=np.float32)

        if has_n_axis:
            dst_ds[n_idx] = payload
        else:
            dst_ds[...] = payload

    return {
        "source_nx": src_nx,
        "source_ny": src_ny,
        "source_nt": src_nt,
        "source_t_start_inclusive": int(time_start),
        "source_t_stop_exclusive": int(time_stop),
        "source_max_timestep_inclusive": int(time_stop - 1),
        "target_nx": target_nx,
        "target_ny": target_ny,
        "target_nt": target_nt,
        "n_samples": n_samples,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "time_stride": int(time_stride),
        "layout": out_layout,
        "convert_to_omega": bool(convert_to_omega),
    }


def _maybe_copy_time_dataset(
    src: h5py.File,
    dst: h5py.File,
    key: str,
    time_stride: int,
    min_timestep: int,
    max_timestep: int | None,
) -> None:
    if not key:
        return
    if key not in src:
        return
    src_ds = src[key]
    values = np.asarray(src_ds)
    if values.ndim == 0:
        time_start = None
        time_stop = None
    else:
        time_start, time_stop = _resolve_time_bounds(
            int(values.shape[-1]),
            min_timestep=min_timestep,
            max_timestep=max_timestep,
        )
    if values.ndim == 0:
        trimmed = values
    elif values.ndim == 1:
        trimmed = values[time_start:time_stop:time_stride]
    else:
        trimmed = values[
            (slice(None),) * (values.ndim - 1)
            + (slice(time_start, time_stop, time_stride),)
        ]

    out = dst.create_dataset(key, data=trimmed, dtype=src_ds.dtype)
    _copy_dataset_attrs(src_ds, out)


def _clamped_snapshots(existing: Any, available: int) -> int:
    try:
        current = int(existing)
    except Exception:
        current = available
    if current < 2:
        current = available
    return int(max(2, min(current, available)))


def _auto_split_counts(n_samples: int, test_fraction: float) -> tuple[int, int]:
    if n_samples < 2:
        raise ValueError(f"Need at least 2 trajectories to split train/test, got {n_samples}.")
    frac = float(np.clip(test_fraction, 0.05, 0.95))
    n_test = int(round(n_samples * frac))
    n_test = max(1, min(n_samples - 1, n_test))
    n_train = n_samples - n_test
    return int(n_train), int(n_test)


def _patch_yaml_config(path: Path, metadata: Mapping[str, Any], test_fraction: float) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")

    report: Dict[str, Any] = {}

    grid = payload.setdefault("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("Config key 'grid' must be a mapping.")
    grid["nx"] = int(metadata["target_nx"])
    grid["ny"] = int(metadata["target_ny"])
    grid["Lx"] = float(metadata["Lx"])
    grid["Ly"] = float(metadata["Ly"])
    report["grid.nx"] = grid["nx"]
    report["grid.ny"] = grid["ny"]
    report["grid.Lx"] = grid["Lx"]
    report["grid.Ly"] = grid["Ly"]

    time_cfg = payload.setdefault("time", {})
    if not isinstance(time_cfg, dict):
        raise ValueError("Config key 'time' must be a mapping.")
    n_snapshots = _clamped_snapshots(time_cfg.get("n_snapshots"), int(metadata["target_nt"]))
    time_cfg["n_snapshots"] = n_snapshots
    report["time.n_snapshots"] = n_snapshots

    dt = metadata.get("dt")
    dt_value = _safe_float(dt)
    if dt_value is not None and dt_value > 0.0:
        time_cfg["t_final"] = float(dt_value) * float(max(n_snapshots - 1, 1))
        report["time.t_final"] = time_cfg["t_final"]

    physics = payload.setdefault("physics", {})
    if not isinstance(physics, dict):
        raise ValueError("Config key 'physics' must be a mapping.")
    nu_value = _safe_float(metadata.get("nu"))
    if nu_value is not None and nu_value > 0.0:
        physics["nu"] = float(nu_value)
        report["physics.nu"] = physics["nu"]
    forcing = physics.setdefault("forcing", {})
    if not isinstance(forcing, dict):
        forcing = {}
        physics["forcing"] = forcing
    forcing["type"] = "none"
    forcing["amplitude"] = 0.0
    report["physics.forcing.type"] = "none"
    report["physics.forcing.amplitude"] = 0.0

    data = payload.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config key 'data' must be a mapping.")
    n_train, n_test = _auto_split_counts(int(metadata["n_samples"]), test_fraction=test_fraction)
    data["n_train_trajectories"] = int(n_train)
    data["n_test_trajectories"] = int(n_test)
    report["data.n_train_trajectories"] = data["n_train_trajectories"]
    report["data.n_test_trajectories"] = data["n_test_trajectories"]

    external = data.setdefault("external", {})
    if not isinstance(external, dict):
        raise ValueError("Config key 'data.external' must be a mapping.")
    external["source"] = "pdebench"
    report["data.external.source"] = external["source"]

    pde_cfg = external.setdefault("pdebench", {})
    if not isinstance(pde_cfg, dict):
        raise ValueError("Config key 'data.external.pdebench' must be a mapping.")
    pde_cfg["file_path"] = str(Path(str(metadata["output_file"])).resolve())
    pde_cfg["dataset_key"] = str(metadata["output_key"])
    pde_cfg["layout"] = str(metadata["layout"])
    pde_cfg["channel_index"] = 0
    pde_cfg["time_stride"] = 1
    pde_cfg["spatial_stride"] = 1
    pde_cfg["n_train"] = int(n_train)
    pde_cfg["n_test"] = int(n_test)
    if dt_value is not None and dt_value > 0.0:
        pde_cfg["dt"] = float(dt_value)

    report["data.external.pdebench.file_path"] = pde_cfg["file_path"]
    report["data.external.pdebench.dataset_key"] = pde_cfg["dataset_key"]
    report["data.external.pdebench.layout"] = pde_cfg["layout"]
    report["data.external.pdebench.channel_index"] = pde_cfg["channel_index"]
    report["data.external.pdebench.time_stride"] = pde_cfg["time_stride"]
    report["data.external.pdebench.spatial_stride"] = pde_cfg["spatial_stride"]
    report["data.external.pdebench.n_train"] = pde_cfg["n_train"]
    report["data.external.pdebench.n_test"] = pde_cfg["n_test"]
    if "dt" in pde_cfg:
        report["data.external.pdebench.dt"] = pde_cfg["dt"]

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    target_nx = int(args.nx)
    target_ny = int(args.ny)
    time_stride = max(1, int(args.time_stride))
    min_timestep = max(0, int(args.min_timestep))
    max_timestep = None if args.max_timestep is None else int(args.max_timestep)

    if target_nx <= 0 or target_ny <= 0:
        raise ValueError("Target nx and ny must be positive.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression_kwargs = _build_compression_kwargs(args.compression, args.compression_level)
    copy_time_key = str(args.copy_time_key).strip()

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        _copy_file_attrs(src, dst)
        src_key = _pick_dataset_key(src, args.dataset_key)
        src_ds = src[src_key]
        layout = _resolve_layout(src_ds.ndim, args.layout)
        source_cfg_blob = _read_source_config_blob(src)
        Lx, Ly = _infer_domain_lengths(source_cfg_blob, args.lx, args.ly)

        convert_setting = str(args.convert_to_omega).strip().lower()
        if convert_setting == "always":
            convert_to_omega = True
        elif convert_setting == "never":
            convert_to_omega = False
        else:
            convert_to_omega = str(src_key).strip().lower() == "velocity"

        out_key = _resolve_output_key(args.output_key, src_key=src_key, convert_to_omega=convert_to_omega)
        process_meta = _process_dataset(
            src_ds=src_ds,
            dst_file=dst,
            out_key=out_key,
            layout=layout,
            target_nx=target_nx,
            target_ny=target_ny,
            time_stride=time_stride,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            convert_to_omega=convert_to_omega,
            Lx=Lx,
            Ly=Ly,
            compression_kwargs=compression_kwargs,
        )
        _maybe_copy_time_dataset(
            src,
            dst,
            copy_time_key,
            time_stride=time_stride,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
        )

        dt = _infer_dt(src, source_cfg_blob=source_cfg_blob, time_key=copy_time_key, time_stride=time_stride)
        nu = _infer_nu(source_cfg_blob, src_ds=src_ds, src_file=src)

        dst.attrs["processed_source_file"] = str(input_path.resolve())
        dst.attrs["processed_source_dataset_key"] = str(src_key)
        dst.attrs["processed_source_layout"] = str(layout)
        dst.attrs["processed_output_dataset_key"] = str(out_key)
        dst.attrs["processed_output_layout"] = str(process_meta["layout"])
        dst.attrs["processed_convert_to_omega"] = bool(convert_to_omega)
        dst.attrs["processed_target_nx"] = int(target_nx)
        dst.attrs["processed_target_ny"] = int(target_ny)
        dst.attrs["processed_time_stride"] = int(time_stride)
        dst.attrs["processed_min_timestep"] = int(min_timestep)
        if max_timestep is not None:
            dst.attrs["processed_max_timestep"] = int(max_timestep)
        dst.attrs["processed_Lx"] = float(Lx)
        dst.attrs["processed_Ly"] = float(Ly)
        if dt is not None and dt > 0.0:
            dst.attrs["processed_dt"] = float(dt)
        if nu is not None and nu > 0.0:
            dst.attrs["processed_nu"] = float(nu)

    metadata = {
        **process_meta,
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "input_key": str(src_key),
        "output_key": str(out_key),
        "min_timestep": min_timestep,
        "max_timestep": max_timestep,
        "Lx": float(Lx),
        "Ly": float(Ly),
        "dt": dt,
        "nu": nu,
    }

    if not bool(args.skip_config_update):
        cfg_path = Path(args.patch_config).expanduser()
        report = _patch_yaml_config(cfg_path, metadata=metadata, test_fraction=float(args.test_fraction))
        print(f"Patched config: {cfg_path.resolve()}")
        for key, value in report.items():
            print(f"  {key}={value}")

    if args.delete_source:
        input_path.unlink()
        print(f"Deleted source: {input_path.resolve()}")

    print(f"Input: {metadata['input_file']}")
    print(f"Output: {metadata['output_file']}")
    print(f"Input dataset key: {metadata['input_key']}")
    print(f"Output dataset key: {metadata['output_key']}")
    print(f"Output layout: {metadata['layout']}")
    print(f"Source HxW: {metadata['source_nx']}x{metadata['source_ny']}")
    print(f"Target HxW: {metadata['target_nx']}x{metadata['target_ny']}")
    print(f"Source T: {metadata['source_nt']}")
    print(f"Source min timestep kept (inclusive): {metadata['source_t_start_inclusive']}")
    print(f"Source max timestep kept (inclusive): {metadata['source_max_timestep_inclusive']}")
    print(f"Target T: {metadata['target_nt']}")
    print(f"Sample count: {metadata['n_samples']}")
    print(f"Spatial stride: x{metadata['stride_x']}, y{metadata['stride_y']}")
    print(f"Time stride: {metadata['time_stride']}")
    print(f"Converted to omega: {metadata['convert_to_omega']}")
    if metadata["dt"] is not None:
        print(f"Resolved dt: {metadata['dt']}")
    if metadata["nu"] is not None:
        print(f"Resolved nu: {metadata['nu']}")
    print(f"Domain: Lx={metadata['Lx']}, Ly={metadata['Ly']}")


if __name__ == "__main__":
    main()
