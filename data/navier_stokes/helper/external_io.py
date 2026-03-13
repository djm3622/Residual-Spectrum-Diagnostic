"""Helper functions for Navier-Stokes external-data adapters."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from utils.progress import progress_iter


def _resolve_pdebench_input_paths(file_path: str, file_paths: Sequence[str]) -> List[Path]:
    file_specs: List[str] = []
    primary = str(file_path).strip()
    if primary:
        file_specs.append(primary)
    file_specs.extend(_as_string_list(file_paths))
    file_specs = [spec for spec in file_specs if spec]
    if not file_specs:
        raise ValueError(
            "No PDEBench files configured. Set data.external.pdebench.file_path, "
            "or provide a list/glob via data.external.pdebench.file_paths."
        )

    resolved: List[Path] = []
    seen: set[str] = set()
    unmatched_globs: List[str] = []
    for spec in file_specs:
        expanded = str(Path(spec).expanduser())
        if any(ch in expanded for ch in ("*", "?", "[")):
            matches = sorted(glob.glob(expanded))
            if not matches:
                unmatched_globs.append(spec)
                continue
        else:
            matches = [expanded]

        for match in matches:
            path = Path(match).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"PDEBench file not found: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"PDEBench path is not a file: {path}")
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)

    if not resolved:
        if unmatched_globs:
            raise FileNotFoundError(
                "No PDEBench files resolved. Unmatched glob(s): " + ", ".join(unmatched_globs)
            )
        raise FileNotFoundError("No PDEBench files resolved from configured paths/globs.")
    return resolved


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
        if arr.shape[1] <= 8 and arr.shape[2] > 8 and arr.shape[3] > 8:
            ch = min(idx, arr.shape[1] - 1)
            return arr[:, ch, :, :]
        if arr.shape[-1] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            ch = min(idx, arr.shape[-1] - 1)
            return arr[..., ch]
        return arr

    if arr.ndim == 5:
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


def _velocity_to_vorticity_trajectories(velocity_traj: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Convert velocity trajectories [N, T, X, Y, C] into vorticity [N, T, X, Y]."""
    arr = np.asarray(velocity_traj, dtype=np.float64)
    if arr.ndim != 5 or arr.shape[-1] < 2:
        raise ValueError(
            "Expected velocity trajectories with shape [N, T, X, Y, C>=2], "
            f"got {arr.shape}."
        )

    u = arr[..., 0]
    v = arr[..., 1]
    nx = int(u.shape[-2])
    ny = int(u.shape[-1])

    dx = float(Lx) / float(nx)
    dy = float(Ly) / float(ny)
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    u_hat = np.fft.fft2(u, axes=(-2, -1))
    v_hat = np.fft.fft2(v, axes=(-2, -1))
    dv_dx = np.fft.ifft2((1j * kx_grid) * v_hat, axes=(-2, -1)).real
    du_dy = np.fft.ifft2((1j * ky_grid) * u_hat, axes=(-2, -1)).real
    omega = dv_dx - du_dy
    return np.asarray(omega, dtype=np.float32)


def _pdebench_array_to_trajectories(
    array: np.ndarray,
    layout: str,
    channel_index: int,
    keep_channels: bool = False,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    resolved_layout = _resolve_layout(arr.ndim, layout)
    axis_map = {label: idx for idx, label in enumerate(resolved_layout)}

    required = {"N", "T", "H", "W"}
    if not required.issubset(axis_map):
        raise ValueError(f"PDEBench layout must include N,T,H,W. Got '{resolved_layout}'.")

    if "C" in axis_map and not keep_channels:
        ch = min(max(0, int(channel_index)), arr.shape[axis_map["C"]] - 1)
        arr = np.take(arr, indices=ch, axis=axis_map["C"])
        resolved_layout = resolved_layout.replace("C", "")
        axis_map = {label: idx for idx, label in enumerate(resolved_layout)}

    if "C" in axis_map:
        perm = [axis_map["N"], axis_map["T"], axis_map["H"], axis_map["W"], axis_map["C"]]
    else:
        perm = [axis_map["N"], axis_map["T"], axis_map["H"], axis_map["W"]]
    arr = np.transpose(arr, axes=perm)
    if arr.ndim not in {4, 5}:
        raise ValueError(f"Expected NTHW or NTHWC after transpose, got shape {arr.shape}")
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


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        values: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    text = str(value).strip()
    return [text] if text else []
