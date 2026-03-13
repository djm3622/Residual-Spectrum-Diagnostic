"""Helper functions for reaction-diffusion external-data adapters."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


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
    dataset_key: str,
    layout_hint: str,
    u_channel_index: int,
    v_channel_index: int,
    time_stride: int,
    spatial_stride: int,
    target_nx: int,
    target_ny: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    u_trajectories: List[np.ndarray] = []
    v_trajectories: List[np.ndarray] = []
    for sample_idx in selected_indices:
        key = sample_keys[int(sample_idx)]
        group = handle[key]
        resolved_dataset_key = str(dataset_key).strip() or "data"
        if resolved_dataset_key not in group:
            if "data" in group:
                resolved_dataset_key = "data"
            else:
                raise KeyError(
                    f"Dataset key '{dataset_key}' not found in sample group '{key}'. "
                    f"Available keys: {list(group.keys())}"
                )
        raw = np.asarray(group[resolved_dataset_key], dtype=np.float32)
        layout = _resolve_layout(raw.ndim, layout_hint, grouped_hint=True)
        u_traj, v_traj = _array_to_uv_trajectory(
            raw,
            layout=layout,
            u_channel_index=int(u_channel_index),
            v_channel_index=int(v_channel_index),
        )
        u_traj = u_traj[::time_stride, ::spatial_stride, ::spatial_stride]
        v_traj = v_traj[::time_stride, ::spatial_stride, ::spatial_stride]
        u_trajectories.append(_match_trajectory_resolution(u_traj, target_nx, target_ny))
        v_trajectories.append(_match_trajectory_resolution(v_traj, target_nx, target_ny))
    return u_trajectories, v_trajectories


def _load_root_samples(
    dataset: Any,
    selected_indices: List[int],
    layout: str,
    u_channel_index: int,
    v_channel_index: int,
    time_stride: int,
    spatial_stride: int,
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
            u_channel_index=int(u_channel_index),
            v_channel_index=int(v_channel_index),
        )
        u_traj = u_traj[::time_stride, ::spatial_stride, ::spatial_stride]
        v_traj = v_traj[::time_stride, ::spatial_stride, ::spatial_stride]
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
