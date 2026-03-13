"""Helper functions for reaction-diffusion PDEBench processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import yaml

from data.helper.pdebench_download_helpers import (
    align_config_with_downloaded_h5,
    inspect_h5_for_alignment,
)


def _build_compression_kwargs(method: str, level: int) -> Dict[str, Any]:
    if method == "none":
        return {}
    if method == "gzip":
        return {"compression": "gzip", "compression_opts": int(np.clip(level, 0, 9))}
    return {"compression": method}


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _find_sample_data_dataset(sample_group: h5py.Group) -> Tuple[str, h5py.Dataset]:
    if "data" in sample_group and isinstance(sample_group["data"], h5py.Dataset):
        return "data", sample_group["data"]
    for key in sample_group.keys():
        obj = sample_group.get(key)
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            return key, obj
    raise RuntimeError(f"Sample group '{sample_group.name}' has no data dataset.")


def _sample_group_keys(handle: h5py.File) -> list[str]:
    keys = [key for key in handle.keys() if isinstance(handle.get(key), h5py.Group)]
    return sorted(keys)


def _downsample_spatial_last_hwc(data: np.ndarray, target_nx: int, target_ny: int) -> tuple[np.ndarray, int, int]:
    if data.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C] array, got shape {data.shape}.")
    src_nx = int(data.shape[1])
    src_ny = int(data.shape[2])
    if src_nx % target_nx != 0 or src_ny % target_ny != 0:
        raise ValueError(f"Source HxW={src_nx}x{src_ny} is not divisible by target {target_nx}x{target_ny}.")
    stride_x = src_nx // target_nx
    stride_y = src_ny // target_ny
    return np.asarray(data[:, ::stride_x, ::stride_y, :], dtype=np.float32), stride_x, stride_y


def _copy_sample_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    target_nx: int,
    target_ny: int,
    compression_kwargs: Dict[str, Any],
) -> tuple[int, int, int, int]:
    _copy_attrs(src_group.attrs, dst_group.attrs)
    data_key, src_data = _find_sample_data_dataset(src_group)
    downsampled, stride_x, stride_y = _downsample_spatial_last_hwc(
        np.asarray(src_data, dtype=np.float32),
        target_nx=target_nx,
        target_ny=target_ny,
    )
    dst_data = dst_group.create_dataset(data_key, data=downsampled, dtype=np.float32, **compression_kwargs)
    _copy_attrs(src_data.attrs, dst_data.attrs)

    if "grid" in src_group and isinstance(src_group["grid"], h5py.Group):
        src_grid = src_group["grid"]
        dst_grid = dst_group.create_group("grid")
        _copy_attrs(src_grid.attrs, dst_grid.attrs)
        for axis in ("t", "x", "y"):
            if axis not in src_grid or not isinstance(src_grid[axis], h5py.Dataset):
                continue
            values = np.asarray(src_grid[axis])
            if axis == "x":
                values = values[::stride_x]
            elif axis == "y":
                values = values[::stride_y]
            dst_axis = dst_grid.create_dataset(axis, data=values, dtype=values.dtype)
            _copy_attrs(src_grid[axis].attrs, dst_axis.attrs)

    src_nx = int(src_data.shape[1])
    src_ny = int(src_data.shape[2])
    return src_nx, src_ny, stride_x, stride_y


def _patch_yaml_config(config_path: Path, output_path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")

    inspection = inspect_h5_for_alignment(output_path)
    report = align_config_with_downloaded_h5(
        raw_config=raw,
        file_path=output_path,
        pde_name="2d_reacdiff",
        inspection=inspection,
        set_source=True,
    )

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return report
