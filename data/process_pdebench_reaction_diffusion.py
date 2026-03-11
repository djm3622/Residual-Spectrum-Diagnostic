#!/usr/bin/env python3
"""Downsample PDEBench reaction-diffusion states and patch run config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download_pdebench import align_config_with_downloaded_h5, inspect_h5_for_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a downsampled copy of PDEBench reaction-diffusion data "
            "and patch a YAML config with inferred physics/time/grid metadata."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Source HDF5 path.")
    parser.add_argument("--output", type=str, required=True, help="Destination HDF5 path.")
    parser.add_argument("--nx", type=int, default=64, help="Target x-resolution (default: 64).")
    parser.add_argument("--ny", type=int, default=64, help="Target y-resolution (default: 64).")
    parser.add_argument(
        "--patch-config",
        type=str,
        default="configs/reaction_diffusion.yaml",
        help="Config YAML to patch (default: configs/reaction_diffusion.yaml).",
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
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output if it exists.")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        default=False,
        help="Delete source file after successful write.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    config_path = Path(args.patch_config).expanduser()
    target_nx = int(args.nx)
    target_ny = int(args.ny)

    if target_nx <= 0 or target_ny <= 0:
        raise ValueError("Target nx/ny must be positive.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression_kwargs = _build_compression_kwargs(args.compression, args.compression_level)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        _copy_attrs(src.attrs, dst.attrs)
        sample_keys = _sample_group_keys(src)
        if not sample_keys:
            raise RuntimeError(
                "Expected grouped sample structure (e.g. 0000/data). "
                f"Found top-level keys: {list(src.keys())[:20]}"
            )

        src_nx = src_ny = stride_x = stride_y = 0
        for key in sample_keys:
            src_group = src[key]
            if not isinstance(src_group, h5py.Group):
                continue
            dst_group = dst.create_group(key)
            src_nx, src_ny, stride_x, stride_y = _copy_sample_group(
                src_group=src_group,
                dst_group=dst_group,
                target_nx=target_nx,
                target_ny=target_ny,
                compression_kwargs=compression_kwargs,
            )

    report = _patch_yaml_config(config_path=config_path, output_path=output_path)

    if args.delete_source:
        input_path.unlink()
        print(f"Deleted source: {input_path.resolve()}")

    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Samples: {len(sample_keys)}")
    print(f"Source HxW: {src_nx}x{src_ny}")
    print(f"Target HxW: {target_nx}x{target_ny}")
    print(f"Spatial stride: x{stride_x}, y{stride_y}")
    print(f"Patched config: {config_path.resolve()}")
    for key, value in report.items():
        print(f"[ok]   {key} = {value!r}")


if __name__ == "__main__":
    main()
