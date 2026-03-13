#!/usr/bin/env python3
"""Downsample PDEBench reaction-diffusion states and patch run config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.reaction_diffusion.helper.process_pdebench_helpers import (
    _build_compression_kwargs,
    _copy_attrs,
    _copy_sample_group,
    _patch_yaml_config,
    _sample_group_keys,
)


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
