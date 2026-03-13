#!/usr/bin/env python3
"""Process PDEBench Navier-Stokes HDF5 into a run-ready omega dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.navier_stokes.helper.downsample_helpers import (
    _build_compression_kwargs,
    _copy_file_attrs,
    _infer_domain_lengths,
    _infer_dt,
    _infer_nu,
    _maybe_copy_time_dataset,
    _patch_yaml_config,
    _pick_dataset_key,
    _process_dataset,
    _read_source_config_blob,
    _resolve_layout,
    _resolve_output_key,
)


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
