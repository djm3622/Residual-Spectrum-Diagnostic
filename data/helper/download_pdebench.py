#!/usr/bin/env python3
"""Download PDEBench data files and optionally patch local run config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.helper.pdebench_download_helpers import (
    PDEBENCH_METADATA_URL,
    PDE_NAMES,
    align_config_with_downloaded_h5,
    download_row,
    filter_rows,
    inspect_h5_for_alignment,
    load_metadata_rows,
    patch_config_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_pdebench.py",
        description=(
            "Download PDEBench files into a standard local directory and "
            "auto-fill data.external.pdebench.file_path in a YAML config."
        ),
    )
    parser.add_argument(
        "--pde-name",
        action="append",
        dest="pde_names",
        required=True,
        help=(
            "PDEBench dataset key to download. Repeat for multiple values. "
            f"Choices: {', '.join(PDE_NAMES)}. "
            "Reaction-diffusion aliases: reaction_diffusion, reaction-diffusion, rd, gray_scott."
        ),
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        default="external_data/pdebench",
        help="Root destination for downloaded PDEBench files (default: external_data/pdebench).",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default="configs/navier_stokes.yaml",
        help="Config YAML to patch (default: configs/navier_stokes.yaml).",
    )
    parser.add_argument(
        "--metadata-url",
        type=str,
        default=PDEBENCH_METADATA_URL,
        help="Override metadata CSV URL (defaults to official PDEBench GitHub raw URL).",
    )
    parser.add_argument(
        "--filename-contains",
        action="append",
        default=None,
        help="Optional filename substring filter. Repeat to allow multiple substrings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help=(
            "Maximum number of files to download after filtering (default: 1). "
            "Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--autofill-source",
        action="store_true",
        default=True,
        help="Set data.external.source to 'pdebench' when patching config (default: true).",
    )
    parser.add_argument(
        "--no-autofill-source",
        action="store_true",
        default=False,
        help="Do not set data.external.source when patching config.",
    )
    parser.add_argument(
        "--align-run-config",
        action="store_true",
        default=True,
        help=(
            "Auto-align run config fields from downloaded HDF5 metadata "
            "(dataset_key/layout/dt, and physics forcing defaults) (default: true)."
        ),
    )
    parser.add_argument(
        "--no-align-run-config",
        action="store_true",
        default=False,
        help="Disable automatic run-config alignment.",
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        default=False,
        help="Do not patch config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print actions without downloading or writing config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_folder = Path(args.root_folder).expanduser()
    config_path = Path(args.config_yaml).expanduser()
    pde_names = [name.strip().lower() for name in args.pde_names]
    filename_contains = args.filename_contains
    max_files = int(args.max_files)
    set_source = bool(args.autofill_source) and not bool(args.no_autofill_source)
    align_run_config = bool(args.align_run_config) and not bool(args.no_align_run_config)

    rows = load_metadata_rows(args.metadata_url)
    selected_rows = filter_rows(rows, pde_names, filename_contains, max_files=max_files)
    if not selected_rows:
        raise RuntimeError(
            "No files matched filters. "
            "Adjust --pde-name / --filename-contains / --max-files."
        )

    print(f"metadata rows matched: {len(selected_rows)}")
    downloaded_paths = [download_row(root_folder, row, dry_run=args.dry_run) for row in selected_rows]

    if args.skip_config_update:
        print("[info] config update skipped")
        return

    chosen_file = downloaded_paths[0]
    chosen_row = selected_rows[0]
    pde_name = str(chosen_row["PDE"]).strip().lower()
    patch_config_file(
        config_path=config_path,
        file_path=chosen_file,
        pde_name=pde_name,
        set_source=set_source,
        align_run_config=align_run_config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
