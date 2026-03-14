#!/usr/bin/env python3
"""Download preset PDEBench 2D compressible CFD files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.helper.pdebench_download_helpers import (  # noqa: E402
    PDEBENCH_METADATA_URL,
    download_row,
    load_metadata_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_pdebench_2d_cfd.py",
        description=(
            "Download 2D compressible CFD files from PDEBench using preset profiles. "
            "Default profile 'rand' targets the non-turbulent candidate files."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("rand", "turb", "all"),
        default="rand",
        help=(
            "2D CFD file profile: "
            "'rand' (2D_CFD_Rand_*), "
            "'turb' (2D_CFD_Turb_*), "
            "'all' (every 2D_CFD row in metadata)."
        ),
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        default="external_data/pdebench",
        help="Root destination folder for downloads (default: external_data/pdebench).",
    )
    parser.add_argument(
        "--metadata-url",
        type=str,
        default=PDEBENCH_METADATA_URL,
        help="Override metadata CSV URL (default: official PDEBench metadata).",
    )
    parser.add_argument(
        "--filename-contains",
        action="append",
        default=None,
        help=(
            "Optional extra filename substring filter. Repeatable. "
            "All provided fragments must be present."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help=(
            "Maximum number of files to download after filtering "
            "(default: 0 = no limit)."
        ),
    )
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default="",
        help="Optional path to write a CSV manifest of matched files.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        default=False,
        help="Write manifest/list only; skip data download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print actions without downloading files.",
    )
    return parser.parse_args()


def _profile_match(row: Dict[str, str], profile: str) -> bool:
    pde_name = str(row.get("PDE", "")).strip().lower()
    if pde_name != "2d_cfd":
        return False

    filename = str(row.get("Filename", "")).strip()
    if profile == "rand":
        return filename.startswith("2D_CFD_Rand_")
    if profile == "turb":
        return filename.startswith("2D_CFD_Turb_")
    return True


def _extra_filter(filename: str, fragments: List[str]) -> bool:
    normalized = filename.strip().lower()
    return all(fragment in normalized for fragment in fragments)


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["PDE", "Filename", "URL", "Path", "MD5"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    rows = load_metadata_rows(args.metadata_url)

    selected = [row for row in rows if _profile_match(row, args.profile)]
    fragments = [
        part.strip().lower()
        for part in (args.filename_contains or [])
        if part and part.strip()
    ]
    if fragments:
        selected = [
            row
            for row in selected
            if _extra_filter(str(row.get("Filename", "")), fragments)
        ]

    selected.sort(key=lambda row: str(row.get("Filename", "")))
    if int(args.max_files) > 0:
        selected = selected[: int(args.max_files)]

    if not selected:
        raise RuntimeError("No 2D_CFD files matched the requested filters.")

    print(f"matched rows: {len(selected)}")
    for row in selected:
        print(f"- {row['Filename']}")

    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv).expanduser()
        _write_manifest(manifest_path, selected)
        print(f"[ok] wrote manifest: {manifest_path}")

    if args.manifest_only:
        print("[info] manifest-only mode enabled; skipping download.")
        return

    root_folder = Path(args.root_folder).expanduser()
    for row in selected:
        download_row(root_folder, row, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
