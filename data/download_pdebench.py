#!/usr/bin/env python3
"""Download PDEBench data files and optionally patch local run config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import sys
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml

PDEBENCH_METADATA_URL = (
    "https://raw.githubusercontent.com/pdebench/PDEBench/main/"
    "pdebench/data_download/pdebench_data_urls.csv"
)

PDE_NAMES = (
    "advection",
    "burgers",
    "1d_cfd",
    "diff_sorp",
    "1d_reacdiff",
    "2d_cfd",
    "darcy",
    "2d_reacdiff",
    "ns_incom",
    "swe",
    "3d_cfd",
)

REQUIRED_METADATA_COLUMNS = ("PDE", "URL", "Path", "Filename", "MD5")


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
            f"Choices: {', '.join(PDE_NAMES)}"
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
        default=False,
        help="Also set data.external.source to 'pdebench' when patching config.",
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


def load_metadata_rows(metadata_url: str) -> List[Dict[str, str]]:
    try:
        with urlopen(metadata_url) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Failed to fetch metadata CSV ({metadata_url}): HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch metadata CSV ({metadata_url}): {exc.reason}") from exc

    text = payload.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise RuntimeError("Metadata CSV is empty.")

    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in reader.fieldnames]
    if missing:
        raise RuntimeError(
            "Metadata CSV missing required columns: "
            f"{', '.join(missing)}. Found: {reader.fieldnames}"
        )
    return rows


def filter_rows(
    rows: List[Dict[str, str]],
    pde_names: List[str],
    filename_contains: List[str] | None,
    max_files: int,
) -> List[Dict[str, str]]:
    allowed = {name.strip().lower() for name in pde_names}
    unsupported = [name for name in sorted(allowed) if name not in PDE_NAMES]
    if unsupported:
        raise ValueError(
            "Unsupported --pde-name values: "
            f"{', '.join(unsupported)}. Choices: {', '.join(PDE_NAMES)}"
        )

    selected = [
        row
        for row in rows
        if str(row["PDE"]).strip().lower() in allowed
    ]

    if filename_contains:
        filters = [frag for frag in (item.strip() for item in filename_contains) if frag]
        if filters:
            selected = [
                row
                for row in selected
                if any(fragment in str(row["Filename"]) for fragment in filters)
            ]

    if max_files > 0:
        selected = selected[:max_files]

    return selected


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_row(root_folder: Path, row: Dict[str, str], dry_run: bool) -> Path:
    rel_folder = Path(str(row["Path"]).strip())
    filename = str(row["Filename"]).strip()
    url = str(row["URL"]).strip()
    expected_md5 = str(row.get("MD5", "")).strip().lower()

    destination_dir = root_folder / rel_folder
    destination_file = destination_dir / filename

    if dry_run:
        print(f"[dry-run] download {url} -> {destination_file}")
        return destination_file

    destination_dir.mkdir(parents=True, exist_ok=True)

    if destination_file.exists() and expected_md5:
        existing_md5 = md5sum(destination_file)
        if existing_md5 == expected_md5:
            print(f"[skip] {destination_file} (md5 ok)")
            return destination_file
        print(f"[warn] {destination_file} md5 mismatch, re-downloading")

    tmp_file = destination_file.with_suffix(destination_file.suffix + ".part")
    try:
        with urlopen(url) as response, tmp_file.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        tmp_file.replace(destination_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    if expected_md5:
        downloaded_md5 = md5sum(destination_file)
        if downloaded_md5 != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {destination_file}. expected={expected_md5}, got={downloaded_md5}"
            )

    print(f"[ok] {destination_file}")
    return destination_file


def patch_config_file(
    config_path: Path,
    file_path: Path,
    set_source: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] patch {config_path} data.external.pdebench.file_path={file_path.resolve()}")
        if set_source:
            print(f"[dry-run] patch {config_path} data.external.source=pdebench")
        return

    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")

    data = raw.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config key 'data' must be a mapping.")

    external = data.setdefault("external", {})
    if not isinstance(external, dict):
        raise ValueError("Config key 'data.external' must be a mapping.")

    if set_source:
        external["source"] = "pdebench"

    pdebench_cfg = external.setdefault("pdebench", {})
    if not isinstance(pdebench_cfg, dict):
        raise ValueError("Config key 'data.external.pdebench' must be a mapping.")

    pdebench_cfg["file_path"] = str(file_path.resolve())

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    print(f"[ok] patched {config_path} with data.external.pdebench.file_path={file_path.resolve()}")


def main() -> None:
    args = parse_args()

    root_folder = Path(args.root_folder).expanduser()
    config_path = Path(args.config_yaml).expanduser()
    pde_names = [name.strip().lower() for name in args.pde_names]
    filename_contains = args.filename_contains
    max_files = int(args.max_files)

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
    patch_config_file(
        config_path=config_path,
        file_path=chosen_file,
        set_source=bool(args.autofill_source),
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
