"""Metadata fetching and file-download helpers for PDEBench datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

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

PDE_NAME_ALIASES = {
    "reaction_diffusion": "2d_reacdiff",
    "reaction-diffusion": "2d_reacdiff",
    "rd": "2d_reacdiff",
    "gray_scott": "2d_reacdiff",
    "grayscott": "2d_reacdiff",
}

REQUIRED_METADATA_COLUMNS = ("PDE", "URL", "Path", "Filename", "MD5")


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
    allowed = {
        PDE_NAME_ALIASES.get(name.strip().lower(), name.strip().lower())
        for name in pde_names
    }
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
                if any(_filename_fragment_matches(str(row["Filename"]), fragment) for fragment in filters)
            ]

    if max_files > 0:
        selected = selected[:max_files]

    return selected


def _strip_numeric_zero_padding(text: str) -> str:
    """Collapse zero-padded numeric tokens (e.g. 049 -> 49) inside freeform text."""

    def _repl(match: re.Match[str]) -> str:
        prefix, digits, suffix = match.group(1), match.group(2), match.group(3)
        return f"{prefix}{int(digits)}{suffix}"

    return re.sub(r"(^|[^0-9])0+([0-9]+)([^0-9]|$)", _repl, text)


def _filename_fragment_matches(filename: str, fragment: str) -> bool:
    """
    Case-insensitive substring match with a zero-padding fallback.
    This allows filters like '...512-049.h5' to match real files named '...512-49.h5'.
    """
    filename_norm = filename.strip().lower()
    fragment_norm = fragment.strip().lower()
    if not fragment_norm:
        return False
    if fragment_norm in filename_norm:
        return True

    depadded_fragment = _strip_numeric_zero_padding(fragment_norm)
    if depadded_fragment != fragment_norm and depadded_fragment in filename_norm:
        return True
    return False


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
        print(f"[dry-run] would download {url} -> {destination_file}")
        return destination_file

    destination_dir.mkdir(parents=True, exist_ok=True)

    if destination_file.exists() and expected_md5:
        current_md5 = md5sum(destination_file)
        if current_md5 == expected_md5:
            print(f"[skip] exists and md5 matches: {destination_file}")
            return destination_file
        print(f"[warn] md5 mismatch for existing file; re-downloading: {destination_file}")

    print(f"[download] {url}")
    with urlopen(url) as response, destination_file.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if expected_md5:
        actual_md5 = md5sum(destination_file)
        if actual_md5 != expected_md5:
            destination_file.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {destination_file.name}: expected {expected_md5}, got {actual_md5}"
            )
        print(f"[ok] md5 verified: {destination_file}")
    else:
        print(f"[ok] downloaded (no md5 provided): {destination_file}")

    return destination_file
