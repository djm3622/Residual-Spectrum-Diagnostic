"""Backward-compatible re-export module for PDEBench download helper APIs."""

from __future__ import annotations

from data.helper.pdebench_download_alignment import (
    align_config_with_downloaded_h5,
    inspect_h5_for_alignment,
    patch_config_file,
)
from data.helper.pdebench_download_metadata import (
    PDEBENCH_METADATA_URL,
    PDE_NAMES,
    download_row,
    filter_rows,
    load_metadata_rows,
)

__all__ = [
    "PDEBENCH_METADATA_URL",
    "PDE_NAMES",
    "load_metadata_rows",
    "filter_rows",
    "download_row",
    "inspect_h5_for_alignment",
    "align_config_with_downloaded_h5",
    "patch_config_file",
]
