"""Backward-compatible entrypoint for PDEBench downloader."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.helper.download_pdebench import (
    align_config_with_downloaded_h5,
    inspect_h5_for_alignment,
    main,
    patch_config_file,
)


if __name__ == "__main__":
    main()
