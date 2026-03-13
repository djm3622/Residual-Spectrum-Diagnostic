"""Backward-compatible entrypoint for reaction-diffusion PDEBench processing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.reaction_diffusion.process_pdebench import main


if __name__ == "__main__":
    main()
