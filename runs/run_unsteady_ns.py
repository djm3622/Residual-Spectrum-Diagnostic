#!/usr/bin/env python3
"""Entry script for the unsteady_ns case study.

Usage:
    python3 runs/run_unsteady_ns.py configs/unsteady_ns.yaml tfno 1 --device auto --loss l2 --basis fourier
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.run_navier_stokes import main


if __name__ == "__main__":
    main()
