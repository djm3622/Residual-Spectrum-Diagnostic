#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash cleanup_checkpoints.sh /home/djm3622/Residual-Spectrum-Diagnostic/checkpoints

ROOT="${1:-/home/djm3622/Residual-Spectrum-Diagnostic/checkpoints}"
DRY_RUN="${DRY_RUN:-0}"

echo "Checkpoint root: $ROOT"
echo "Dry run: $DRY_RUN"

# Keep only these two filenames everywhere under checkpoints:
#   model_clean_best.npz
#   model_noisy_best.npz
# Delete everything else matching model_*.npz

if [[ "$DRY_RUN" == "1" ]]; then
  find "$ROOT" -type f -name 'model_*.npz' \
    ! -name 'model_clean_best.npz' \
    ! -name 'model_noisy_best.npz' -print
else
  find "$ROOT" -type f -name 'model_*.npz' \
    ! -name 'model_clean_best.npz' \
    ! -name 'model_noisy_best.npz' -delete
  echo "Done."
fi
