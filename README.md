# Residual-Spectrum-Diagnostic

Structured repo for RSD experiments on:
- 2D incompressible Navier-Stokes (vorticity form)
- 2D Gray-Scott reaction-diffusion

## Repository Layout

- `runs/`: entry scripts (CLI)
- `configs/`: YAML run configurations
- `data/`: PDE solvers and problem config dataclasses
- `models/`: surrogate model implementations
- `utils/`: diagnostics, noise, plotting, config/io helpers
- `output/`: run artifacts (kept mostly git-ignored)
- `checkpoints/`: model checkpoints (kept mostly git-ignored)

## Entry Scripts

All runs use:

```bash
python3 runs/<entry_script>.py <config.yaml> <method> <seed_number> [--device <auto|cpu|cuda|mps>]
```

Examples:

```bash
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1 --device cuda
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml conv 7 --device mps
```

## Method Argument

Supported `method` values:
- `conv` (aliases: `convolutional`, `spectral`)

## YAML Controls

Each YAML controls all run parameters, including:
- grid/domain
- PDE physics constants
- time integration
- train/test trajectory counts
- training hyperparameters (`noise_level`, `lr`, `n_iter`, `batch_size`, `grad_clip`)
- progress bars (`progress.enabled`, `progress.data_generation`, `progress.training`, `progress.evaluation`)
- RSD band settings (`omega_1_frac`, `omega_2_frac`)
- output/checkpoint roots and artifact toggles
- fit-visualization indices (`eval_pair_index`, `test_case_index`, `test_step_index`)

### Device Selection

- Set `training.device` in YAML (`auto`, `cpu`, `cuda`, `mps`), or pass `--device` to override.
- `auto` resolves in this order: `cuda` -> `mps` -> `cpu`.
- Explicit `cuda`/`mps` raises an error if unavailable.

## Outputs

Each run writes to deterministic paths:
- `output/<experiment>/<method>/seed_<seed>/results.json`
- `output/<experiment>/<method>/seed_<seed>/summary.png` (if enabled)
- `output/<experiment>/<method>/seed_<seed>/fit_quality/eval_clean.png`
- `output/<experiment>/<method>/seed_<seed>/fit_quality/eval_noisy.png`
- `output/<experiment>/<method>/seed_<seed>/fit_quality/test_clean.png`
- `output/<experiment>/<method>/seed_<seed>/fit_quality/test_noisy.png`
- `checkpoints/<experiment>/<method>/seed_<seed>/model_clean.npz`
- `checkpoints/<experiment>/<method>/seed_<seed>/model_noisy.npz`

## Install

```bash
python3 -m pip install -r requirements.txt
```
