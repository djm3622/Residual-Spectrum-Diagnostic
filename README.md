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

## Entry Scripts (3 arguments)

All runs use exactly:

```bash
python3 runs/<entry_script>.py <config.yaml> <method> <seed_number>
```

Examples:

```bash
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml linear 7
```

## Method Argument

Supported `method` values:
- `conv` (aliases: `convolutional`, `spectral`)
- `linear` (alias: `dense`)

## YAML Controls

Each YAML controls all run parameters, including:
- grid/domain
- PDE physics constants
- time integration
- train/test trajectory counts
- training hyperparameters (`noise_level`, `lr`, `n_iter`)
- RSD band settings (`omega_1_frac`, `omega_2_frac`)
- output/checkpoint roots and artifact toggles

## Outputs

Each run writes to deterministic paths:
- `output/<experiment>/<method>/seed_<seed>/results.json`
- `output/<experiment>/<method>/seed_<seed>/summary.png` (if enabled)
- `checkpoints/<experiment>/<method>/seed_<seed>/model_clean.npz`
- `checkpoints/<experiment>/<method>/seed_<seed>/model_noisy.npz`

## Install

```bash
python3 -m pip install -r requirements.txt
```
