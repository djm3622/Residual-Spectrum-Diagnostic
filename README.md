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
python3 runs/<entry_script>.py <config.yaml> <method> <seed_number> [--device <auto|cpu|cuda|mps>] [--loss <combined|l2|l1|spectral_decay|energy>] [--basis <fourier|laplace|wavelet|svd>]
```

Examples:

```bash
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1 --device cuda
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml conv 7 --device mps --loss spectral_decay --basis wavelet
```

## Method Argument

Supported `method` values:
- `conv` (aliases: `convolutional`, `spectral`) -> learned one-step conv surrogate
- `physics` (aliases: `gray_scott`, `grayscott`) -> explicit physics-only stepper
- `conv_nn` (aliases: `conv_legacy`, `nn`) -> alias for the learned conv surrogate

## YAML Controls

Each YAML controls all run parameters, including:
- grid/domain
- PDE physics constants
- time integration
- train/test trajectory counts
- training hyperparameters (`noise_level`, `lr`, `n_iter`, `batch_size`, `grad_clip`)
- rollout-stability controls (`training.rollout_horizon`, `training.rollout_weight`)
- coupled-species balancing controls (`training.u_weight`, `training.v_weight`, `training.channel_balance_cap`)
- transient-tracking controls (`training.dynamics_weight`, `training.early_step_bias`, `training.early_step_decay`)
- model capacity controls for NS conv surrogate (`training.model_width`, `training.model_depth`)
- training objective (`training.loss`: `combined`, `l2`, `l1`, `spectral_decay`, `energy`)
- progress bars (`progress.enabled`, `progress.data_generation`, `progress.training`, `progress.evaluation`)
- RSD projection basis (`rsd.basis`: `fourier`, `laplace`, `wavelet`, `svd`) and band settings (`omega_1_frac`, `omega_2_frac`)
- output/checkpoint roots and artifact toggles
- fit-visualization indices (`eval_pair_index`, `test_case_index`, `test_step_index`)

### Device Selection

- Set `training.device` in YAML (`auto`, `cpu`, `cuda`, `mps`), or pass `--device` to override.
- `auto` resolves in this order: `cuda` -> `mps` -> `cpu`.
- Explicit `cuda`/`mps` raises an error if unavailable.

## Outputs

Each run writes to deterministic paths:
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/results.json`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/summary.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/eval_clean.png`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/eval_noisy.png`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/test_clean.png`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/test_noisy.png`
- `checkpoints/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/model_clean.npz`
- `checkpoints/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/model_noisy.npz`

## Install

```bash
python3 -m pip install -r requirements.txt
```
