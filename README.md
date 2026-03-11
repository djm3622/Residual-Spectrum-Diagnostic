# Residual-Spectrum-Diagnostic

Structured repo for RSD experiments on:
- 2D incompressible Navier-Stokes (vorticity form)
- 2D Gray-Scott reaction-diffusion

## Repository Layout

- `runs/`: entry scripts (CLI)
- `configs/`: YAML run configurations
- `data/`: PDE solvers and problem config dataclasses
- `models/`: surrogate model implementations
- `models/navier_stokes/`: NS surrogate package (`layers/`, `blocks/`, `helpers/`, `models/`)
- `models/reaction_diffusion/`: RD surrogate package (`layers/`, `blocks/`, `helpers/`, `models/`)
- `utils/`: diagnostics, noise, plotting, config/io helpers
- `output/`: run artifacts (kept mostly git-ignored)
- `checkpoints/`: model checkpoints (kept mostly git-ignored)

### Backward Compatibility

- Existing import paths are preserved:
  - `from models.navier_stokes import ...`
  - `from models.reaction_diffusion import ...`
- Existing run CLIs are unchanged (`runs/run_navier_stokes.py`, `runs/run_reaction_diffusion.py`).
- Refactor note: internals moved from flat files into package subdirectories for maintainability, while public API names were re-exported.

## Entry Scripts

All runs use:

```bash
python3 runs/<entry_script>.py <config.yaml> <method> <seed_number> [--device <auto|cpu|cuda|mps>] [--loss <combined|l2|l1|spectral_decay|energy>] [--basis <fourier|laplace|wavelet|svd>]
```

Examples:

```bash
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml conv 1
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml fno 1 --device cuda
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml tfno 1 --device cuda
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml uno 1 --device cuda
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml fno 7 --device mps --loss spectral_decay --basis wavelet
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml tfno 7 --device mps
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml uno 7 --device mps
```

## Method Argument

Supported `method` values:
- `conv` (aliases: `convolutional`, `spectral`) -> learned one-step conv surrogate
- `fno` (aliases: `neuralop_fno`, `operator_fno`) -> Fourier Neural Operator (`neuraloperator`)
- `tfno` (aliases: `neuralop_tfno`, `operator_tfno`) -> Tucker-factorized FNO (`neuraloperator`)
- `uno` (aliases: `neuralop_uno`, `operator_uno`) -> U-shaped Neural Operator (`neuraloperator`)
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
- validation monitoring split (`training.validation_fraction`) used only for progress reporting
- coupled-species balancing controls (`training.u_weight`, `training.v_weight`, `training.channel_balance_cap`)
- transient-tracking controls (`training.dynamics_weight`, `training.early_step_bias`, `training.early_step_decay`)
- model capacity controls for NS conv surrogate (`training.model_width`, `training.model_depth`)
- neural-operator controls (`training.neural_operator.common` and per-operator overrides in
  `training.neural_operator.fno`, `training.neural_operator.tfno`, `training.neural_operator.uno`)
- training objective (`training.loss`: `combined`, `l2`, `l1`, `spectral_decay`, `energy`)
- progress bars (`progress.enabled`, `progress.data_generation`, `progress.training`, `progress.evaluation`)
  - when training progress is enabled, tqdm shows `train_loss` and `val_loss` live
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
