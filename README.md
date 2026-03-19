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
- Added case-study entrypoint: `runs/run_unsteady_ns.py` (uses the same NS train/eval pipeline as `run_navier_stokes.py`).
- Refactor note: internals moved from flat files into package subdirectories for maintainability, while public API names were re-exported.

## Entry Scripts

All runs use:

```bash
python3 runs/<entry_script>.py <config.yaml> <method> <seed_number> [--device <auto|cpu|cuda|mps>] [--loss <combined|l2|l1|spectral_decay|energy>] [--basis <fourier|laplace|wavelet|svd>] [--resume-clean-checkpoint <path>] [--resume-noisy-checkpoint <path>]
```

Examples:

```bash
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml tfno 1 --device cuda
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml itfno 1 --device cuda
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml uno 1 --device cuda
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml rno 1 --device cuda
python3 runs/run_navier_stokes.py configs/navier_stokes.yaml swin 1 --device cuda
python3 runs/run_unsteady_ns.py configs/unsteady_ns.yaml tfno 1 --device cuda
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml tfno 7 --device mps
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml itfno 7 --device mps
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml uno 7 --device mps
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml rno 7 --device mps
python3 runs/run_reaction_diffusion.py configs/reaction_diffusion.yaml attn_unet 7 --device mps
```

## Method Argument

Supported `method` values:
- `tfno` (aliases: `neuralop_tfno`, `operator_tfno`) -> Tucker-factorized FNO (`neuraloperator`)
- `itfno` (aliases: `neuralop_itfno`, `operator_itfno`, `implicit_tfno`) -> implicit iterative TFNO (shared hidden block + fixed-point-style updates)
- `uno` (aliases: `neuralop_uno`, `operator_uno`) -> U-shaped Neural Operator (`neuraloperator`)
- `rno` (aliases: `neuralop_rno`, `operator_rno`) -> Recurrent Neural Operator (`neuraloperator`)
- `conv` (aliases: `convolutional`, `legacy_conv`) -> legacy project conv surrogate
- `swin` (aliases: `swin_transformer`, `swin_t`) -> Swin-T backbone + dense decoder baseline
- `attn_unet` (aliases: `attention_unet`, `unet_attn`) -> attention U-Net baseline (SCSE decoder attention)
- `physics` (aliases: `gray_scott`, `grayscott`) -> explicit physics-only stepper

## YAML Controls

Each YAML controls all run parameters, including:
- grid/domain
- PDE physics constants
- time integration
- train/test trajectory counts
- data source selection (`data.external.source`: `generated`, `neuraloperator`, `pdebench`, `fno_mat`)
- training hyperparameters (`noise_level`, `lr`, `n_iter`, `batch_size`, `grad_clip`, `weight_decay`)
- dataloader worker count (`training.dataloader_num_workers`; default `-1` = auto multi-core)
- optional one-cycle learning-rate schedule (`training.use_one_cycle_lr`, `training.one_cycle_pct_start`, `training.one_cycle_div_factor`, `training.one_cycle_final_div_factor`)
- checkpoint cadence (`training.checkpoint_every_epochs`, default `20`)
- early stopping patience in epochs (`training.early_stopping_patience`, default `20`; disabled when no validation split)
- rollout-stability controls (`training.rollout_horizon`, `training.rollout_weight`)
- validation monitoring split (`training.validation_fraction`) used only for progress reporting
- coupled-species balancing controls (`training.u_weight`, `training.v_weight`, `training.channel_balance_cap`)
- transient-tracking controls (`training.dynamics_weight`, `training.early_step_bias`, `training.early_step_decay`)
- neural-operator controls (`training.neural_operator.common` and per-operator overrides in
  `training.neural_operator.tfno`, `training.neural_operator.itfno`, `training.neural_operator.uno`, `training.neural_operator.rno`)
  - optional temporal-window mode for TFNO/ITFNO/UNO via `training.neural_operator.<op>.temporal`:
    `enabled`, `input_steps`, `output_steps` (must equal `input_steps`), `target_mode` (`shifted` or `next_block`), `n_modes_time`
  - ITFNO fixed-point update controls via `training.neural_operator.itfno.implicit`:
    `steps`, `dt`, `relaxation`
  - optional recurrent RNO unroll controls via `training.neural_operator.rno.recurrent`:
    `n_blocks`, `warmup_steps`
- baseline-model controls (`training.baseline_models.common` and per-model overrides in
  `training.baseline_models.swin`, `training.baseline_models.attn_unet`, `training.baseline_models.conv`)
  - shared options: `norm`, `activation`, `dropout`
  - `swin` options: `pretrained`, `freeze_backbone`, `decoder_channels`, `use_attention`, `scse_reduction`
  - `attn_unet` options: `base_channels`, `depth`, `scse_reduction`
  - `conv` options: NS `model_width`/`model_depth`, RD `width`
- training objective (`training.loss`: `combined`, `l2`, `l1`, `spectral_decay`, `energy`)
- progress bars (`progress.enabled`, `progress.data_generation`, `progress.training`, `progress.evaluation`)
  - when training progress is enabled, tqdm shows `train_loss`, `val_loss`, and optimizer `lr` live
- RSD projection basis (`rsd.basis`: `fourier`, `laplace`, `wavelet`, `svd`) and band settings (`omega_1_frac`, `omega_2_frac`)
- multiscale spectral diagnostics band count (`rsd.spectral_band_count`, default `8`)
- optional RSD no-forcing residual mode for RD (`rsd.assume_no_forcing`: `true|false`)
- output/checkpoint roots and artifact toggles
- fit-visualization indices (`eval_pair_index`, `test_case_index`, `test_step_index`)

### External Navier-Stokes Data

`runs/run_navier_stokes.py` supports four sources through `data.external`:
- `generated`: default pseudo-spectral solver trajectories (current behavior).
- `neuraloperator`: uses `neuralop.data.datasets.load_navier_stokes_pt(...)`.
- `pdebench`: reads local PDEBench HDF5 trajectories (for example, `tensor` datasets).
- `fno_mat`: reads full-trajectory FNO MAT files (for example, `NavierStokes_V1e-3_N5000_T50.mat`) via `data.external.fno_mat`.

Notes:
- NeuralOperator PT data is one-step-oriented; the runner constructs short trajectories from each batch sample's `x/y` pair.
- PDEBench mode expects a local HDF5 file path (`data.external.pdebench.file_path`) and supports layout mapping via `data.external.pdebench.layout` (`AUTO`, `NTHW`, `NTHWC`, etc.).
- FNO MAT mode expects trajectories under key `u`; loader orientation is auto-resolved to `[N,T,H,W]` and then converted into standard NS trajectory windows/pairs through the existing training helpers.
- For PDEBench `ns_incom` velocity files, the loader converts `(u, v)` to vorticity `omega = dv/dx - du/dy` before model training/evaluation so NS-RSD metrics remain on vorticity trajectories.
- In all modes, noisy training data is now explicitly corrupted in memory (HF spectral noise injection) before fitting the noisy model.

### Unsteady NS Case Study (`unsteady_ns`)

This case study uses the original FNO trajectory dataset `NavierStokes_V1e-3_N5000_T50.mat`.
Temporal-window training is configured explicitly with `input_steps: 10` (under `training.neural_operator.*.temporal`).

Download:

```bash
python3 data/download_fno_ns_mat.py --output-dir external_data/fno
```

Train + eval (single command; eval is integrated into the run):

```bash
python3 runs/run_unsteady_ns.py configs/unsteady_ns.yaml tfno 1 --device cuda
```

Equivalent eval command:

```bash
python3 runs/run_unsteady_ns.py configs/unsteady_ns.yaml tfno 1 --device cuda
```

Expected output paths:
- `output/unsteady_ns/tfno/loss_l2/basis_fourier/seed_1/results.json`
- `output/unsteady_ns/tfno/loss_l2/basis_fourier/seed_1/summary.png`
- `output/unsteady_ns/tfno/loss_l2/basis_fourier/seed_1/fit_quality/eval_clean.png`
- `checkpoints/unsteady_ns/tfno/loss_l2/basis_fourier/seed_1/model_clean.npz`
- `checkpoints/unsteady_ns/tfno/loss_l2/basis_fourier/seed_1/model_noisy.npz`

#### PDEBench Downloader Helper

Use `data/download_pdebench.py` to fetch PDEBench files into a standard folder and patch your YAML automatically:

```bash
python3 data/download_pdebench.py \
  --pde-name ns_incom \
  --root-folder external_data/pdebench \
  --config-yaml configs/navier_stokes.yaml \
  --autofill-source
```

Useful flags:
- `--filename-contains <text>`: narrow matches (repeatable).
- `--max-files 0`: remove file limit (default is `1`).
- `--skip-config-update`: download only, no YAML edit.
- `--dry-run`: print actions without downloading/writing.

By default, the helper now also aligns run-config fields automatically after download:
- sets `data.external.source: pdebench`
- sets `data.external.pdebench.file_path`
- inspects HDF5 and sets `data.external.pdebench.dataset_key`, `layout`, `channel_index`, and `dt` (when available)
- disables synthetic forcing defaults (`physics.forcing.type: none`, `physics.forcing.amplitude: 0.0`)
- adjusts stride/snapshot compatibility when it can do so safely

Disable auto-alignment if needed:

```bash
python3 data/download_pdebench.py --pde-name ns_incom --no-align-run-config
```

#### PDEBench NS Processor (Downsample + Omega)

Use `data/downsample_pdebench_ns.py` to convert downloaded NS velocity data into a run-ready vorticity file and auto-align `configs/navier_stokes.yaml`:

```bash
python3 data/downsample_pdebench_ns.py \
  --input external_data/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-0.h5 \
  --output external_data/pdebench/2D/NS_incom/ns_incom_inhom_2d_64_omega.h5 \
  --nx 64 \
  --ny 64 \
  --overwrite
```

By default, this script patches `configs/navier_stokes.yaml` after processing. Use `--patch-config <path>` to target a different file, or `--skip-config-update` to disable patching.

When patching config, it updates all NS run-critical fields from processed metadata:
- `grid.nx`, `grid.ny`, `grid.Lx`, `grid.Ly`
- `time.n_snapshots` (clamped to available), `time.t_final` (from resolved `dt`)
- `physics.nu`, and forcing defaults (`type: none`, `amplitude: 0.0`)
- `data.n_train_trajectories`, `data.n_test_trajectories` (auto split from sample count)
- `data.external.source`
- `data.external.pdebench.file_path`, `dataset_key`, `layout`, `channel_index`, `time_stride`, `spatial_stride`, `n_train`, `n_test`, `dt` (when available)

### Device Selection

- Set `training.device` in YAML (`auto`, `cpu`, `cuda`, `mps`), or pass `--device` to override.
- `auto` resolves in this order: `cuda` -> `mps` -> `cpu`.
- Explicit `cuda`/`mps` raises an error if unavailable.

## Outputs

Each run writes to deterministic paths:
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/results.json`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/summary.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/pde_residual_space_time.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/boundary_condition_error.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/spectral_multiband_error.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/spectral_band_error_profile.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/metrics_vs_l2.png` (if enabled)
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/eval_clean.png`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/eval_noisy.png`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/test_clean.png`
- `output/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/fit_quality/test_noisy.png`
- `checkpoints/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/model_clean.npz`
- `checkpoints/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/model_noisy.npz`
- `checkpoints/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/model_<clean|noisy>_epoch_XXXX.npz` (saved every `training.checkpoint_every_epochs`)
- `checkpoints/<experiment>/<method>/loss_<loss>/basis_<basis>/seed_<seed>/model_<clean|noisy>_best.npz` (lowest validation loss checkpoint)

Checkpoint files now store full training state (model weights, optimizer, scheduler, AMP grad-scaler, and RNG state) so runs can be resumed from the exact saved epoch.
Use `--resume-clean-checkpoint <path>` and/or `--resume-noisy-checkpoint <path>` to resume either phase directly.

`results.json` now includes, in addition to L2/HFV/LFV:
- space-time PDE residual metrics (`clean/noisy_pde_residual_st_rms`, plus `u/v` components for RD)
- periodic boundary-condition error (`clean/noisy_boundary_error`)
- multi-band spectral errors (`clean/noisy_spectral_multiband_error`, low/mid/high splits, and per-band entries)
- metric-to-L2 correlation block (`metric_vs_l2`) for clean/noisy trajectories

## Install

```bash
python3 -m pip install -r requirements.txt
```
