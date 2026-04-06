# LSPR Spectra Master

LSPR concentration prediction and physics-guided deep learning experiments for Ag/BSA spectra. The repository contains:

- Baseline concentration predictors (`Model A`, `Model B`)
- Fusion predictor with BSA physics branch (`Model C`)
- Stage 2.5 / Stage 3 physics-guided joint training with Hill constraint (`3A`, `3B`, `3C`)
- Figure-generation scripts and paper draft materials

## Recommended Environment

Use `conda` with Python 3.9.

- Recommended: `py39`
- Backup option: `gan`
- Do not use `base` or Python 3.12 for project verification

Verified command on this repo:

```powershell
conda run -n py39 pytest
```

Verified result on 2026-04-02:

```text
32 passed, 1 warning
```

## Installation

Create and populate a clean environment:

```powershell
conda create -n lspr_py39 python=3.9 -y
conda activate lspr_py39
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If you already have the shared workstation environments, you can directly use:

```powershell
conda activate py39
```

## Repository Layout

- `src/core/`: models, training primitives, Hill modules, inference engine
- `src/gui/`: desktop GUI entrypoint
- `scripts/`: training, evaluation, plotting, experiment runners
- `tests/`: automated tests
- `models/pretrained/`: pretrained weights and normalization artifacts
- `outputs/`: generated metrics, figures, and experiment snapshots
- `docx/`: paper drafts, formulas, and phase plan

## Reproducibility Entry Points

All commands below assume the working directory is the repository root:

`C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master`

### 1. Validate the environment

```powershell
conda run -n py39 pytest
```

### 2. Run the GUI

```powershell
conda run -n py39 python main.py
```

### 3. Train baseline models

Model A:

```powershell
conda run -n py39 python scripts/train_concentration_v1.py
```

Model B:

```powershell
conda run -n py39 python scripts/train_concentration_v2.py
```

Model C:

```powershell
conda run -n py39 python scripts/train_concentration_v2_fusion.py
```

### 4. Run Stage 3 experiments

Single profile / single seed:

```powershell
conda run -n py39 python scripts/run_stage3_experiment.py --profile 3C-learnable-regressor --seed 20260325
```

Direct training entry:

```powershell
conda run -n py39 python scripts/train_joint_physics_dl.py --stage3-profile 3C-learnable-regressor --seed 20260325
```

Fit Hill parameters:

```powershell
conda run -n py39 python scripts/fit_stage3_hill_params.py
```

### 5. Generate paper figures

Stage 3 comparison figures:

```powershell
conda run -n py39 python scripts/plot_stage3_comparison.py
```

Main ablation and result figures:

```powershell
conda run -n py39 python scripts/plot_ablation_full.py
```

Supplementary figures:

```powershell
conda run -n py39 python scripts/plot_supplementary.py
```

### 6. Evaluate saved predictors

```powershell
conda run -n py39 python scripts/evaluate_test_predict.py
```

## Key Artifacts

- Fusion predictor weights: `models/pretrained/spectral_predictor_v2_fusion.pth`
- Fusion normalization stats: `models/pretrained/predictor_v2_fusion_norm_params.pth`
- Stage 3 Hill params: `models/pretrained/stage3_hill_params.pth`
- Best Stage 3 predictor: `models/pretrained/spectral_predictor_v2_stage3_3c_learnable_regressor.pth`
- Best Stage 3 generator: `models/pretrained/spectral_generator_stage3_3c_learnable_regressor.pth`

## Notes

- Data files are expected under `data/processed/`, including the fixed split directory `splits_reconstructed`.
- The plotting scripts read from existing CSV outputs in `outputs/`; they are not generic plotting wrappers.
- The paper naming currently uses both `Model D` and `3C-learnable-regressor` for the Stage 3 best configuration. Keep that mapping explicit when writing manuscript text.
