# Stage 3 Hill Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage 3 Hill-constrained training on top of the Stage 2.5 joint-training pipeline, starting from a fixed-parameter frozen baseline and extending to regressor-unfrozen and learnable-Hill variants.

**Architecture:** Keep `scripts/train_joint_physics_dl.py` as the single training entrypoint, but move Hill math, profiles, and hill-aware alternating training into focused `src/core/stage3_*` modules. Use a separate fitting script to produce fixed Hill parameters from `Reconstructed_Preprocessed_Features_and_Delta.xlsx`, then add a stage3 experiment runner mirroring the existing stage25 runner for reproducible seed snapshots.

**Tech Stack:** Python, PyTorch, pandas, NumPy, matplotlib, pytest, PowerShell, `conda run -n gan`

---

## File Map

- Create: `src/core/stage3_hill.py`
  Responsibility: Hill curve modules, paired `delta_lambda` extraction from feature tables, soft-argmax peak extraction, fixed vs learnable Hill parameterization, and `L_hill` helper functions.
- Create: `src/core/stage3_config.py`
  Responsibility: Stage 3 profiles (`3A-fixed-frozen`, `3B-fixed-regressor`, `3C-learnable-regressor`) and argument override helpers.
- Create: `src/core/stage3_training.py`
  Responsibility: hill-aware generator step and alternating epoch runner so stage25 training primitives remain unchanged.
- Modify: `scripts/train_joint_physics_dl.py`
  Responsibility: stage3 CLI, stage3 profile application, hill parameter loading/building, stage3 output tags, stage3-aware epoch runner, and metrics/checkpoint metadata.
- Create: `scripts/fit_stage3_hill_params.py`
  Responsibility: build paired BSA/Ag `delta_lambda` dataset from `sheet1`, fit fixed Hill parameters, and save `models/pretrained/stage3_hill_params.pth`.
- Create: `scripts/run_stage3_experiment.py`
  Responsibility: stage3 source-asset resolution, train command assembly, run-name generation, and snapshotting stage3 artifacts into seed directories.
- Create: `tests/test_stage3_hill.py`
  Responsibility: unit tests for Hill math, fixed/learnable parameter constraints, paired dataset extraction, and soft-argmax peak extraction.
- Create: `tests/test_stage3_config.py`
  Responsibility: stage3 profile defaults and argument override tests.
- Create: `tests/test_stage3_training.py`
  Responsibility: hill-aware generator-step and alternating-runner tests.
- Create: `tests/test_train_joint_physics_dl_stage3.py`
  Responsibility: stage3 CLI/profile integration and output-tag tests.
- Create: `tests/test_run_stage3_experiment.py`
  Responsibility: stage3 runner path/build/snapshot tests.

### Task 1: Stage 3 Hill Core

**Files:**
- Create: `src/core/stage3_hill.py`
- Test: `tests/test_stage3_hill.py`

- [ ] **Step 1: Write the failing Hill-core tests**

```python
import math

import pandas as pd
import torch

from src.core.stage3_hill import (
    FixedHillCurve,
    LearnableHillCurve,
    build_delta_lambda_table,
    hill_delta_lambda,
    soft_argmax_peak_nm,
)


def test_hill_delta_lambda_matches_expected_scalar_value():
    conc = torch.tensor([[10.0]], dtype=torch.float32)
    result = hill_delta_lambda(conc, delta_lambda_max=8.0, k_half=10.0, hill_n=1.0)
    assert torch.allclose(result, torch.tensor([[4.0]], dtype=torch.float32), atol=1e-6)


def test_build_delta_lambda_table_pairs_bsa_and_ag_rows():
    df = pd.DataFrame(
        [
            {"sample_id": "S1", "stage": "BSA", "concentration_ng_ml": 1.0, "peak_wavelength_nm": 602.0},
            {"sample_id": "S1", "stage": "Ag", "concentration_ng_ml": 1.0, "peak_wavelength_nm": 604.5},
        ]
    )
    paired = build_delta_lambda_table(df)
    assert list(paired.columns) == ["sample_id", "concentration_ng_ml", "lambda_bsa_nm", "lambda_ag_nm", "delta_lambda_nm"]
    assert paired.iloc[0]["delta_lambda_nm"] == 2.5


def test_soft_argmax_peak_nm_recovers_single_peak_inside_window():
    wavelengths = torch.tensor([600.0, 601.0, 602.0, 603.0, 604.0], dtype=torch.float32)
    spectra = torch.tensor([[0.0, 0.0, 1.0, 4.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([False, True, True, True, False])
    peak_nm = soft_argmax_peak_nm(spectra, wavelengths, mask, temperature=0.25)
    assert 602.4 <= float(peak_nm.item()) <= 603.2


def test_learnable_hill_curve_stays_positive_and_has_regularization():
    module = LearnableHillCurve(delta_lambda_max=8.0, k_half=10.0, hill_n=1.4)
    out = module(torch.tensor([[5.0]], dtype=torch.float32))
    assert float(out.item()) > 0.0
    assert float(module.regularization_loss().item()) == 0.0


def test_fixed_hill_curve_matches_closed_form():
    module = FixedHillCurve(delta_lambda_max=8.0, k_half=10.0, hill_n=1.0)
    out = module(torch.tensor([[10.0]], dtype=torch.float32))
    assert math.isclose(float(out.item()), 4.0, rel_tol=1e-6)
```

- [ ] **Step 2: Run the Hill-core tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_stage3_hill.py -q
```

Expected: FAIL with `ImportError` or `ModuleNotFoundError` for `src.core.stage3_hill`.

- [ ] **Step 3: Write the minimal Hill-core implementation**

```python
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn


def hill_delta_lambda(concentration_ng_ml: torch.Tensor, delta_lambda_max: float, k_half: float, hill_n: float):
    conc = torch.clamp(concentration_ng_ml, min=0.0)
    numerator = delta_lambda_max * torch.pow(conc, hill_n)
    denominator = torch.pow(torch.tensor(k_half, dtype=conc.dtype, device=conc.device), hill_n) + torch.pow(conc, hill_n)
    return numerator / (denominator + 1e-8)


def build_delta_lambda_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "stage", "concentration_ng_ml", "peak_wavelength_nm"}
    missing = required.difference(feature_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    work = feature_df.copy()
    work["stage"] = work["stage"].astype(str).str.upper()
    bsa = work[work["stage"] == "BSA"][["sample_id", "concentration_ng_ml", "peak_wavelength_nm"]].rename(columns={"peak_wavelength_nm": "lambda_bsa_nm"})
    ag = work[work["stage"] == "AG"][["sample_id", "concentration_ng_ml", "peak_wavelength_nm"]].rename(columns={"peak_wavelength_nm": "lambda_ag_nm"})
    merged = ag.merge(bsa, on=["sample_id", "concentration_ng_ml"], how="inner")
    merged["delta_lambda_nm"] = merged["lambda_ag_nm"] - merged["lambda_bsa_nm"]
    return merged.sort_values(["concentration_ng_ml", "sample_id"]).reset_index(drop=True)


def soft_argmax_peak_nm(spectra: torch.Tensor, wavelengths_nm: torch.Tensor, window_mask: torch.Tensor, temperature: float):
    masked = spectra[:, window_mask]
    wl = wavelengths_nm[window_mask].to(masked.device)
    scaled = masked / max(float(temperature), 1e-6)
    weights = torch.softmax(scaled, dim=1)
    return torch.sum(weights * wl.unsqueeze(0), dim=1, keepdim=True)


class FixedHillCurve(nn.Module):
    def __init__(self, delta_lambda_max: float, k_half: float, hill_n: float):
        super().__init__()
        self.register_buffer("delta_lambda_max", torch.tensor(float(delta_lambda_max)))
        self.register_buffer("k_half", torch.tensor(float(k_half)))
        self.register_buffer("hill_n", torch.tensor(float(hill_n)))

    def forward(self, concentration_ng_ml: torch.Tensor) -> torch.Tensor:
        return hill_delta_lambda(concentration_ng_ml, float(self.delta_lambda_max), float(self.k_half), float(self.hill_n))

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.delta_lambda_max.dtype, device=self.delta_lambda_max.device)


class LearnableHillCurve(nn.Module):
    def __init__(self, delta_lambda_max: float, k_half: float, hill_n: float):
        super().__init__()
        self.delta_lambda_max_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(delta_lambda_max)))))
        self.k_half_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(k_half)))))
        self.hill_n_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(hill_n - 1.0)))))
        self.register_buffer("delta_lambda_max_init", torch.tensor(float(delta_lambda_max)))
        self.register_buffer("k_half_init", torch.tensor(float(k_half)))
        self.register_buffer("hill_n_init", torch.tensor(float(hill_n)))

    def constrained_parameters(self):
        delta_lambda_max = torch.nn.functional.softplus(self.delta_lambda_max_raw)
        k_half = torch.nn.functional.softplus(self.k_half_raw) + 1e-6
        hill_n = 1.0 + torch.nn.functional.softplus(self.hill_n_raw)
        return delta_lambda_max, k_half, hill_n

    def forward(self, concentration_ng_ml: torch.Tensor) -> torch.Tensor:
        delta_lambda_max, k_half, hill_n = self.constrained_parameters()
        return hill_delta_lambda(concentration_ng_ml, float(delta_lambda_max), float(k_half), float(hill_n))

    def regularization_loss(self) -> torch.Tensor:
        delta_lambda_max, k_half, hill_n = self.constrained_parameters()
        return (delta_lambda_max - self.delta_lambda_max_init) ** 2 + (k_half - self.k_half_init) ** 2 + (hill_n - self.hill_n_init) ** 2
```

- [ ] **Step 4: Run the Hill-core tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_stage3_hill.py -q
```

Expected: `5 passed`

- [ ] **Step 5: Commit the Hill-core module**

```bash
git add src/core/stage3_hill.py tests/test_stage3_hill.py
git commit -m "feat: add stage3 hill core"
```

### Task 2: Stage 3 Profiles And Fixed-Parameter Fitting

**Files:**
- Create: `src/core/stage3_config.py`
- Create: `scripts/fit_stage3_hill_params.py`
- Create: `tests/test_stage3_config.py`

- [ ] **Step 1: Write the failing Stage 3 config tests**

```python
import math
from argparse import Namespace

from src.core.stage3_config import apply_stage3_profile_overrides, build_stage3_profile


def test_build_stage3_profile_3a_uses_frozen_predictor_and_fixed_hill():
    profile = build_stage3_profile("3A-fixed-frozen")
    assert profile.base_stage25_profile == "2.5C"
    assert profile.predictor_train_mode == "frozen"
    assert profile.hill_mode == "fixed"
    assert math.isclose(profile.w_hill, 0.05)


def test_build_stage3_profile_3c_uses_learnable_hill_and_regressor():
    profile = build_stage3_profile("3C-learnable-regressor")
    assert profile.predictor_train_mode == "regressor"
    assert profile.hill_mode == "learnable_kn"


def test_apply_stage3_profile_overrides_sets_args():
    args = Namespace(stage3_profile=None, predictor_train_mode="all", w_hill=0.0, hill_mode="off")
    profile = build_stage3_profile("3B-fixed-regressor")
    apply_stage3_profile_overrides(args, profile)
    assert args.stage3_profile == "3B-fixed-regressor"
    assert args.predictor_train_mode == "regressor"
    assert args.hill_mode == "fixed"
```

- [ ] **Step 2: Run the Stage 3 config tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_stage3_config.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `src.core.stage3_config`.

- [ ] **Step 3: Write the Stage 3 profile and fitting implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3Profile:
    name: str
    base_stage25_profile: str
    predictor_train_mode: str
    predictor_lr: float
    generator_lr: float
    p_steps: int
    g_steps: int
    w_cycle: float
    w_mono: float
    w_recon: float
    w_hill: float
    hill_mode: str
    hill_reg_weight: float


_PROFILE_MAP = {
    "3A-FIXED-FROZEN": Stage3Profile("3A-fixed-frozen", "2.5C", "frozen", 0.0, 1e-4, 0, 1, 0.003, 0.05, 0.02, 0.05, "fixed", 0.0),
    "3B-FIXED-REGRESSOR": Stage3Profile("3B-fixed-regressor", "2.5C", "regressor", 1e-4, 1e-4, 1, 1, 0.003, 0.05, 0.02, 0.1, "fixed", 0.0),
    "3C-LEARNABLE-REGRESSOR": Stage3Profile("3C-learnable-regressor", "2.5C", "regressor", 1e-4, 1e-4, 1, 1, 0.003, 0.05, 0.02, 0.1, "learnable_kn", 1e-3),
}


def build_stage3_profile(name: str) -> Stage3Profile:
    key = name.strip().upper()
    if key not in _PROFILE_MAP:
        raise ValueError(f"Unsupported Stage 3 profile: {name}")
    return _PROFILE_MAP[key]


def apply_stage3_profile_overrides(args, profile: Stage3Profile):
    args.stage3_profile = profile.name
    args.predictor_train_mode = profile.predictor_train_mode
    args.predictor_lr = profile.predictor_lr
    args.generator_lr = profile.generator_lr
    args.p_steps = profile.p_steps
    args.g_steps = profile.g_steps
    args.w_cycle = profile.w_cycle
    args.w_mono = profile.w_mono
    args.w_recon = profile.w_recon
    args.w_hill = profile.w_hill
    args.hill_mode = profile.hill_mode
    args.hill_reg_weight = profile.hill_reg_weight
    return args
```

```python
import argparse
from pathlib import Path

import pandas as pd
import torch

from src.core.stage3_hill import build_delta_lambda_table


def fit_fixed_hill_params(delta_df: pd.DataFrame):
    grouped = delta_df.groupby("concentration_ng_ml", as_index=False)["delta_lambda_nm"].median()
    conc = torch.tensor(grouped["concentration_ng_ml"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    target = torch.tensor(grouped["delta_lambda_nm"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    raw = torch.nn.Parameter(torch.tensor([5.0, 10.0, 0.5], dtype=torch.float32))
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(1500):
        opt.zero_grad()
        delta_max = torch.nn.functional.softplus(raw[0])
        k_half = torch.nn.functional.softplus(raw[1]) + 1e-6
        hill_n = 1.0 + torch.nn.functional.softplus(raw[2])
        pred = delta_max * conc.pow(hill_n) / (k_half.pow(hill_n) + conc.pow(hill_n) + 1e-8)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
    return {"delta_lambda_max": float(delta_max.detach()), "k_half": float(k_half.detach()), "hill_n": float(hill_n.detach())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-xlsx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_excel(args.features_xlsx, sheet_name="sheet1")
    params = fit_fixed_hill_params(build_delta_lambda_table(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, args.out)
```

- [ ] **Step 4: Run the Stage 3 config tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_stage3_config.py -q
```

Expected: `3 passed`

- [ ] **Step 5: Commit the Stage 3 config and fixed-fit script**

```bash
git add src/core/stage3_config.py scripts/fit_stage3_hill_params.py tests/test_stage3_config.py
git commit -m "feat: add stage3 profiles and hill parameter fitter"
```

### Task 3: Hill-Aware Alternating Training Primitives

**Files:**
- Create: `src/core/stage3_training.py`
- Create: `tests/test_stage3_training.py`

- [ ] **Step 1: Write the failing stage3-training tests**

```python
import torch
import torch.nn as nn

from src.core.stage3_hill import FixedHillCurve
from src.core.stage3_training import generator_step_with_hill, run_stage3_alternating_epoch


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 1)

    def forward(self, x_spectrum, x_physics):
        flat = x_spectrum.view(x_spectrum.size(0), -1)
        return self.linear(torch.cat([flat, x_physics], dim=1))


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, y_log):
        return self.linear(y_log).unsqueeze(1)


def _build_batch():
    xb = torch.tensor([[[0.1, 0.2], [0.0, 0.1]], [[0.3, 0.4], [0.1, 0.0]]], dtype=torch.float32)
    pb = torch.tensor([[603.0, 0.1, 0.2], [603.5, 0.2, 0.1]], dtype=torch.float32)
    yb = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    rb = xb[:, 0, :]
    wavelengths = torch.tensor([602.0, 603.0], dtype=torch.float32)
    return xb, pb, yb, rb, wavelengths


def test_generator_step_with_hill_reports_loss_hill():
    predictor = TinyPredictor()
    generator = TinyGenerator()
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    xb, pb, yb, rb, wavelengths = _build_batch()
    losses = generator_step_with_hill(
        predictor=predictor,
        generator=generator,
        batch=(xb, pb, yb, rb),
        wavelengths_nm=wavelengths,
        generator_optimizer=optimizer,
        hill_curve=FixedHillCurve(delta_lambda_max=5.0, k_half=10.0, hill_n=1.2),
        hill_weight=0.1,
        cycle_weight=0.01,
        recon_weight=0.05,
        hill_window_center_nm=603.0,
        hill_window_half_width_nm=1.0,
        hill_temperature=0.25,
        hill_reg_weight=0.0,
    )
    assert losses["loss_hill"] >= 0


def test_run_stage3_alternating_epoch_counts_hill_generator_steps():
    predictor = TinyPredictor()
    generator = TinyGenerator()
    predictor_optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    xb, pb, yb, rb, wavelengths = _build_batch()
    stats = run_stage3_alternating_epoch(
        predictor=predictor,
        generator=generator,
        train_batches=[(xb, pb, yb, rb)],
        wavelengths_nm=wavelengths,
        predictor_optimizer=predictor_optimizer,
        generator_optimizer=generator_optimizer,
        hill_curve=FixedHillCurve(delta_lambda_max=5.0, k_half=10.0, hill_n=1.2),
        p_steps=0,
        g_steps=1,
        mono_weight=0.05,
        cycle_weight=0.01,
        recon_weight=0.05,
        hill_weight=0.1,
        hill_window_center_nm=603.0,
        hill_window_half_width_nm=1.0,
        hill_temperature=0.25,
        hill_reg_weight=0.0,
    )
    assert stats["generator_steps"] == 1
    assert stats["generator_loss_hill"] >= 0
```

- [ ] **Step 2: Run the stage3-training tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_stage3_training.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `src.core.stage3_training`.

- [ ] **Step 3: Write the minimal hill-aware alternating training implementation**

```python
from typing import Iterable

import torch
import torch.nn.functional as F

from src.core.stage25_training import predictor_step
from src.core.stage3_hill import soft_argmax_peak_nm


def _tensor_gradient_1d(x: torch.Tensor) -> torch.Tensor:
    left = x[:, 1:2] - x[:, 0:1]
    mid = (x[:, 2:] - x[:, :-2]) * 0.5
    right = x[:, -1:] - x[:, -2:-1]
    return torch.cat([left, mid, right], dim=1)


def generator_step_with_hill(
    predictor,
    generator,
    batch,
    wavelengths_nm,
    generator_optimizer,
    hill_curve,
    hill_weight,
    cycle_weight,
    recon_weight,
    hill_window_center_nm,
    hill_window_half_width_nm,
    hill_temperature,
    hill_reg_weight,
):
    _xb, pb, yb, rb = batch
    old_flags = [p.requires_grad for p in predictor.parameters()]
    for param in predictor.parameters():
        param.requires_grad = False
    generator_optimizer.zero_grad()
    gen_raw = generator(yb).squeeze(1)
    gen_diff = _tensor_gradient_1d(gen_raw)
    x_gen = torch.stack([gen_raw, gen_diff], dim=1)
    pred_cycle = predictor(x_gen, pb)
    loss_cycle = F.mse_loss(pred_cycle, yb)
    loss_recon = F.mse_loss(gen_raw, rb)
    mask = torch.abs(wavelengths_nm - hill_window_center_nm) <= hill_window_half_width_nm
    lambda_ag_hat = soft_argmax_peak_nm(gen_raw, wavelengths_nm, mask, hill_temperature)
    lambda_bsa = pb[:, 0:1]
    conc_ng_ml = torch.clamp(torch.pow(10.0, yb) - 1e-3, min=0.0)
    delta_lambda_hat = lambda_ag_hat - lambda_bsa
    delta_lambda_target = hill_curve(conc_ng_ml)
    loss_hill = F.mse_loss(delta_lambda_hat, delta_lambda_target)
    loss_reg = hill_curve.regularization_loss() if hasattr(hill_curve, "regularization_loss") else torch.zeros_like(loss_hill)
    loss = cycle_weight * loss_cycle + recon_weight * loss_recon + hill_weight * loss_hill + hill_reg_weight * loss_reg
    loss.backward()
    generator_optimizer.step()
    for param, flag in zip(predictor.parameters(), old_flags):
        param.requires_grad = flag
    return {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_cycle": float(loss_cycle.detach().cpu().item()),
        "loss_recon": float(loss_recon.detach().cpu().item()),
        "loss_hill": float(loss_hill.detach().cpu().item()),
        "loss_hill_reg": float(loss_reg.detach().cpu().item()),
    }


def run_stage3_alternating_epoch(
    predictor,
    generator,
    train_batches: Iterable,
    wavelengths_nm,
    predictor_optimizer,
    generator_optimizer,
    hill_curve,
    p_steps,
    g_steps,
    mono_weight,
    cycle_weight,
    recon_weight,
    hill_weight,
    hill_window_center_nm,
    hill_window_half_width_nm,
    hill_temperature,
    hill_reg_weight,
):
    predictor_losses = []
    generator_losses = []
    predictor_steps = 0
    generator_steps = 0
    for batch in train_batches:
        for _ in range(max(p_steps, 0)):
            predictor_losses.append(predictor_step(predictor, batch, predictor_optimizer, mono_weight))
            predictor_steps += 1
        for _ in range(max(g_steps, 0)):
            generator_losses.append(
                generator_step_with_hill(
                    predictor=predictor,
                    generator=generator,
                    batch=batch,
                    wavelengths_nm=wavelengths_nm,
                    generator_optimizer=generator_optimizer,
                    hill_curve=hill_curve,
                    hill_weight=hill_weight,
                    cycle_weight=cycle_weight,
                    recon_weight=recon_weight,
                    hill_window_center_nm=hill_window_center_nm,
                    hill_window_half_width_nm=hill_window_half_width_nm,
                    hill_temperature=hill_temperature,
                    hill_reg_weight=hill_reg_weight,
                )
            )
            generator_steps += 1
    mean_hill = sum(item["loss_hill"] for item in generator_losses) / len(generator_losses) if generator_losses else 0.0
    return {
        "predictor_steps": predictor_steps,
        "generator_steps": generator_steps,
        "predictor_loss": sum(item["loss_total"] for item in predictor_losses) / len(predictor_losses) if predictor_losses else 0.0,
        "generator_loss": sum(item["loss_total"] for item in generator_losses) / len(generator_losses) if generator_losses else 0.0,
        "generator_loss_hill": mean_hill,
    }
```

- [ ] **Step 4: Run the stage3-training tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_stage3_training.py -q
```

Expected: `2 passed`

- [ ] **Step 5: Commit the stage3 training primitives**

```bash
git add src/core/stage3_training.py tests/test_stage3_training.py
git commit -m "feat: add stage3 hill-aware training primitives"
```

### Task 4: Integrate Stage 3 Into The Joint Training Script

**Files:**
- Modify: `scripts/train_joint_physics_dl.py`
- Create: `tests/test_train_joint_physics_dl_stage3.py`

- [ ] **Step 1: Write the failing stage3 training-script tests**

```python
from argparse import Namespace

from scripts import train_joint_physics_dl as joint_script
from src.core.stage3_config import build_stage3_profile


def test_apply_stage3_profile_overrides_training_args():
    args = Namespace(stage3_profile=None, predictor_train_mode="all", hill_mode="off", w_hill=0.0)
    profile = build_stage3_profile("3B-fixed-regressor")
    joint_script.apply_stage3_profile(args, profile)
    assert args.stage3_profile == "3B-fixed-regressor"
    assert args.predictor_train_mode == "regressor"
    assert args.hill_mode == "fixed"
    assert args.w_hill == 0.1


def test_build_output_tag_prefers_stage3_profile_name():
    assert joint_script.build_output_tag(stage25_profile=None, stage3_profile="3A-fixed-frozen") == "stage3_3a_fixed_frozen"


def test_run_joint_training_epoch_uses_stage3_runner_when_hill_enabled():
    calls = []

    def fake_stage3_runner(**kwargs):
        calls.append(kwargs)
        return {"predictor_steps": 0, "generator_steps": 1, "predictor_loss": 0.0, "generator_loss": 0.1, "generator_loss_hill": 0.02}

    stats = joint_script.run_joint_training_epoch(
        predictor="predictor",
        generator="generator",
        train_loader=["batch"],
        predictor_optimizer=None,
        generator_optimizer="g-opt",
        update_strategy="alternating",
        mono_weight=0.05,
        cycle_weight=0.01,
        recon_weight=0.05,
        p_steps=0,
        g_steps=1,
        hill_weight=0.1,
        stage3_runner=fake_stage3_runner,
        hill_context={"enabled": True},
    )
    assert stats["generator_loss_hill"] == 0.02
    assert calls
```

- [ ] **Step 2: Run the stage3 training-script tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_train_joint_physics_dl_stage3.py -q
```

Expected: FAIL with missing `apply_stage3_profile`, unexpected `build_output_tag` signature, or missing `stage3_runner`.

- [ ] **Step 3: Integrate stage3 CLI, profile loading, hill module loading, and stage3 output tags**

```python
from src.core.stage3_config import apply_stage3_profile_overrides, build_stage3_profile
from src.core.stage3_hill import FixedHillCurve, LearnableHillCurve
from src.core.stage3_training import run_stage3_alternating_epoch


def apply_stage3_profile(args, profile):
    return apply_stage3_profile_overrides(args, profile)


def build_output_tag(stage25_profile: str, stage3_profile: str = "") -> str:
    if stage3_profile:
        return "stage3_" + stage3_profile.lower().replace("-", "_")
    if stage25_profile:
        return "stage25_" + stage25_profile.lower().replace(".", "p")
    return "cycle"
```

```python
parser.add_argument("--stage3-profile", type=str, choices=["3A-fixed-frozen", "3B-fixed-regressor", "3C-learnable-regressor"])
parser.add_argument("--hill-mode", type=str, default="off", choices=["off", "fixed", "learnable_kn", "learnable_all"])
parser.add_argument("--hill-params-path", type=str, default=os.path.join(root, "models", "pretrained", "stage3_hill_params.pth"))
parser.add_argument("--w-hill", type=float, default=0.0)
parser.add_argument("--hill-reg-weight", type=float, default=0.0)
parser.add_argument("--hill-window-center-nm", type=float, default=603.0)
parser.add_argument("--hill-window-half-width-nm", type=float, default=15.0)
parser.add_argument("--hill-temperature", type=float, default=0.25)
if args.stage3_profile:
    apply_stage3_profile(args, build_stage3_profile(args.stage3_profile))
```

```python
if hill_context and hill_context["enabled"]:
    stats = stage3_runner(
        predictor=predictor,
        generator=generator,
        train_batches=train_loader,
        wavelengths_nm=hill_context["wavelengths_nm"],
        predictor_optimizer=predictor_optimizer,
        generator_optimizer=generator_optimizer,
        hill_curve=hill_context["hill_curve"],
        p_steps=p_steps,
        g_steps=g_steps,
        mono_weight=mono_weight,
        cycle_weight=cycle_weight,
        recon_weight=recon_weight,
        hill_weight=hill_weight,
        hill_window_center_nm=hill_context["hill_window_center_nm"],
        hill_window_half_width_nm=hill_context["hill_window_half_width_nm"],
        hill_temperature=hill_context["hill_temperature"],
        hill_reg_weight=hill_context["hill_reg_weight"],
    )
```

- [ ] **Step 4: Run the stage3 training-script tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_train_joint_physics_dl_stage3.py -q
```

Expected: `3 passed`

- [ ] **Step 5: Commit the stage3 training-script integration**

```bash
git add scripts/train_joint_physics_dl.py tests/test_train_joint_physics_dl_stage3.py
git commit -m "feat: integrate stage3 hill training into joint script"
```

### Task 5: Stage 3 Experiment Runner

**Files:**
- Create: `scripts/run_stage3_experiment.py`
- Create: `tests/test_run_stage3_experiment.py`

- [ ] **Step 1: Write the failing stage3-runner tests**

```python
from pathlib import Path

from scripts import run_stage3_experiment as run_script


def test_build_run_name_uses_stage3_profile_and_seed():
    assert run_script.build_run_name("3A-fixed-frozen", 20260325) == "stage3_3a_fixed_frozen_seed20260325"


def test_build_train_command_includes_stage3_profile(tmp_path: Path):
    paths = {
        "train": tmp_path / "train.xlsx",
        "val": tmp_path / "val.xlsx",
        "test": tmp_path / "test.xlsx",
        "features_xlsx": tmp_path / "features.xlsx",
        "fusion_weights": tmp_path / "fusion.pth",
        "fusion_norm": tmp_path / "norm.pth",
        "generator_weights": tmp_path / "generator.pth",
        "hill_params": tmp_path / "hill.pth",
    }
    command = run_script.build_train_command(paths=paths, profile="3A-fixed-frozen", seed=20260325, joint_epochs=60, pretrain_gen_epochs=5)
    assert "--stage3-profile" in command
    assert "3A-fixed-frozen" in command
    assert "--hill-params-path" in command
```

- [ ] **Step 2: Run the stage3-runner tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_run_stage3_experiment.py -q
```

Expected: FAIL with `ImportError` for `scripts.run_stage3_experiment`.

- [ ] **Step 3: Write the minimal stage3 runner**

```python
import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master")


def build_run_name(profile: str, seed: int, suffix: str = "") -> str:
    base = "stage3_" + profile.lower().replace("-", "_") + f"_seed{seed}"
    return base if not suffix else f"{base}_{suffix}"


def build_stage3_run_paths(source_root: Path):
    split_root = source_root / "data" / "processed" / "splits_reconstructed"
    pretrained_root = source_root / "models" / "pretrained"
    return {
        "train": split_root / "train_preprocessed_pairs.xlsx",
        "val": split_root / "val_preprocessed_pairs.xlsx",
        "test": split_root / "test_preprocessed_pairs.xlsx",
        "features_xlsx": source_root / "data" / "processed" / "Reconstructed_Preprocessed_Features_and_Delta.xlsx",
        "fusion_weights": pretrained_root / "spectral_predictor_v2_fusion.pth",
        "fusion_norm": pretrained_root / "predictor_v2_fusion_norm_params.pth",
        "generator_weights": pretrained_root / "spectral_generator_cycle.pth",
        "hill_params": pretrained_root / "stage3_hill_params.pth",
    }


def build_train_command(paths, profile: str, seed: int, joint_epochs: int, pretrain_gen_epochs: int):
    return [
        "python",
        "scripts/train_joint_physics_dl.py",
        "--stage3-profile",
        profile,
        "--seed",
        str(seed),
        "--joint-epochs",
        str(joint_epochs),
        "--pretrain-gen-epochs",
        str(pretrain_gen_epochs),
        "--train",
        str(paths["train"]),
        "--val",
        str(paths["val"]),
        "--test",
        str(paths["test"]),
        "--features-xlsx",
        str(paths["features_xlsx"]),
        "--fusion-weights",
        str(paths["fusion_weights"]),
        "--fusion-norm",
        str(paths["fusion_norm"]),
        "--generator-weights",
        str(paths["generator_weights"]),
        "--hill-params-path",
        str(paths["hill_params"]),
    ]
```

- [ ] **Step 4: Run the stage3-runner tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest tests/test_run_stage3_experiment.py -q
```

Expected: `2 passed`

- [ ] **Step 5: Commit the stage3 runner**

```bash
git add scripts/run_stage3_experiment.py tests/test_run_stage3_experiment.py
git commit -m "feat: add stage3 experiment runner"
```

### Task 6: Fixed-Hill Artifact And 3A Smoke Run

**Files:**
- Modify: `models/pretrained/stage3_hill_params.pth` (generated artifact)
- Output: `outputs/stage3_3a_fixed_frozen_seed20260325/`

- [ ] **Step 1: Generate the fixed Hill parameter artifact**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python scripts/fit_stage3_hill_params.py `
  --features-xlsx "C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master/data/processed/Reconstructed_Preprocessed_Features_and_Delta.xlsx" `
  --out "C:/Users/Spc/.config/superpowers/worktrees/LSPR_Spectra_Master/stage25-alternating-joint-training/models/pretrained/stage3_hill_params.pth"
```

Expected: exit 0 and a new `stage3_hill_params.pth` file in `models/pretrained/`.

- [ ] **Step 2: Dry-run the stage3 experiment runner**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python scripts/run_stage3_experiment.py `
  --profile 3A-fixed-frozen `
  --seed 20260325 `
  --dry-run
```

Expected: prints a `stage3_3a_fixed_frozen_seed20260325` run name and a command containing `--stage3-profile 3A-fixed-frozen`.

- [ ] **Step 3: Execute the 3A single-seed smoke run**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python scripts/run_stage3_experiment.py `
  --profile 3A-fixed-frozen `
  --seed 20260325
```

Expected:
- exit 0
- `outputs/stage3_3a_fixed_frozen_seed20260325/` created
- stage3 metrics CSV and checkpoint snapshot present

- [ ] **Step 4: Run the full stage3-related automated test suite**

Run:

```powershell
$env:PYTHONPATH='.'; conda run -n gan python -m pytest `
  tests/test_stage3_hill.py `
  tests/test_stage3_config.py `
  tests/test_stage3_training.py `
  tests/test_train_joint_physics_dl_stage3.py `
  tests/test_run_stage3_experiment.py -q
```

Expected: all stage3 tests pass.

- [ ] **Step 5: Commit the generated artifact path and smoke-run support code**

```bash
git add src/core/stage3_hill.py src/core/stage3_config.py src/core/stage3_training.py scripts/fit_stage3_hill_params.py scripts/run_stage3_experiment.py scripts/train_joint_physics_dl.py tests/test_stage3_hill.py tests/test_stage3_config.py tests/test_stage3_training.py tests/test_train_joint_physics_dl_stage3.py tests/test_run_stage3_experiment.py
git commit -m "feat: add stage3 fixed hill training pipeline"
```

## Self-Review

### Spec Coverage

- `L_hill` chain: covered by Tasks 1, 3, and 4.
- Fixed Hill parameters: covered by Task 2 and Task 6.
- `3A-fixed-frozen`, `3B-fixed-regressor`, `3C-learnable-regressor` profiles: covered by Task 2 and Task 4.
- Stage 3 experiment runner and seed snapshots: covered by Task 5 and Task 6.
- TDD and verification commands: present in every task.

### Placeholder Scan

- No unresolved placeholders remain.
- Every task names exact files and exact commands.
- Each code-changing step includes concrete code.

### Type Consistency

- Stage 3 profiles consistently use names `3A-fixed-frozen`, `3B-fixed-regressor`, and `3C-learnable-regressor`.
- Hill modules consistently expose `forward()` and `regularization_loss()`.
- Stage 3 runner consistently uses the `stage3_...` output-tag and run-name convention.
