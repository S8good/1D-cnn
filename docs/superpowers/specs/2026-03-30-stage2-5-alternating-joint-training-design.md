# Stage 2.5 Alternating Joint Training Design

## Context

The current Stage 2 joint-training entrypoint is [`scripts/train_joint_physics_dl.py`](C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master/scripts/train_joint_physics_dl.py). It updates `SpectralPredictorV2_Fusion` and `SpectrumGenerator` with a single optimizer and a single backward pass over `L_conc + L_cycle + L_mono + L_recon`.

Documented Stage 2 results as of March 26, 2026 show that `C+Cycle(regressor)` did not produce stable gains over `Model C` across three seeds. The project goal for the next step is therefore not to continue minor weight tuning on the same synchronized update scheme, but to insert a mandatory Stage 2.5 gate that tests whether alternating updates can recover performance, reduce interference between predictor and generator, and provide a stable base for later `L_hill` integration.

## Problem Statement

The synchronized Stage 2 update scheme couples two different objectives too tightly:

- The predictor must preserve `Model C` regression quality on real spectra.
- The generator must learn a concentration-to-spectrum mapping useful for `L_cycle`.
- The shared backward pass allows cycle and reconstruction gradients to perturb the predictor even when those gradients do not improve the main regression task.

Without a cleaner optimization structure, adding `L_hill` in Stage 3 would stack a second uncertain constraint on top of an already unstable training backbone. That would make failure analysis ambiguous and reduce the value of later ablation evidence.

## Success Criteria

Stage 2.5 exists to answer one question before Stage 3 starts:

Can the project obtain a joint-training backbone with alternating updates such that cycle-based learning is at least non-destructive to the main regression task and stable enough to support Hill loss integration?

Primary success metrics are overall regression metrics:

- `MAE`
- `RMSE`
- `R2`

Secondary metrics are used as supporting evidence, not as first-pass decision makers:

- low-concentration error
- high-concentration error
- monotonicity violation rate
- run-to-run variance
- training success rate

## Approaches Considered

### Approach 1: Keep the current synchronized update and continue weight tuning

This is the lowest-effort option, but it is not recommended. Stage 2 already produced a negative result under the current optimization pattern. More weight sweeps would generate cost without changing the failure mode.

### Approach 2: Replace Stage 2 with a single balanced alternating-update version

This is simpler to document, but it hides the escalation and fallback logic. If the first alternating-update configuration fails, the project would still need an implicit downgrade path.

### Approach 3: Add a funnel-style Stage 2.5 with explicit downgrade levels

This is the recommended approach. Stage 2.5 becomes a mandatory gate between Stage 2 and Stage 3 and is internally decomposed into:

- `Stage 2.5A` aggressive
- `Stage 2.5B` balanced
- `Stage 2.5C` conservative

The team starts at the highest-upside configuration and only relaxes ambition when the evidence says it is necessary. This keeps the optimization effort disciplined and makes the decision to enter Stage 3 auditable.

## Recommended Design

### 1. Stage graph change

The project phase order changes from:

`Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5`

to:

`Stage 0 -> Stage 1 -> Stage 2 -> Stage 2.5 -> Stage 3 -> Stage 4 -> Stage 5`

Stage 2 remains the synchronized-update baseline and negative reference. Stage 2.5 is the optimization-repair gate.

### 2. Stage 2.5 objective

Stage 2.5 will redesign the joint optimization logic around alternating `P-step` and `G-step` updates.

The design intent is:

- Predictor updates should stay dominated by real-spectrum concentration supervision.
- Generator updates should improve synthetic-spectrum usability without destabilizing the predictor.
- The joint path should become stable enough that adding `L_hill` in Stage 3 tests Hill supervision itself, not unresolved Stage 2 interference.

### 3. Stage 2.5A, 2.5B, 2.5C ladder

#### Stage 2.5A: Aggressive

Purpose: pursue an actual regression improvement over `Model C`.

Allowed mechanisms:

- alternating `P-step` and `G-step`
- asymmetric learning rates
- update frequency ratios such as `P:G = 1:1`, `2:1`, or `1:2`
- staged unfreezing of predictor submodules
- controlled reuse of `L_recon` and `L_cycle`

Decision target:

- beat `Model C` on the primary metrics or produce a clearly positive tradeoff

#### Stage 2.5B: Balanced

Purpose: reduce optimization freedom and retain only the structures that improve stability.

Allowed mechanisms:

- alternating `P-step` and `G-step`
- limited predictor finetuning scope
- a smaller hyperparameter search than 2.5A

Decision target:

- keep overall regression essentially non-degraded while improving training stability and variance behavior

#### Stage 2.5C: Conservative

Purpose: prove that joint training can stay usable without materially harming the main task.

Allowed mechanisms:

- alternating updates
- minimal predictor movement
- generator kept usable for later `L_hill` attachment

Decision target:

- establish a clean and stable mother configuration for Stage 3 even if standalone cycle gain is weak

### 4. Entry criteria for Stage 3

Stage 3 must not start immediately after any Stage 2.5 experiment. It only starts when one of the following conditions is satisfied.

#### Route A: Strong entry from Stage 2.5A

Across a three-seed paired comparison against `Model C` using the same split and the same seed set:

- either `MAE` or `RMSE` improves by at least `3%`, with `R2` no worse than `Model C - 0.01`
- or `R2` improves by at least `0.02`, with both `MAE` and `RMSE` worsening by no more than `2%`
- at least two of the three seeds have `MAE` no worse than their paired `Model C` runs
- no unstable run behavior is observed, including divergence, unusable checkpoints, or generator collapse

If Route A is satisfied, Stage 3 uses the best Stage 2.5A configuration as the parent configuration for `L_hill`.

#### Route B: Acceptable entry from Stage 2.5B

If Stage 2.5A fails to produce clear gains, Stage 2.5B may unlock Stage 3 when all of the following are true:

- `MAE` and `RMSE` each worsen by no more than `2%` relative to `Model C`
- `R2` drops by no more than `0.01`
- all three seeds converge successfully and produce usable weights
- standard deviation is no worse than the synchronized Stage 2 baseline

If Route B is satisfied, Stage 3 is allowed, but the paper/report narrative must describe cycle as a stabilizing or enabling mechanism rather than an independently proven accuracy booster.

#### Route C: Resource-limited entry from Stage 2.5C

If 2.5A and 2.5B fail, Stage 2.5C can unlock Stage 3 only under a conservative interpretation:

- `MAE` and `RMSE` each worsen by no more than `5%` relative to `Model C`
- `R2` drops by no more than `0.02`
- training success rate is `100%`
- generator outputs remain physically usable and do not show obvious spectral collapse

If Route C is used, Stage 3 must be framed as a test of whether Hill loss improves physical consistency and extreme-range behavior, not as a guaranteed path to better overall regression.

#### Blockers

Stage 3 is blocked if any of the following remain true after Stage 2.5:

- even Stage 2.5C shows clear overall regression degradation
- run-to-run variance grows beyond the synchronized Stage 2 baseline
- generator outputs collapse or become unusable for the cycle path
- predictor main-task performance is visibly dragged down by the joint path

### 5. Artifact retention and cleanup policy

Stage 2.5 is expected to generate many exploratory outputs. The cleanup rule is:

Keep:

- best weights for each retained substage
- three-seed summary tables
- final comparison plots
- minimal training logs needed to explain the decision
- phase conclusions in markdown or csv form

Delete:

- failed run checkpoints that are not referenced by any retained summary
- duplicate intermediate exports from the same configuration
- temporary figures generated only for quick inspection
- temporary csv files used only for one-off debugging or tuning
- cache-like artifacts such as `__pycache__`

The cleanup process must preserve stage evidence. It should remove redundancy, not destroy the reasoning trail.

### 6. Required updates to the main phase-plan document

The project plan document [`docx/PROJECT_PHASE_PLAN_CN.md`](C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master/docx/PROJECT_PHASE_PLAN_CN.md) will need these structural changes:

- insert `Stage 2.5` into the phase overview
- add a full Stage 2.5 section with the 2.5A/2.5B/2.5C ladder
- add explicit Stage 3 entry conditions
- update risk and fallback language to include automatic downgrade from 2.5A to 2.5B to 2.5C
- revise the weekly schedule so Stage 2.5 has its own execution window
- revise milestone definitions so the stable alternating-update backbone becomes the new milestone before Hill integration

## Scope Boundaries

This design does not yet specify exact code edits, exact command lines, or the final hyperparameter grid. Those belong in the implementation plan after this design is approved.

This design also does not declare that Stage 2.5 must succeed. It defines how success and failure are measured so that the team can stop guessing and make the Stage 3 decision on evidence.

## Testing and Validation Expectations

When implementation starts, validation must be based on:

- fixed split files
- fixed seed set for paired comparison
- the same evaluation script family used for `Model C`
- retained summary outputs that let the team compare `Model C`, Stage 2, and the final accepted Stage 2.5 configuration

## User-Approved Decisions Captured Here

- Use a funnel-style Stage 2.5, not a single undifferentiated alternation stage.
- Order of escalation is `aggressive -> balanced -> conservative`.
- Overall regression metrics take priority over physical-consistency-only narratives at the Stage 2.5 gate.
- Temporary waste files created during exploration should be cleaned up, while stage-level artifacts and decision evidence must be retained.
