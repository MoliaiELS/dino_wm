# DINO-WM FYP Progress

Last updated: 2026-09-14

## Current status

- Current phase: Phase 1 - dataset generation pilot
- Overall status: Phase 1 generator and audit implementation completed locally; remote validation pending
- Main task: PushT simulator-based recovery dynamics and visual representation experiments
- Runtime authority: remote server `idac_sever`
- Dataset authority: `/mnt/slurmfs-4090node3/user_data/yguo704/dino_wm_dataset`

## Confirmed decisions

- [x] Use simulation before real-robot experiments.
- [x] Do not require UR5 for the feasibility study.
- [x] Use PushT as the main manipulation environment.
- [x] Use PointMaze only as an engineering smoke-test environment.
- [x] Generate paired nominal, failure and recovery trajectories in simulation.
- [x] Treat fixed-budget `D_SFR_balanced` versus `D_SF_balanced` as the primary recovery-data comparison.
- [x] Run Experiment A before committing substantial compute to Experiment B.
- [x] Probe learned temporal predictive representations rather than unchanged raw frozen DINO features.
- [x] Maintain scenario-level train/validation/test splits to prevent branch leakage.

## Environment audit

### Local repository

- Branch: `main`
- Phase 0 tested commit: `87f0d20`
- Local project execution is prohibited by `AGENTS.md`.

### Remote repository and environment

- Remote repository matched local audited commit at the time of inspection.
- Non-interactive SSH requires `source ~/miniforge3/etc/profile.d/conda.sh` before `source bash.sh`.
- `bash.sh` activates the `dino_wm` environment and sets `DATASET_DIR`.
- Importing the common `env` package no longer loads PointMaze/MuJoCo for PushT.
- PointMaze's optional MuJoCo path still needs its own runtime dependencies/settings if that environment is used later; no system configuration was changed.

### Existing datasets

- PointMaze exists remotely.
- PointMaze tensor shapes:
  - states: `(2000, 100, 4)`
  - actions: `(2000, 100, 2)`
  - sequence lengths: 2000 trajectories, all length 100
- PushT data is not currently present under `DATASET_DIR`.

### PushT smoke probe

- Direct PushT module loading succeeded remotely without importing PointMaze.
- RGB observation: `(224, 224, 3)`, `uint8`.
- Proprioception: 4 dimensions in the current `with_velocity=True` setup.
- Existing state: 7 dimensions.
- Action: 2 dimensions.
- Repeating reset from the same current state produced maximum absolute difference `0.0`.
- Versioned simulator snapshots now include both bodies' position, velocity, angle, angular velocity, force, torque and center of gravity, plus task/control/RNG state.
- Restoring the same mid-episode snapshot produced identical oracle state, legacy state, coverage and RGB trajectories.

## Milestones

### Phase 0 - Runtime and environment

- [x] Make PushT import independent of PointMaze/MuJoCo.
- [x] Define one explicit relative-action convention: command `[-1, 1]^2`, scaled by 100 simulator pixels.
- [x] Add action clipping shared by the simulator, dataset metadata and planners.
- [x] Add complete simulator snapshot and restore support.
- [x] Add task-success evaluation based on object-goal coverage.
- [x] Recompute normalization statistics from valid frames in each training split and reuse them for validation.
- [x] Add and pass remote deterministic replay tests.

### Phase 1 - Dataset generation pilot

- [x] Implement a geometry-based oracle for the controlled goal-aligned translation pilot.
- [x] Implement balanced agent/object lateral perturbations with low/medium/high severity bins.
- [x] Implement exact `S/F1/F2/R` branching.
- [x] Store scenario IDs, pair IDs, complete branch snapshots, aligned simulator state, RGB, actions and metadata.
- [x] Split scenarios before window generation and keep every pair in one split.
- [ ] Generate approximately 200 pilot paired scenarios.
- [ ] Audit branch-state equality, temporal alignment and label balance.
- [ ] Estimate final sample size from pilot variance.

### Phase 2 - Experiment A

- [ ] Implement `StateWorldModel` using shared DINO-WM temporal/planning infrastructure.
- [ ] Train `D_S`, `D_SF`, `D_SFR`, `D_SF_balanced` and `D_SFR_balanced` models.
- [ ] Measure 1/5/10/20-step dynamics prediction.
- [ ] Measure counterfactual action ranking.
- [ ] Measure closed-loop recovery success, coverage, steps and action cost.
- [ ] Run at least 3 pilot seeds.
- [ ] Decide whether the recovery signal is strong enough to proceed to Experiment B.

### Phase 3 - Experiment B

- [ ] Implement temporal dynamics adapter/context representation over frozen DINO tokens.
- [ ] Define representation extraction consistently across datasets.
- [ ] Implement progress, off-nominal and recoverability probes.
- [ ] Add raw DINO, random adapter and oracle-state baselines.
- [ ] Implement goal-set latent planning objective.
- [ ] Evaluate visual closed-loop recovery.

### Phase 4 - OOD and final evaluation

- [ ] Held-out perturbation severity.
- [ ] Held-out perturbation type.
- [ ] Held-out initial object and goal configurations.
- [ ] Run 5 final seeds after configuration lock.
- [ ] Report paired confidence intervals and effect sizes.
- [ ] Produce final tables, plots and failure-case videos.

## Open risks and questions

- A finite-horizon recoverability dataset may be imbalanced because many open-tabletop states are eventually recoverable.
- The oracle controller must be strong enough that recovery labels reflect state recoverability rather than planner failure.
- A single goal image can bias visual planning toward an irrelevant final agent pose; a goal set is planned instead.
- Existing Hydra Submitit configs request H100 resources although the available cluster documentation primarily lists 4090/3090 nodes.

## Decision gates

### Gate A - Dataset validity

Proceed only if exact branch restoration, action validity, label balance and temporal alignment checks pass.

### Gate B - Recovery dynamics signal

Proceed to the full visual experiment only if `D_SFR_balanced` improves both counterfactual action ranking and closed-loop recovery over `D_SF_balanced`, with uncertainty reported.

### Gate C - Representation claim

Make a representation claim only if the learned temporal representation improves probes over raw DINO and random-adapter controls on scenario-disjoint test data.

## Run log

No training or data-generation job has been submitted yet. Phase 0 used short login-node CPU smoke tests only.

### 2026-09-14 14:18 - Phase 0 deterministic validation

- Phase/purpose: Phase 0 environment, normalization and planner-bound validation
- Git commit: `87f0d20`
- Command/config: `python -m pytest -q tests/test_pusht_phase0.py`
- Dataset path/version: synthetic temporary fixtures; no authoritative dataset read
- Seeds: simulator replay seed 7
- SLURM job ID/node: not applicable; remote login-node smoke test
- Log/output path: terminal output
- Status: completed
- Key metrics/error: 7 tests passed in 2.57 seconds; 4 dependency deprecation warnings; exact branch replay error was zero
- Decision/next action: Phase 0 validity gate passed; proceed to Phase 1 pilot generator

### 2026-09-14 14:19 - PushT visual smoke test

- Phase/purpose: Phase 0 rendering and deterministic branch visualization
- Git commit: `87f0d20`
- Command/config: `python visualize_pusht_phase0.py --output-dir plan_outputs/phase0_pusht_smoke`
- Dataset path/version: not applicable; simulation-generated frames
- Seeds: 7
- SLURM job ID/node: not applicable; remote login-node smoke test
- Log/output path: `~/dino_wm/plan_outputs/phase0_pusht_smoke/`
- Status: completed
- Key metrics/error: 10 RGB frames at 224x224; oracle replay max error 0.0; RGB replay max error 0; final coverage 0.3055
- Decision/next action: visual/runtime path is ready for Phase 1 data-generation work

Use the following template for every meaningful remote run:

```markdown
### YYYY-MM-DD HH:MM - Short run name

- Phase/purpose:
- Git commit:
- Command/config:
- Dataset path/version:
- Seeds:
- SLURM job ID/node:
- Log/output path:
- Status: planned | queued | running | completed | failed | cancelled
- Key metrics/error:
- Decision/next action:
```

## Changelog

### 2026-09-14

- Implemented the Phase 1 paired PushT generator, dataset variants, post-split window indices, structural audit, bootstrap intervals and pilot sample-size estimator; remote validation is pending.
- Scoped the pilot to goal-aligned translation with lateral perturbations; deferred rotation perturbations to OOD evaluation until a rotation-capable oracle is validated.
- Implemented the Phase 0 PushT runtime, action, snapshot, task-evaluation and normalization changes.
- Passed all 7 Phase 0 remote tests at commit `87f0d20`.
- Generated and verified the direct PushT visual smoke-test artifacts on the remote server.
- Read the current Notion feasibility plan.
- Audited the local DINO-WM dataset, model, planning and PushT environment code.
- Inspected the remote repository, Conda bootstrap and available dataset layout.
- Confirmed that simulation-first experiments are feasible without UR5.
- Selected PushT as the main environment and PointMaze as a pipeline smoke test.
- Defined the paired counterfactual dataset design and fixed-budget control.
- Identified the frozen-DINO representation issue and selected a learned temporal representation for Experiment B.
- Added project workflow and experiment rules to `AGENTS.md`.
