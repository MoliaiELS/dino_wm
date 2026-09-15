# DINO-WM FYP Progress

Last updated: 2026-09-15

## Current status

- Current phase: Phase 2 - Experiment A
- Overall status: the success-matched Experiment A feasibility result now reproduces on 60 fresh scenarios; recovery-specific prediction/ranking and closed-loop success improve across three seeds, so Gate B passes with an explicit unresolved goal-retention limitation
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
- [x] Identify success-count imbalance as a mechanism-level confound in the original SF/SFR comparison.
- [x] Add an on-manifold nominal continuation `N` and `D_SFN_balanced` success-matched control.
- [x] Complete the three-seed `D_SFN_balanced` versus `D_SFR_balanced` recovery-specific attribution pilot.
- [x] Lock P3 on validation only and confirm the Experiment A result on 60 newly generated all-test scenarios.

## Environment audit

### Local repository

- Branch: `main`
- Phase 0 tested commit: `87f0d20`
- Phase 1 generated-data commit: `a976e2c`
- Phase 1 final regression commit: `a87a976`
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
- PushT Phase 1 pilot exists at `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`.
- PushT success-matched attribution dataset exists at `$DATASET_DIR/pusht_recovery_phase1_pilot_v2`.
- v1 size/content: 200 paired scenarios, 800 aligned MP4 branches, 21 MB total.
- v2 size/content: 200 paired scenarios, 1,000 aligned MP4 branches, 26 MB total.
- A single complete scenario plus full manifest/audit is tracked under `tests/fixtures/pusht_phase1_pilot_v1` for lightweight regression tests.

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
- [x] Implement balanced agent-lateral/agent-retreat perturbations with low/medium/high severity bins.
- [x] Implement exact `S/F1/F2/R` branching.
- [x] Store scenario IDs, pair IDs, complete branch snapshots, aligned simulator state, RGB, actions and metadata.
- [x] Split scenarios before window generation and keep every pair in one split.
- [x] Generate 200 pilot paired scenarios.
- [x] Audit branch-state equality, temporal alignment and perturbation-cell balance; Gate A passed.
- [x] Estimate final sample size from pilot variance and impose a conservative six-cell design floor of 180 scenarios.
- [x] Add `N`, action-phase labels, R/N distinctiveness audits and six-cell aligned visual diagnostics in schema v2.
- [x] Generate and audit the 200-pair v2 dataset with equal SFN/SFR success counts and windows.

### Phase 2 - Experiment A

- [x] Implement `StateWorldModel` using shared DINO-WM temporal/planning infrastructure.
- [~] Train `D_S`, `D_SF`, `D_SFR`, `D_SF_balanced` and `D_SFR_balanced` models.
- [x] Measure 1/5/10/20-step dynamics prediction for the primary fixed-budget comparison across 3 pilot seeds.
- [x] Measure counterfactual action ranking across 3 pilot seeds.
- [x] Measure closed-loop recovery success, coverage, steps and action cost across 3 pilot seeds.
- [x] Run at least 3 pilot seeds.
- [x] Apply Gate B after the three-seed pilot: not passed; remain in Phase 2 and repair the planner/model interface before Experiment B.
- [x] Run the success-matched SFN/SFR attribution comparison across seeds 0/1/2.
- [x] Aggregate R-branch and reposition/recontact-prefix prediction separately from mixed branch prediction.
- [x] Re-apply Gate B after attribution: recovery-specific dynamics signal passed, but end-to-end Gate B remains not passed because success-rate and final-coverage intervals cross zero.
- [x] Re-apply Gate B on the fresh confirmatory set: passed for the controlled feasibility claim; closed-loop success improves from 3/180 to 42/180 with paired 95% CI excluding zero.

### Phase 3 - Experiment B

- [~] Implement temporal dynamics adapter/context representation over frozen DINO tokens.
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
- The controlled pilot has perfect S/R versus F1/F2 separation; Experiment A must check that this does not make the learned comparison trivially saturated.
- Object displacement and rotation remain unvalidated OOD perturbations and require a rotation-capable oracle before use as recovery labels.
- A single goal image can bias visual planning toward an irrelevant final agent pose; a goal set is planned instead.
- Existing Hydra Submitit configs request H100 resources although the available cluster documentation primarily lists 4090/3090 nodes.
- Long-horizon CEM can exploit state-model error: one seed produced predicted near-goal costs while real coverage stayed near the branch state. Formal closed-loop evaluation must use calibrated short-horizon feedback planning and report action/coverage traces.
- Across three corrected seeds, recovery-rich models reach substantially higher peak coverage but often lose progress before the 35-step endpoint; success and final-coverage uncertainty still cross zero. Diagnose goal retention and replanning drift before changing the world-model claim.
- The success-matched v2 comparison removes the “more successful trajectories” explanation for prediction/ranking improvements, but it does not remove planner exploitation or goal-retention failure as explanations for weak final success.
- The original 30-scenario test set was used during early planner calibration and remains pilot-only evidence. This leakage concern is addressed for the present claim by validation-only P3 selection followed by the fresh seed-20260916 all-test set; any future planner change requires another fresh test.
- The locked-P3 fresh test passes the success criterion, but SFR retention loss is significantly worse; Experiment B must report this separately and must not reinterpret peak coverage as stable final-state control.

## Decision gates

### Gate A - Dataset validity

Proceed only if exact branch restoration, action validity, label balance and temporal alignment checks pass.

### Gate B - Recovery dynamics signal

Proceed to the full visual experiment only if `D_SFR_balanced` improves both counterfactual action ranking and closed-loop recovery over the fixed-budget controls; use `D_SFN_balanced` as the primary success-count-matched attribution control, with uncertainty reported.

### Gate C - Representation claim

Make a representation claim only if the learned temporal representation improves probes over raw DINO and random-adapter controls on scenario-disjoint test data.

## Run log

Phase 0 used short login-node CPU smoke tests only. Phase 1 batch generation is tracked below.

### 2026-09-15 15:05 - Experiment B cached-DINO pipeline implementation

- Phase/purpose: begin the minimum visual representation experiment after the controlled Experiment A Gate B pass
- Git commit: this implementation commit; exact hash to be recorded after remote regression
- Command/config: deterministically rerender every saved 27-D simulator state at 224 px; frozen `dinov2_vits14` patch tokens; fixed adaptive 4x4 spatial pooling; three-frame action-conditioned causal ViT adapter; visual-token plus proprio prediction loss
- Dataset path/version: source `$DATASET_DIR/pusht_recovery_phase1_pilot_v2`; planned cache `$DATASET_DIR/pusht_recovery_dino_cache_v2`; fresh source/cache kept separate
- Seeds: implementation tests first; visual training seeds planned as 0/1/2 after cache audit and one-seed smoke
- SLURM job ID/node: not submitted
- Log/output path: pending
- Status: implementation in progress
- Key metrics/error: added cache schema, deterministic sim-state renderer, leakage-safe visual windows, explicit probe labels, temporal world model, frozen ridge/logistic probe evaluation, paired seed×scenario aggregation and SLURM entry points; initial Phase 3 plus Phase 2 regression passed 14 tests; no representation-quality runtime claim yet
- Decision/next action: pass remote unit/integration tests, generate a small cache smoke subset, audit token/render consistency, then cache the full v2 data before any visual training

### 2026-09-15 12:10 - Fresh confirmatory dataset and locked-P3 evaluation

- Phase/purpose: test the validation-selected P3 planner on simulator scenes that were not used for model training, planner selection or earlier pilot reporting
- Git commit: dataset/evaluator `338634e`; result aggregation/plotting `6944a70`; report and variable-length trace fix `44d6950`
- Command/config: generate 60 `--all-test` scenarios with seed 20260916 and exactly 10 scenarios per perturbation type/severity cell; evaluate all SFN/SFR checkpoints with locked P3 (horizon 4, repeat 2, action norm cap 0.5, trajectory/progress/object-speed weights 0.5/1.0/0.5, minimum predicted improvement 0.005)
- Dataset path/version: `$DATASET_DIR/pusht_recovery_confirmatory_v2_seed20260916_60`; checkpoints remain under `$DATASET_DIR/phase2_runs/attribution_v2_65b750f`
- Seeds: new simulator base seed 20260916; training seeds 0/1/2
- SLURM job ID/node: generation `16268` on `4090node3`; closed-loop seed 0 `16269/16270`, seed 1 `16271/16272`, seed 2 `16274/16275`; fresh offline seed 0 `16276/16277`, seed 1 `16278/16279`, seed 2 `16280/16281`, all completed on 4090 nodes with exit code 0. Mistyped variant attempt `16273` failed before evaluation and produced no result; four initial offline submissions were rejected by the two-job QoS submit limit and were resubmitted sequentially.
- Log/output path: `$DATASET_DIR/logs/cf60-*-<job>.out`; per-checkpoint results `confirmatory_p3_seed20260916_60.json` and `confirmatory_offline_seed20260916_60.json`; aggregate `confirmatory_seed20260916_60_aggregate.json`
- Status: completed
- Key metrics/error: Gate A passed with 60 test scenarios, 10 per cell, S/N/R success 1.0, F1/F2 success 0, exact branch/action/alignment errors 0. Fresh recovery top-1 is SFN 0.661 versus SFR 0.994, paired delta +0.333 [0.211, 0.461]. Recovery-prefix 20-step object-goal RMSE is 15.924 versus 3.927 px. Closed-loop success is 3/180 versus 42/180, delta +0.217 [0.039, 0.417]; maximum coverage delta +0.203 [0.070, 0.348]; action cost delta -4.838 [-6.639, -2.989]. Final coverage delta +0.074 [-0.103, 0.278] remains uncertain, while retention-loss delta +0.129 [0.062, 0.210] is significantly worse for SFR. No evaluated action exceeds the 0.5 norm cap. The first figure render found and then fixed variable-length traces caused by early success termination; final remote aggregate/Phase-2 regression passed 13 tests and the report image rendered successfully.
- Decision/next action: Gate B passes for the controlled Experiment A feasibility claim because independent recovery ranking and closed-loop success both improve on fresh scenarios. Preserve the retention limitation, freeze this result, and begin only a small Experiment B implementation rather than enlarging the claim to OOD or real robotics.

### 2026-09-15 11:24 - Phase 2 validation planner selection locked

- Phase/purpose: select one planner configuration on validation data before generating an untouched confirmatory set
- Git commit: planner implementation `edf424f`; evaluation-only dataset mode is in the following implementation commit
- Command/config: P0 current short planner; P1 adds train-support norm cap/no-op; P2 adds two independent control blocks; P3 adds two control blocks plus trajectory, progress-regression and near-goal object-speed costs. P1 and P3 were expanded to all three training seeds; the selection score used absolute performance pooled across SFN and SFR, not the treatment-control delta.
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v2`, validation split only
- Seeds: training seeds 0/1/2
- SLURM job ID/node: P0 `16250/16251`; P1 seed 0 `16252/16253`, seed 1 `16260/16261`, seed 2 `16264/16265`; P2 seed 0 `16255/16256`; P3 seed 0 `16258/16259`, seed 1 `16262/16263`, seed 2 `16266/16267`; all completed on `4090node1` with exit code 0
- Log/output path: `$DATASET_DIR/logs/v2-*-p{0,1,2,3}*-<job>.out`; per-run JSON under `$DATASET_DIR/phase2_runs/attribution_v2_65b750f/<variant>/seed_<n>/validation_p*.json`; aggregates `validation_p1_aggregate.json` and `validation_p3_aggregate.json`
- Status: completed; P3 locked
- Key metrics/error: P3 pooled across both variants has success 0.072 versus P1 0.017, final coverage 0.403 versus 0.334, retention loss 0.162 versus 0.200 and action cost 13.23 versus 15.35. Under P3, SFN/SFR validation success is 2/90 versus 11/90; final coverage 0.397 versus 0.409, paired delta +0.012 [-0.102, 0.131]; maximum coverage 0.481 versus 0.648, delta +0.167 [0.039, 0.298]; action cost 14.66 versus 11.80, delta -2.86 [-4.44, -1.27]. Success delta +0.100 still has CI [-0.022, 0.233].
- Decision/next action: lock P3 exactly as horizon 4/repeat 2, norm cap 0.5, initial std 0.25, action/smoothness/staging weights 0.02/0.01/1.0, trajectory/progress/object-speed weights 0.5/1.0/0.5 and minimum predicted improvement 0.005. Do not tune further; generate 60 new all-test scenarios with 10 per perturbation cell and evaluate all six checkpoints.

### 2026-09-15 11:05 - Phase 2 validation-only planner repair implementation

- Phase/purpose: remove the observed planner/data-distribution mismatch without retraining or touching the test split for model selection
- Git commit: this implementation commit; exact hash to be recorded after remote validation
- Command/config: add explicit train/valid/test evaluator routing; norm-bounded CEM actions; persistent zero-action candidate; predicted-improvement safeguard; trajectory, progress-regression and near-goal object-speed costs; coverage-retention/action-distribution diagnostics
- Dataset path/version: implementation targets `$DATASET_DIR/pusht_recovery_phase1_pilot_v2`; no dataset mutation
- Seeds: existing SFN/SFR checkpoints, training seeds 0/1/2; initial planner selection will use validation scenarios only
- SLURM job ID/node: not submitted
- Log/output path: pending
- Status: implementation completed locally; remote regression pending
- Key metrics/error: prior diagnostic found R train action norm mean 0.085 and q99 0.502, while SFR CEM executed actions averaged 0.708 with 18.5% saturation; current horizon 4/repeat 4 exposes only one independent control block
- Decision/next action: pass remote regression, then screen a small predefined planner ladder on validation scenarios before any fresh confirmatory test

### 2026-09-15 10:32 - Phase 2 v2 three-seed attribution completed

- Phase/purpose: complete the success-count-matched `D_SFN_balanced` versus `D_SFR_balanced` feasibility experiment and decide whether the observed gain is recovery-specific
- Git commit: data/training `27a8323`; stratified evaluation `9631a08`; aggregation `17c9b25`
- Command/config: 50 epochs, batch 128, shared `D_SF` train-only normalization, identical 580,587-parameter StateWorldModel, 1-step plus 5-step rollout loss; test prediction at 1/5/10/20 steps; closed-loop CEM horizon 4, repeat 4, 256 samples, top-k 32, 4 iterations
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v2` (`pusht-recovery-pairs-v2`); run group `$DATASET_DIR/phase2_runs/attribution_v2_65b750f`
- Seeds: training seeds 0/1/2; paired evaluation on the same 30 scenario-disjoint test pairs per seed
- SLURM job ID/node: training `16213/16214`, `16220/16221`, `16224/16225`; closed-loop `16218/16219`, `16231/16232`, `16236/16237`; seed-0 evaluation refresh `16246/16247` on `3090node1`; all completed with exit code 0
- Log/output path: `$DATASET_DIR/logs/p2-v2-*.out`; aggregate `$DATASET_DIR/phase2_runs/attribution_v2_65b750f/three_seed_attribution_aggregate.json`
- Status: completed
- Key metrics/error: recovery top-1 is 0.811 for SFN versus 1.000 for SFR, paired delta +0.189 with crossed-bootstrap 95% CI [0.067, 0.333]. On R only, 20-step object-goal RMSE is 9.323 versus 3.252 px; on reposition/recontact starts it is 14.666 versus 4.532 px. Closed-loop maximum coverage is 0.469 versus 0.696, delta +0.227 [0.110, 0.361], and action cost is 31.831 versus 24.576, delta -7.255 [-13.472, -0.829]. Final coverage delta +0.081 [-0.106, 0.299] is uncertain and both conditions achieve only 1/90 successes. Relevant remote regression passes 12 tests; no retraining was required for the final aggregation refresh.
- Decision/next action: the mechanism-level attribution signal passes—improvements cannot be reduced to extra successful trajectories—but strict Gate B remains not passed. Keep Phase 2, diagnose goal retention on validation scenarios, lock the planner, then use fresh confirmatory pairs before Experiment B.

### 2026-09-15 10:08 - Phase 2 v2 stratified offline result and closed-loop jobs

- Phase/purpose: verify that the one-seed SFN/SFR difference occurs on R and its recovery prefix, then test whether the signal is usable in closed loop
- Git commit: stratified evaluation `9631a08`; completed fixture `a9aa096`; offline SLURM entry `a63c8e8`
- Command/config: branch-specific 1/5/10/20-step prediction; R windows whose starting phase is `reposition` or `recontact`; same 30 paired test scenarios. Closed-loop uses horizon 4, action repeat 4, 256 samples, top-k 32, 4 CEM iterations, action cost 0.01, no smoothness cost and staging weight 1.0
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v2`
- Seeds: training seed 0; deterministic per-scenario planning seeds
- SLURM job ID/node: stratified evaluation `16216/16217`, completed on `3090node1` in 17s/14s; closed-loop `16218/16219` submitted
- Log/output path: `$DATASET_DIR/phase2_runs/attribution_v2_65b750f/seed_0_sfn_vs_sfr_stratified.json`; closed-loop logs `$DATASET_DIR/logs/p2-v2-{sfn,sfr}-cl-<job>.out`
- Status: offline completed; closed-loop queued/running
- Key metrics/error: R-branch 20-step object-goal RMSE is 10.315 px for SFN versus 3.573 px for SFR. Starting only at recovery-prefix states, it is 16.287 versus 4.650 px; corresponding agent-object RMSE is 21.460 versus 2.136 px. Recovery top-1 is 0.80 versus 1.00 with paired scenario-bootstrap delta +0.20 [0.067, 0.367]. All related regression tests pass (15 tests); an initial fixture-only missing `success` field was corrected before evaluation.
- Decision/next action: one-seed offline evidence supports recovery-specific rather than success-count-only value. Complete the matched closed-loop pilot before deciding whether additional training seeds are warranted.

### 2026-09-15 09:53 - Phase 2 v2 one-seed SFN/SFR attribution training

- Phase/purpose: test whether off-nominal recovery data adds value beyond an equal number of successful on-manifold nominal continuations
- Git commit: training/data interface `27a8323`; dataset audit record `65b750f`
- Command/config: `D_SFN_balanced` versus `D_SFR_balanced`, seed 0, shared `D_SF` train-only normalization, 50 epochs, batch 128, identical 580,587-parameter StateWorldModel and 1-step + 5-step rollout loss
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v2`
- Seeds: training seed 0
- SLURM job ID/node: `16213` SFN and `16214` SFR completed on `3090node1` with exit code `0` in 4m27s and 4m04s
- Log/output path: `$DATASET_DIR/logs/p2-v2-sfn-s0-16213.out`, `$DATASET_DIR/logs/p2-v2-sfr-s0-16214.out`; run group `$DATASET_DIR/phase2_runs/attribution_v2_65b750f`
- Status: completed; branch/phase-stratified re-evaluation pending
- Key metrics/error: best validation loss was 0.01339 for SFN and 0.02206 for SFR; these losses are not cross-condition outcomes because R is a more complex target distribution. On the common 30-pair test, preliminary aggregate prediction/ranking gives recovery top-1 0.80 versus 1.00, paired delta +0.20 with scenario-bootstrap 95% CI [0.067, 0.367], margin delta +0.130 [0.052, 0.209], and selection-regret reduction +0.102 [0.033, 0.179]. Twenty-step object-goal RMSE is 4.882 versus 2.536 px. This is one training seed only.
- Decision/next action: add and run branch-specific plus recovery-prefix prediction reporting before interpreting the multi-step result; do not launch more seeds yet

### 2026-09-15 09:47 - Phase 1 v2 200-pair attribution dataset

- Phase/purpose: generate the full small-pilot dataset containing the success-matched nominal continuation `N` for the SFN versus SFR attribution comparison
- Git commit: generator `27a8323`; diagnostic updates `92aedca`; tracked figures `3f1d255`
- Command/config: 200 scenarios, 224-pixel RGB, 50-step S, equal 35-step N/F1/F2/R branches, 21-state windows, videos and full audit enabled
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v2` (`pusht-recovery-pairs-v2`)
- Seeds: data seed 20260914
- SLURM job ID/node: `16212`, completed on `4090node3` with exit code `0` in 3m23s
- Log/output path: `$DATASET_DIR/logs/p1-v2-data-16212.out`
- Status: completed
- Key metrics/error: dataset size 26 MB; Gate A passed with zero errors/warnings, zero counterfactual branch error, zero N-to-S snapshot error, zero action violations and zero alignment violations. S/N/R success rates are 1.0 and F1/F2 are 0.0. SFN and SFR each contain 400 success / 200 failure trajectories and identical 8,820/1,890/1,890 train/valid/test windows. Across 200 pairs, R versus N action RMSE is 0.104 command units, agent-object RMSE 11.04 px, object-goal RMSE 12.75 px, and recovery prefix averages 6.685 of 35 steps (19.1%, range 4–9).
- Decision/next action: full v2 attribution dataset is structurally valid and R is measurably off nominal, though the prefix remains short for retreat cases. Start exactly one matched training seed for SFN versus SFR with shared D_SF normalization; inspect offline failure-state evidence before closed-loop or more seeds.

### 2026-09-15 09:40 - Phase 1/2 recovery-specific attribution v2 implementation and smoke

- Phase/purpose: distinguish recovery-specific dynamics coverage from the simpler explanation that SFR contains more successful examples
- Git commit: implementation `27a8323`; visualization layout follow-up pending
- Command/config: add equal-horizon nominal continuation `N`, `D_SFN_balanced`, action-phase labels, R-versus-N distinctiveness audit and six-cell visual diagnostics
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_v2_smoke_27a8323`; existing v1 pilot remains immutable
- Seeds: data seed 20260914; no training seed started yet
- SLURM job ID/node: not applicable yet
- Log/output path: smoke `audit.json`; remote-generated figures copied to `report_assets/phase1-v2-six-cell-*.png`
- Status: completed
- Key metrics/error: remote Phase 1/2 regression passed 12 tests. Six-cell smoke Gate A passed with zero branch/snapshot/action/alignment violations; S/N/R success 1.0 and F1/F2 success 0.0. `D_SFN_balanced` and `D_SFR_balanced` both contain 12 success and 6 failure trajectories. Aligned R versus N has mean action RMSE 0.101 command units, agent-object RMSE 11.06 px, object-goal RMSE 11.90 px, and a mean 6.67-step recovery prefix (19.0% of the 35-step branch; range 4–9 steps).
- Decision/next action: R is measurably distinct from N but the retreat-low recovery prefix remains short. The v2 structure is sufficient for this feasibility attribution test; generate the 200-pair v2 pilot under SLURM, then run one-seed SFN/SFR before deciding whether to expand.

### 2026-09-14 22:05 - 原始框架与 recovery 扩展代码整理

- Phase/purpose: 建立 Phase 0–2 文件索引，明确原始 DINO-WM、对原文件的兼容性修改、本项目新增模块以及生成产物之间的边界
- Git commit: pending
- Command/config: 以原始框架提交 `5b12dea` 为基线审计当前变更；仅编辑文档，不执行本地项目代码
- Dataset path/version: 未读取或修改数据集；远端权威数据仍为 `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`
- Seeds: not applicable
- SLURM job ID/node: not applicable
- Log/output path: `IMPLEMENTATION_OVERVIEW.md`、`AGENTS.md`、`Progress.md`
- Status: completed
- Key metrics/error: 将原始视觉 DINO-WM 路线与当前 oracle-state Experiment A 明确分开；列出 11 个被修改的原框架文件、Phase 0/1/2 新增入口、测试、报告资产、远端产物和未完成项
- Decision/next action: 以后以 `IMPLEMENTATION_OVERVIEW.md` 作为代码地图；Gate B 仍未通过，下一步只在 validation scenarios 上修复 goal-retention，再使用 fresh confirmatory test

### 2026-09-14 21:36 - Phase 1/2 Chinese experiment report and figures

- Phase/purpose: consolidate the paired dataset, corrected fixed-budget configurations, metric definitions, three-seed results, validity limits and Gate B decision into a self-contained report
- Git commit: figure generator `cdf1ab6`; report commit pending in this entry
- Command/config: generated five static scientific figures from the authoritative Phase 1 dataset and corrected Phase 2 JSON reports; representative branch montage uses `scenario_000000`
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1` and `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/`
- Seeds: data seed 20260914; training seeds 0, 1 and 2
- SLURM job ID/node: not applicable; short remote CPU rendering run
- Log/output path: `Experiment_Report_Phase2.md`, `report_assets/*.png`, and `scripts/generate_phase2_report_assets.py`
- Status: completed
- Key metrics/error: report directly embeds the branch montage, dataset audit, optimization curves, main comparisons and coverage-retention diagnostic; it defines every prediction, ranking, closed-loop and uncertainty metric and distinguishes all dataset/training/planner configurations
- Decision/next action: use the report as the Phase 2 pilot record; keep Gate B closed and develop planner changes only on validation scenarios before a fresh confirmatory test

### 2026-09-14 21:07 - Phase 2 corrected three-seed aggregate and Gate B decision

- Phase/purpose: aggregate the fixed-budget comparison across both training initialization and matched test-scenario variation
- Git commit: checkpoints `cd99a24`; crossed-bootstrap aggregator and tests `a4fd07d`
- Command/config: seeds 0/1/2; 30 common scenarios per seed; 10,000 crossed bootstrap resamples that independently resample training seeds and scenario IDs
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; report at `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/three_seed_aggregate.json`
- Seeds: training seeds 0, 1 and 2; bootstrap seed 0
- SLURM job ID/node: source jobs `16191`-`16202`; aggregation was a short remote CPU run
- Log/output path: aggregate JSON above; remote regression output in terminal
- Status: completed
- Key metrics/error: recovery ranking averaged 0.800 for `SF` versus 0.989 for `SFR`, paired delta +0.189 with crossed-bootstrap 95% CI [0.056, 0.367]. Twenty-step object-goal RMSE averaged 5.255 versus 2.589 px, with reductions positive for all three seeds. Closed-loop success averaged 0.022 versus 0.056 (2/90 versus 5/90), delta +0.033 [-0.067, 0.156]; final coverage averaged 0.388 versus 0.495, delta +0.107 [-0.054, 0.254]. Peak coverage improved by +0.234 [0.135, 0.340] and action cost fell by 5.44 [-9.41, -1.76]. Aggregator plus Phase 2 regression passed 10 tests remotely.
- Decision/next action: Gate B is not passed because success and final-coverage intervals cross zero despite robust offline, peak-coverage and action-cost gains. Stay in Phase 2; diagnose loss of achieved coverage, develop planner changes only on validation scenarios, then lock them and evaluate on fresh confirmatory pairs.

### 2026-09-14 21:02 - Phase 2 corrected seed-2 closed-loop comparison

- Phase/purpose: complete the third matched closed-loop recovery comparison
- Git commit: checkpoints `cd99a24`; evaluation/comparison code available at `a4fd07d`
- Command/config: all 30 held-out post-perturbation snapshots; maximum 35 execution steps; calibrated short-horizon CEM with horizon 4, action repeat 4, 256 samples, top-k 32, 4 iterations, action cost 0.01, zero smoothness cost and staging weight 1.0
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; report at `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/seed_2_closed_loop_short_comparison.json`
- Seeds: training seed 2; deterministic per-scenario planner seeds; paired bootstrap seeds 0-2
- SLURM job ID/node: `16201` (46s) and `16202` (48s), both completed on 4090node1
- Log/output path: `$DATASET_DIR/logs/p2-sf-cr-cl2-16201.out`, `$DATASET_DIR/logs/p2-sfr-cr-cl2-16202.out`, and the comparison JSON above
- Status: completed
- Key metrics/error: success was 1/30 for `SF` and 0/30 for `SFR`, paired delta -0.033 with 95% CI [-0.100, 0.000]. Final coverage improved by +0.142 but its interval [-0.009, 0.295] crossed zero; action cost fell by 4.31 with interval [-8.55, 0.01]. Both jobs exited 0.
- Decision/next action: seed 2 does not pass the closed-loop half of Gate B; aggregate all three seeds with crossed resampling

### 2026-09-14 20:58 - Phase 2 corrected seed-2 offline comparison

- Phase/purpose: complete the third corrected fixed-budget offline comparison
- Git commit: checkpoints `cd99a24`; comparison code available at `1c5debb`
- Command/config: universal 30-scenario `S/F1/F2/R` test set; 10,000 paired bootstrap resamples
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; report at `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/seed_2_offline_comparison.json`
- Seeds: training seed 2; bootstrap seeds 0-2
- SLURM job ID/node: source training jobs `16199` and `16200`; comparison was a short remote CPU run
- Log/output path: comparison JSON above
- Status: completed
- Key metrics/error: recovery top-1 ranking improved from 0.900 to 1.000, paired delta +0.100 with 95% CI [0.000, 0.200]; the margin delta was +0.060 [-0.026, 0.155]. Object-goal RMSE improved from 0.493 to 0.318 px at 1 step, 1.507 to 0.966 px at 5 steps, 3.332 to 1.577 px at 10 steps and 5.468 to 2.860 px at 20 steps.
- Decision/next action: core task-state prediction again improves, while ranking uncertainty is weaker than seeds 0/1; complete the matched closed-loop evaluation

### 2026-09-14 20:36 - Phase 2 corrected fixed-budget seed-2 training submissions

- Phase/purpose: complete the predeclared three-seed pilot and resolve the seed-sensitive closed-loop result
- Git commit: `cd99a24`
- Command/config: identical to corrected seeds 0 and 1: shared `D_SF` train-only normalization, one-unit scale floors, 1-step plus weighted 5-step rollout loss, 50 epochs, branch balancing and a 580,587-parameter model
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; outputs under `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/`
- Seeds: 2 for both `D_SF_balanced` and `D_SFR_balanced`
- SLURM job ID/node: `16199` (`D_SF_balanced`) and `16200` (`D_SFR_balanced`), both on 4090node1
- Log/output path: `$DATASET_DIR/logs/p2-sf-cr2-16199.out` and `$DATASET_DIR/logs/p2-sfr-cr2-16200.out`
- Status: completed; both exited 0 (`16199` in 6m29s, `16200` in 6m16s)
- Key metrics/error: both runs produced best/latest checkpoints and complete automatic offline evaluation reports
- Decision/next action: paired offline and closed-loop comparisons completed; aggregate across all three training seeds before deciding Gate B

### 2026-09-14 20:35 - Phase 2 corrected seed-1 closed-loop comparison

- Phase/purpose: test whether seed 1's replicated offline advantage transfers to real simulator recovery
- Git commit: checkpoints `cd99a24`; evaluation/comparison code available at `46fb2c1`
- Command/config: all 30 held-out post-perturbation snapshots; maximum 35 execution steps; calibrated short-horizon CEM with horizon 4, action repeat 4, 256 samples, top-k 32, 4 iterations, action cost 0.01, zero smoothness cost and staging weight 1.0
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; report at `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/seed_1_closed_loop_short_comparison.json`
- Seeds: training seed 1; deterministic per-scenario planner seeds; paired bootstrap seeds 0-2
- SLURM job ID/node: `16197` and `16198`, both completed on 4090node1 in 45s
- Log/output path: `$DATASET_DIR/logs/p2-sf-cr-cl1-16197.out`, `$DATASET_DIR/logs/p2-sfr-cr-cl1-16198.out`, and the comparison JSON above
- Status: completed
- Key metrics/error: both variants succeeded on 1/30 scenarios; paired success delta 0.000 with 95% CI [-0.100, 0.100]. Mean final coverage was 0.428 for `SF` and 0.418 for `SFR`, delta -0.011 [-0.176, 0.150]. `SFR` had higher mean maximum coverage (0.717 versus 0.544) and lower action cost (23.04 versus 29.88), paired action-cost delta -6.84 [-12.38, -1.54]. A Shapely intersection warning occurred, but both jobs exited 0 and wrote complete reports.
- Decision/next action: seed 1 does not pass the closed-loop half of Gate B; finish seed 2 and aggregate across training seeds before modifying the planner or model

### 2026-09-14 20:32 - Phase 2 corrected seed-1 offline comparison

- Phase/purpose: replicate the corrected fixed-budget offline recovery-data comparison under a new training initialization
- Git commit: checkpoints `cd99a24`; comparison code available at `46fb2c1`
- Command/config: universal 30-scenario `S/F1/F2/R` test set; 10,000 paired bootstrap resamples
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; report at `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/seed_1_offline_comparison.json`
- Seeds: training seed 1; bootstrap seeds 0-2
- SLURM job ID/node: source training jobs `16195` and `16196`; comparison was a short remote CPU run
- Log/output path: comparison JSON above
- Status: completed
- Key metrics/error: recovery top-1 ranking improved from 0.833 to 1.000; paired delta +0.167 with 95% CI [0.033, 0.300]; margin delta +0.248 [0.178, 0.321]; selection-regret reduction +0.082 [0.019, 0.153]. Object-goal RMSE improved from 0.608 to 0.307 px at 1 step, 1.626 to 0.617 px at 5 steps, 3.057 to 1.173 px at 10 steps and 4.871 to 2.123 px at 20 steps.
- Decision/next action: offline ranking and long-rollout task-state prediction gains replicate across seeds 0 and 1; evaluate closed-loop recovery before interpreting the result

### 2026-09-14 20:18 - Phase 2 corrected fixed-budget seed-1 training submissions

- Phase/purpose: test whether the positive corrected seed-0 recovery signal replicates across training initialization
- Git commit: `cd99a24`
- Command/config: same shared `D_SF` train-only normalization, one-unit scale floors, 1-step plus weighted 5-step rollout loss, 50 epochs, branch balancing and 580,587-parameter model as seed 0
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; outputs under `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/`
- Seeds: 1 for both `D_SF_balanced` and `D_SFR_balanced`
- SLURM job ID/node: `16195` (`D_SF_balanced`) and `16196` (`D_SFR_balanced`), both on 4090node1
- Log/output path: `$DATASET_DIR/logs/p2-sf-cr1-16195.out` and `$DATASET_DIR/logs/p2-sfr-cr1-16196.out`
- Status: completed; both exited 0 (`16195` in 6m21s, `16196` in 6m09s)
- Key metrics/error: both runs produced best/latest checkpoints and complete automatic offline evaluation reports
- Decision/next action: offline comparison replicated the recovery-ranking gain; matched closed-loop evaluation completed before starting seed 2

### 2026-09-14 20:15 - Phase 2 corrected seed-0 closed-loop submissions

- Phase/purpose: complete Gate B's closed-loop half on the corrected fixed-budget checkpoints
- Git commit: checkpoints `cd99a24`; evaluation code available at `b89b147`
- Command/config: all 30 held-out post-perturbation snapshots; maximum 35 execution steps; calibrated short-horizon CEM with horizon 4, action repeat 4, 256 samples, top-k 32, 4 iterations, action cost 0.01, zero smoothness cost and staging weight 1.0
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; outputs under `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/`
- Seeds: training seed 0; deterministic per-scenario planner seeds
- SLURM job ID/node: `16193` (`D_SF_balanced`) and `16194` (`D_SFR_balanced`), both on 4090node1
- Log/output path: `$DATASET_DIR/logs/p2-sf-cr-cl0-16193.out`, `$DATASET_DIR/logs/p2-sfr-cr-cl0-16194.out`, and `closed_loop_short_shaped.json` in each checkpoint directory
- Status: completed; both exited 0 (`16193` in 52s, `16194` in 45s)
- Key metrics/error: success improved from 0/30 to 4/30 (paired rate delta +0.133, 95% CI [0.033, 0.267]); mean final coverage improved from 0.395 to 0.585 (paired delta +0.190 [0.059, 0.319]); mean action cost fell from 30.12 to 24.95 (paired delta -5.17 [-9.78, -0.53]); recovery-rich mean maximum coverage was 0.739
- Decision/next action: corrected seed 0 provisionally passes Gate B because both ranking and closed-loop recovery improved with scenario-paired intervals excluding zero; run training seeds 1 and 2 before treating this as a robust model-level conclusion

### 2026-09-14 20:13 - Phase 2 corrected seed-0 offline comparison

- Phase/purpose: test the fixed-budget recovery-data effect after removing the normalization confound and adding rollout supervision
- Git commit: checkpoints `cd99a24`; comparison code available at `b89b147`
- Command/config: universal 30-scenario `S/F1/F2/R` test set; 10,000 paired bootstrap resamples
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; report at `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/seed_0_offline_comparison.json`
- Seeds: training seed 0; bootstrap seeds 0-2
- SLURM job ID/node: source training jobs `16191` and `16192`; comparison was a short remote CPU run
- Log/output path: comparison JSON above
- Status: completed
- Key metrics/error: recovery top-1 ranking improved from 0.667 to 0.967; paired delta +0.300 with 95% CI [0.100, 0.500]; margin delta +0.122 [0.027, 0.222]; selection-regret reduction +0.156 [0.069, 0.250]. Object-goal RMSE changed from 0.692 to 0.313 px at 1 step, 1.687 to 0.640 px at 5 steps, 3.212 to 1.370 px at 10 steps and 5.426 to 2.784 px at 20 steps.
- Decision/next action: corrected seed-0 offline evidence supports recovery-rich data; Gate B still requires the matched closed-loop result and additional seeds before a robust claim

### 2026-09-14 19:57 - Phase 2 corrected fixed-budget seed-0 training

- Phase/purpose: rerun the primary comparison without preprocessing confounds and with explicit autoregressive rollout supervision
- Git commit: `cd99a24`
- Command/config: `D_SF_balanced` and `D_SFR_balanced`; shared normalization from the common `D_SF` train subset; one-unit state scale floor; 1-step loss plus weighted 5-step rollout loss; otherwise identical 50-epoch, 580,587-parameter configuration
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; outputs under `$DATASET_DIR/phase2_runs/fixed_common_rollout_cd99a24/`
- Seeds: 0 for both conditions
- SLURM job ID/node: `16191` (`D_SF_balanced`, 4090node1) and `16192` (`D_SFR_balanced`, 4090node1)
- Log/output path: `$DATASET_DIR/logs/p2-sf-cr0-16191.out` and `$DATASET_DIR/logs/p2-sfr-cr0-16192.out`
- Status: completed; both jobs exited 0 (`16191` in 6m06s, `16192` in 6m26s)
- Key metrics/error: prerequisite remote regression passed 8 tests; both 50-epoch runs produced best/latest checkpoints and automatic evaluation reports. Best validation loss was 0.01009 for `D_SF_balanced` at epoch 45 and 0.02171 for `D_SFR_balanced` at epoch 47; losses are not treated as the cross-condition outcome because the branch targets differ.
- Decision/next action: paired offline comparison completed with a positive signal; run the same calibrated short-horizon planner on both corrected checkpoints before adding training seeds

### 2026-09-14 19:48-19:54 - Phase 2 planner calibration series

- Phase/purpose: diagnose and reduce model exploitation in closed-loop recovery planning
- Git commit: checkpoints `a98096d`; planner code evolved through `4f764ba` and `5dfb388`
- Command/config: five held-out scenarios per run; job 16186 used horizon 12/action-repeat 3; job 16187 used horizon 20/action-repeat 4; job 16188 added staging shaping to the long horizon; jobs 16189/16190 used horizon 4 as one constant action block with staging weight 1 for recovery-rich/control respectively
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; calibration JSON files under the two `$DATASET_DIR/phase2_runs/pilot_a98096d/.../seed_0/` directories
- Seeds: training seed 0; deterministic planner seeds for the first five test scenarios
- SLURM job ID/node: 16186, 16187, 16188, 16189 and 16190; all completed on 4090node1
- Log/output path: `$DATASET_DIR/logs/p2-*-1618*.out` plus `closed_loop_*_calib5.json`
- Status: completed
- Key metrics/error: low-frequency search reduced recovery-model action cost from about 23 to 4.80 but had 0/5 success; horizon 20 reached mean final coverage 0.488 but 0/5; long-horizon staging shaping regressed to 0.369; short-horizon feedback reached 1/5 and final coverage 0.485 for recovery-rich versus 0/5 and 0.319 for control
- Decision/next action: reject long-horizon shaping as the formal planner; retain short-horizon feedback as the current candidate and add rollout loss to training before retesting

### 2026-09-14 19:43 - Phase 2 seed-0 full closed-loop comparison

- Phase/purpose: test whether the strong offline ranking signal transfers to unconstrained receding-horizon CEM
- Git commit: checkpoints `a98096d`; evaluation `6be0450`
- Command/config: all 30 test snapshots; 35 steps; CEM horizon 12, 256 samples, top-k 32, 4 iterations
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; `$DATASET_DIR/phase2_runs/pilot_a98096d/`
- Seeds: training seed 0 and deterministic per-scenario planner seeds
- SLURM job ID/node: 16184 and 16185, both completed serially on 4090node1
- Log/output path: `$DATASET_DIR/logs/p2-sf-cl0-16184.out`, `$DATASET_DIR/logs/p2-sfr-cl0-16185.out`, and `$DATASET_DIR/phase2_runs/pilot_a98096d/seed_0_closed_loop_comparison.json`
- Status: completed
- Key metrics/error: success 0/30 versus 1/30; final coverage 0.360 versus 0.402; paired coverage delta +0.0417 with 95% CI [-0.0489, 0.1342]; traces showed predicted near-zero goal cost despite unchanged real coverage
- Decision/next action: Gate B not passed; constrain planner search and train with open-loop rollout loss before any multi-seed claim

### 2026-09-14 19:40 - Phase 2 common-normalization control

- Phase/purpose: remove preprocessing as a confound from the primary fixed-budget comparison
- Git commit: `1471ed7`
- Command/config: both balanced variants will use statistics from their common `D_SF = S+F1` training subset; every oracle-state standard deviation has a one-unit physical floor so constant angle coordinates are not divided by numerical noise
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`
- Seeds: planned seed 0 rerun first
- SLURM job ID/node: none; submission was rejected by `AssocMaxSubmitJobLimit` while jobs 16184/16185 occupied the account's two submission slots
- Log/output path: not applicable; remote Phase 2 regression output in terminal
- Status: planned
- Key metrics/error: updated normalization implementation passed all 5 Phase 2 tests remotely; no training job was created by the rejected submission
- Decision/next action: submit the two common-normalization seed-0 runs after the closed-loop jobs release the account slots; treat the earlier variant-specific-normalization result as pilot evidence only

### 2026-09-14 19:38 - Phase 2 seed-0 offline paired comparison

- Phase/purpose: compare fixed-budget models on identical held-out pairs with paired bootstrap uncertainty
- Git commit: model checkpoints `a98096d`; comparison code `6be0450`
- Command/config: universal 30-scenario `S/F1/F2/R` test set; 10,000 paired bootstrap resamples
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; `$DATASET_DIR/phase2_runs/pilot_a98096d/`
- Seeds: training seed 0; bootstrap seeds 0-2
- SLURM job ID/node: training jobs 16182/16183; comparison was a short remote CPU run
- Log/output path: `$DATASET_DIR/phase2_runs/pilot_a98096d/seed_0_offline_comparison.json`
- Status: completed
- Key metrics/error: recovery top-1 ranking improved from 0.633 to 1.000; paired delta +0.367, pair-bootstrap 95% CI [0.200, 0.533]; recovery-margin delta +0.243 [0.185, 0.306]; selection-regret reduction +0.174 [0.092, 0.259]; 20-step object-goal RMSE decreased from 6.20 to 2.09 px
- Decision/next action: signal is promising but does not pass Gate B yet because this is one seed, uses variant-specific normalization, and lacks closed-loop comparison

### 2026-09-14 19:39 - Phase 2 seed-0 closed-loop submissions

- Phase/purpose: evaluate whether the offline ranking gain produces real receding-horizon recovery
- Git commit: checkpoints `a98096d`; evaluation code `6be0450`
- Command/config: all 30 held-out post-perturbation snapshots; maximum 35 steps; bounded CEM horizon 12, 256 samples, top-k 32, 4 iterations
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`
- Seeds: training seed 0; deterministic per-scenario planner seeds
- SLURM job ID/node: `16184` (`D_SF_balanced`, running on 4090node1) and `16185` (`D_SFR_balanced`, pending under `AssocMaxJobsLimit`)
- Log/output path: `$DATASET_DIR/logs/p2-sf-cl0-16184.out` and `$DATASET_DIR/logs/p2-sfr-cl0-16185.out`; result files named `closed_loop.json` in each run directory
- Status: running
- Key metrics/error: no runtime errors at submission/initialization; full evaluation produces output only when all scenarios complete
- Decision/next action: monitor both jobs, then make a paired closed-loop report

### 2026-09-14 19:32 - Phase 2 fixed-budget seed-0 pilot submission

- Phase/purpose: first full-data equal-capacity comparison of neutral-failure versus recovery-rich dynamics
- Git commit: `a98096d`
- Command/config: 50 epochs, batch 128, 580,587-parameter state model, branch-balanced sampling; `D_SF_balanced` and `D_SFR_balanced`
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; outputs under `$DATASET_DIR/phase2_runs/pilot_a98096d/`
- Seeds: 0 for both conditions
- SLURM job ID/node: `16182` and `16183`, both ran serially on 4090node1 because of `AssocMaxJobsLimit`
- Log/output path: `$DATASET_DIR/logs/p2-sf-b0-16182.out` and `$DATASET_DIR/logs/p2-sfr-b0-16183.out`
- Status: completed
- Key metrics/error: storage had 9.5 TB available; best valid losses were 0.00179 (`D_SF_balanced`) and 0.00846 (`D_SFR_balanced`); both checkpoints and unified-test reports completed successfully
- Decision/next action: paired comparison showed a promising recovery signal; rerun the primary comparison with shared train-only normalization before adding seeds

### 2026-09-14 19:30 - Phase 2 closed-loop interface smoke

- Phase/purpose: exercise checkpoint-to-CEM-to-real-simulator recovery execution on one held-out pair
- Git commit: `ca149d8`
- Command/config: tiny smoke checkpoint; CPU; one test scenario; 35 execution steps; CEM horizon 4, 32 samples, top-k 4, 2 iterations
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`
- Seeds: training seed 0; planner seed 0
- SLURM job ID/node: not applicable; remote login-node short smoke
- Log/output path: `$DATASET_DIR/phase2_runs/smoke_ca149d8/D_SFR_balanced/seed_0/closed_loop_smoke.json`
- Status: completed
- Key metrics/error: end-to-end state normalization, model rollout, bounded CEM and simulator execution succeeded; the deliberately under-trained 64-window model did not recover (0/1, final coverage 0.280)
- Decision/next action: treat this only as an interface check, not an experiment result; train the full fixed-budget models before evaluating recovery

### 2026-09-14 19:28 - Phase 2 real-data training and evaluation smoke

- Phase/purpose: validate the full paired loader, train-only normalization, checkpoint and held-out prediction/ranking pipeline
- Git commit: `ca149d8`
- Command/config: `D_SFR_balanced`; CPU; 64 stratified train and 64 valid windows; 81,403-parameter reduced model; 20 epochs with 2 steps per epoch
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`
- Seeds: 0
- SLURM job ID/node: not applicable; remote login-node short smoke
- Log/output path: `$DATASET_DIR/phase2_runs/smoke_ca149d8/D_SFR_balanced/seed_0/`
- Status: completed
- Key metrics/error: training statistics used 17,220 unique train frames; validation loss decreased from 0.433 to 0.358; prediction and ranking JSON was produced; ranking was 1/2 pairs and is not interpretable at smoke scale
- Decision/next action: submit equal-capacity full-data `D_SF_balanced` and `D_SFR_balanced` pilot models under SLURM

### 2026-09-14 19:25 - Phase 2 remote regression

- Phase/purpose: validate Phase 0-2 runtime, paired loader, CPU-safe causal predictor, rollout and physical-unit metrics
- Git commit: `ca149d8`
- Command/config: `python -m pytest -q tests/test_pusht_phase0.py tests/test_pusht_phase1.py tests/test_pusht_phase2.py`
- Dataset path/version: synthetic test fixtures plus the checked-in remote Phase 1 scenario
- Seeds: existing test seeds plus Phase 2 model seed 3
- SLURM job ID/node: not applicable; remote login-node short test
- Log/output path: terminal output
- Status: completed
- Key metrics/error: 14 tests passed in 9.49 seconds; four dependency deprecation warnings only
- Decision/next action: real-data smoke validation, then fixed-budget pilot training

### 2026-09-14 15:04 - Phase 1 final regression with remote fixture

- Phase/purpose: final Phase 0+1 regression after adding one real generated scenario to Git
- Git commit: `a87a976`
- Command/config: `python -m pytest -q tests/test_pusht_phase0.py tests/test_pusht_phase1.py`
- Dataset path/version: Git fixture from `$DATASET_DIR/pusht_recovery_phase1_pilot_v1/scenarios/scenario_000000`
- Seeds: Phase 0 seed 7; Phase 1 test seeds 11 and 1234; generated fixture seed 20260914
- SLURM job ID/node: not applicable; remote login-node short test
- Log/output path: terminal output
- Status: completed
- Key metrics/error: 10 tests passed in 11.38 seconds; only four dependency deprecation warnings
- Decision/next action: Phase 1 is complete; begin Phase 2 state-world-model baselines using the audited split/window manifests

### 2026-09-14 14:57 - Phase 1 full 200-scenario pilot

- Phase/purpose: generate and audit the complete Phase 1 paired simulation pilot
- Git commit: `a976e2c`
- Command/config: SLURM CPU job; 200 scenarios, 224-pixel RGB, 50 nominal steps, 35 branch steps, 21-frame windows, 70/15/15 scenario split
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`
- Seeds: base seed 20260914; deterministic per-scenario seeds
- SLURM job ID/node: `16174`, `4090node3`, 4 CPUs, no GPU, 2-hour limit
- Log/output path: `$DATASET_DIR/logs/phase1_pilot_16174.out`
- Status: completed in 3 minutes 39 seconds with exit code 0
- Key metrics/error: 200 scenarios and 800 videos (21 MB); train/valid/test 140/30/30; S/F1/F2/R success 1.0/0.0/0.0/1.0; recovery-minus-F1 coverage +0.635, bootstrap 95% CI [0.629, 0.641]; branch/action/alignment errors all 0
- Decision/next action: Gate A passed; retain 200 scenarios because it exceeds the conservative 180-scenario stratification floor and proceed to Phase 2 Experiment A

### 2026-09-14 14:56 - Phase 1 pilot submission attempt

- Phase/purpose: submit the full Phase 1 pilot
- Git commit: `a976e2c`
- Command/config: initial `sbatch` request included `--mem=8G`
- Dataset path/version: intended `$DATASET_DIR/pusht_recovery_phase1_pilot_v1`; no data generated
- Seeds: not started
- SLURM job ID/node: none
- Log/output path: submission stderr
- Status: failed
- Key metrics/error: Slurm rejected the request because cluster nodes advertise only `1M` memory in their controller metadata
- Decision/next action: omitted the invalid explicit memory request and submitted CPU job 16174; no cluster configuration was changed

### 2026-09-14 14:37 - Phase 1 micro-pilot validation attempt 1

- Phase/purpose: Phase 1 oracle and six-scenario micro-pilot validation
- Git commit: `6a3de46`
- Command/config: `python -m pytest -q tests/test_pusht_phase1.py`
- Dataset path/version: temporary pytest directory; no authoritative dataset write
- Seeds: 11 and 1234
- SLURM job ID/node: not applicable; remote login-node short test
- Log/output path: terminal output
- Status: failed
- Key metrics/error: 2 tests failed because the initial 105-pixel agent offset overlapped the 120-pixel T stem plus 15-pixel agent radius, imparting unintended torque; nominal final coverage was 0
- Decision/next action: initialize at the geometry-derived 139-pixel contact-safe offset and rerun the micro-pilot

### 2026-09-14 14:42 - Phase 1 micro-pilot validation attempt 2

- Phase/purpose: Phase 1 oracle and six-scenario micro-pilot validation after contact-safe initialization
- Git commit: `220fa63`
- Command/config: `python -m pytest -q tests/test_pusht_phase1.py`
- Dataset path/version: temporary pytest directory; no authoritative dataset write
- Seeds: 11 and 1234
- SLURM job ID/node: not applicable; remote login-node short test
- Log/output path: terminal output and a 20-step diagnostic trace
- Status: failed
- Key metrics/error: fixed vertical nominal test passed at 99.2% final coverage; angled scenario failed because a numerical geometry-ray intersection selected the wrong staging surface near the goal
- Decision/next action: use the controlled translation domain's known 120-pixel rear support distance instead of a numerically unstable ray intersection

### 2026-09-14 14:44 - Phase 1 twelve-scenario calibration pilot

- Phase/purpose: assess recovery signal before submitting the 200-scenario pilot
- Git commit: `3c64a43`
- Command/config: `python generate_pusht_phase1.py --num-scenarios 12 --render-size 96 --output-dir .../pusht_recovery_phase1_micro_3c64a43`
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_micro_3c64a43`
- Seeds: base seed 20260914; 12 deterministic scenario seeds
- SLURM job ID/node: not applicable; 31-second remote calibration run
- Log/output path: dataset `audit.json` and terminal output
- Status: completed
- Key metrics/error: Gate A structural audit passed; S/F1/F2/R success rates were 1.0/0.0/0.0/0.5; exact branch error 0; recovery-minus-F1 coverage +0.249 with bootstrap 95% CI [0.019, 0.470]
- Decision/next action: do not launch the full pilot yet; agent perturbations recovered but object perturbations exposed an inaccurate constant support distance, so replace it with robust Pymunk shape queries and recalibrate

### 2026-09-14 14:47 - Phase 1 twelve-scenario calibration pilot after shape-query fix

- Phase/purpose: reassess recovery after replacing numerical geometry unions with Pymunk shape queries
- Git commit: `905105a`
- Command/config: same 12-scenario/96-pixel calibration at `.../pusht_recovery_phase1_micro_905105a`
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_micro_905105a`
- Seeds: base seed 20260914; 12 deterministic scenario seeds
- SLURM job ID/node: not applicable; 28-second remote calibration run
- Log/output path: dataset `audit.json` and per-branch trajectories
- Status: completed
- Key metrics/error: structural Gate A passed and branch error remained 0, but R success remained 0.5; trajectory inspection showed direct restaging crossed the object and rotated it by several radians
- Decision/next action: add clearance-radius and orbit-waypoint navigation before restaging; full pilot remains blocked until both perturbation types recover reliably

### 2026-09-14 14:50 - Phase 1 twelve-scenario calibration with safe restaging

- Phase/purpose: validate clearance-radius/orbit navigation on both perturbation types
- Git commit: `58e4567`
- Command/config: 12 scenarios, 96-pixel RGB, 50 nominal steps and 35 branch steps
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_micro_58e4567`
- Seeds: base seed 20260914; 12 deterministic scenario seeds
- SLURM job ID/node: not applicable; 28-second remote calibration run
- Log/output path: dataset `audit.json` and inspected `R.npz` trajectories
- Status: completed
- Key metrics/error: S/F1/F2/R success 1.0/0.0/0.0/0.5; recovery-minus-F1 coverage +0.307, bootstrap 95% CI [0.099, 0.505]; all object-lateral cases accumulated contact torque and failed
- Decision/next action: constrain the feasibility pilot to agent-lateral and agent-retreat failures; reserve object displacement and rotation for OOD after validating a rotation-capable oracle

### 2026-09-14 14:53 - Phase 1 constrained-domain calibration

- Phase/purpose: validate the revised recoverable perturbation domain before the batch pilot
- Git commit: `011d772`
- Command/config: 12 scenarios, 96-pixel RGB, balanced agent-lateral/agent-retreat perturbations
- Dataset path/version: `$DATASET_DIR/pusht_recovery_phase1_micro_011d772`
- Seeds: base seed 20260914; 12 deterministic scenario seeds
- SLURM job ID/node: not applicable; 29-second remote calibration run
- Log/output path: dataset `audit.json`
- Status: completed
- Key metrics/error: S/F1/F2/R success 1.0/0.0/0.0/1.0; recovery-minus-F1 final coverage +0.628, bootstrap 95% CI [0.601, 0.654]; branch error/action violations/alignment violations all 0
- Decision/next action: calibration gate passed; add conservative stratification-based sample-size reporting, rerun regression tests, then submit the 200-scenario SLURM pilot

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

### 2026-09-15

- Locked planner P3 using validation-only pooled absolute performance, generated 60 fresh balanced all-test scenarios, and confirmed both recovery ranking and closed-loop success across three training seeds; Experiment A feasibility Gate B now passes with a documented retention-loss limitation.
- Completed the 200-pair schema-v2 dataset with success-matched nominal continuation `N`, action phases, R/N audits and six-cell visual diagnostics.
- Completed three matched SFN/SFR training seeds and stratified evaluation on R and its reposition/recontact prefix.
- Added stratified multi-seed aggregation and recorded the recovery-specific attribution result.
- Added `Experiment_Report_Recovery_Attribution_v2.md`; strict Gate B remains not passed because final closed-loop success did not improve.

### 2026-09-14

- Started Phase 2 with a leakage-safe paired-state loader, a state/action-token
  `StateWorldModel` over the existing causal ViT predictor, multi-horizon and
  counterfactual ranking evaluation, and bounded state-space CEM recovery
  planning. Remote regression and smoke validation are pending.
- Completed the 200-scenario Phase 1 pilot and all structural/statistical audits under SLURM job 16174.
- Added one remotely generated pilot scenario plus the full audit/manifest as a small Git-tracked regression fixture; kept the complete dataset remote.
- Implemented and remotely validated the Phase 1 paired PushT generator, dataset variants, post-split window indices, structural audit, bootstrap intervals and pilot sample-size estimator.
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
