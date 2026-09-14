# PushT Phase 1 paired dataset

Each scenario is assigned to `train`, `valid` or `test` before any windows are
created. All `S/F1/F2/R` branches from one `pair_id` remain in that split.

The pilot is intentionally a controlled goal-aligned translation domain. It
uses balanced agent-lateral and agent-retreat perturbations at low, medium and
high severity. Object displacement and rotation perturbations are reserved for
OOD evaluation until a rotation-capable oracle has been validated.

Each scenario directory contains:

- `S/F1/F2/R.npz`: actions plus aligned proprio, 11-D oracle state, legacy
  state, complete numeric simulator state, coverage, success and contacts;
- `S/F1/F2/R.mp4`: the corresponding `T+1` RGB observations;
- `branch_snapshots.pkl`: exact pre/post-perturbation versioned simulator
  snapshots, including RNG state;
- `metadata.json`: IDs, split, perturbation, branch budgets and outcome labels.

The root `manifest.json` records the generating commit/config and the five
dataset variants. `indices/*.jsonl` contains post-split windows, and
`audit.json` contains structural checks, paired effects, bootstrap intervals
and pilot-based sample-size estimates.
