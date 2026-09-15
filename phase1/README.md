# PushT Phase 1 paired dataset

Each scenario is assigned to `train`, `valid` or `test` before any windows are
created. All `S/N/F1/F2/R` branches from one `pair_id` remain in that split.

The pilot is intentionally a controlled goal-aligned translation domain. It
uses balanced agent-lateral and agent-retreat perturbations at low, medium and
high severity. Object displacement and rotation perturbations are reserved for
OOD evaluation until a rotation-capable oracle has been validated.

Each scenario directory contains:

- `S/N/F1/F2/R.npz`: actions plus aligned proprio, 11-D oracle state, legacy
  state, complete numeric simulator state, coverage, success and contacts;
- `S/N/F1/F2/R.mp4`: the corresponding `T+1` RGB observations;
- `branch_snapshots.pkl`: exact pre/post-perturbation versioned simulator
  snapshots, including RNG state;
- `metadata.json`: IDs, split, perturbation, branch budgets and outcome labels.

`N` is an equal-horizon successful nominal continuation from the exact
pre-perturbation snapshot. It matches `R` in trajectory/success budget while
remaining on the nominal manifold, so `D_SFN_balanced` versus
`D_SFR_balanced` tests whether recovery states add value beyond simply adding
more successful examples. Action-level `phase` labels are audit-only metadata
and are not world-model inputs.

The root `manifest.json` records the generating commit/config and the six
dataset variants. `indices/*.jsonl` contains post-split windows, and
`audit.json` contains structural checks, paired effects, bootstrap intervals
R-versus-N distinctiveness metrics and pilot-based sample-size estimates.
