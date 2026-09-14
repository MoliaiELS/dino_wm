# Remote Phase 1 pilot fixture

This is one complete scenario copied from the remotely generated 200-scenario
pilot at commit `a976e2c`, together with the full pilot's `manifest.json` and
`audit.json`. It is intentionally small enough for Git and exists for schema,
loader and temporal-alignment regression tests.

The fixture is not a standalone replacement for the full dataset: its manifest
still describes all 200 scenarios and their window indices. The authoritative
dataset remains at:

```text
/mnt/slurmfs-4090node3/user_data/yguo704/dino_wm_dataset/pusht_recovery_phase1_pilot_v1
```
