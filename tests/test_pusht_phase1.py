import json
import os
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from env.pusht.pusht_env import PushTEnv
from phase1.pusht_dataset import (
    BRANCHES,
    VARIANTS,
    Phase1Config,
    PushTPhase1Generator,
    rollout_from_snapshot,
)
from phase1.pusht_oracle import GeometricPushTOracle


def test_geometric_oracle_solves_controlled_nominal_translation():
    env = PushTEnv(with_velocity=True, render_size=64)
    try:
        env.seed(11)
        env.reset_to_state = np.array([256, 459, 256, 320, 0, 0, 0])
        env.reset()
        env.set_task_goal(np.array([256, 240, 0], dtype=np.float64))
        initial_snapshot = env.get_sim_state()
        record = rollout_from_snapshot(
            env,
            initial_snapshot,
            horizon=50,
            controller=GeometricPushTOracle(),
        )
        assert record.success[-1]
        assert record.coverage.max() >= env.success_threshold
        assert np.all(record.actions >= env.action_space.low)
        assert np.all(record.actions <= env.action_space.high)
    finally:
        env.close()


def test_micro_dataset_has_exact_pairs_splits_and_alignment(tmp_path):
    config = replace(
        Phase1Config(),
        num_scenarios=6,
        seed=1234,
        render_size=64,
        nominal_horizon=50,
        branch_horizon=24,
        window_length=8,
        bootstrap_samples=200,
    )
    manifest, audit = PushTPhase1Generator(tmp_path / "pilot", config).generate()

    assert audit["gate_a_pass"], audit["errors"]
    assert audit["scenario_count"] == 6
    assert audit["branch_state_max_abs_error"] == 0
    assert audit["action_bound_violations"] == 0
    assert audit["temporal_alignment_violations"] == 0
    assert audit["perturbation_type_counts"] == {
        "agent_lateral": 3,
        "agent_retreat": 3,
    }
    assert audit["severity_counts"] == {"low": 2, "medium": 2, "high": 2}
    assert set(manifest["scenario_splits"].values()) == {"train", "valid", "test"}

    for scenario_id, split in manifest["scenario_splits"].items():
        scenario_dir = tmp_path / "pilot" / "scenarios" / scenario_id
        with open(scenario_dir / "metadata.json", encoding="utf-8") as file:
            metadata = json.load(file)
        assert metadata["split"] == split
        assert set(metadata["branches"]) == set(BRANCHES)

        initial_states = []
        for branch in ("F1", "F2", "R"):
            with np.load(scenario_dir / f"{branch}.npz") as record:
                initial_states.append(record["sim_state"][0])
                assert len(record["oracle_state"]) == len(record["actions"]) + 1
                assert len(record["coverage"]) == len(record["actions"]) + 1
        np.testing.assert_array_equal(initial_states[0], initial_states[1])
        np.testing.assert_array_equal(initial_states[0], initial_states[2])

    for variant, expected_branches in VARIANTS.items():
        for split, index_metadata in manifest["window_indices"][variant].items():
            index_path = tmp_path / "pilot" / index_metadata["path"]
            with open(index_path, encoding="utf-8") as file:
                windows = [json.loads(line) for line in file]
            assert len(windows) == index_metadata["count"]
            assert all(window["split"] == split for window in windows)
            assert all(window["branch"] in expected_branches for window in windows)
