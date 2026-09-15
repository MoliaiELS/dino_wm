import json
import os
from dataclasses import replace
from pathlib import Path

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


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pusht_phase1_pilot_v1"


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
        branch_horizon=35,
        window_length=8,
        bootstrap_samples=200,
    )
    manifest, audit = PushTPhase1Generator(tmp_path / "pilot", config).generate()

    assert audit["gate_a_pass"], audit["errors"]
    assert audit["scenario_count"] == 6
    assert audit["branch_state_max_abs_error"] == 0
    assert audit["action_bound_violations"] == 0
    assert audit["temporal_alignment_violations"] == 0
    assert audit["nominal_continuation_max_abs_error"] == 0
    assert audit["perturbation_type_counts"] == {
        "agent_lateral": 3,
        "agent_retreat": 3,
    }
    assert audit["severity_counts"] == {"low": 2, "medium": 2, "high": 2}
    assert len(audit["perturbation_type_severity_counts"]) == 6
    assert set(audit["variant_final_label_counts"]) == set(VARIANTS)
    assert (
        audit["variant_final_label_counts"]["D_SFN_balanced"]["success"]
        == audit["variant_final_label_counts"]["D_SFR_balanced"]["success"]
    )
    assert audit["recovery_vs_nominal_continuation"] is not None
    assert set(
        audit["recovery_vs_nominal_continuation"]["by_perturbation_cell"]
    ) == {
        "agent_lateral:low",
        "agent_lateral:medium",
        "agent_lateral:high",
        "agent_retreat:low",
        "agent_retreat:medium",
        "agent_retreat:high",
    }
    assert audit["sample_size_recommendation"]["design_floor_scenarios"] == 180
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
                assert len(record["phase"]) == len(record["actions"])
        np.testing.assert_array_equal(initial_states[0], initial_states[1])
        np.testing.assert_array_equal(initial_states[0], initial_states[2])
        with np.load(scenario_dir / "S.npz") as success, np.load(
            scenario_dir / "N.npz"
        ) as nominal_continuation:
            np.testing.assert_array_equal(
                nominal_continuation["sim_state"][0],
                success["sim_state"][metadata["branch_step_in_nominal"]],
            )
            assert len(nominal_continuation["phase"]) == len(
                nominal_continuation["actions"]
            )

    for variant, expected_branches in VARIANTS.items():
        for split, index_metadata in manifest["window_indices"][variant].items():
            index_path = tmp_path / "pilot" / index_metadata["path"]
            with open(index_path, encoding="utf-8") as file:
                windows = [json.loads(line) for line in file]
            assert len(windows) == index_metadata["count"]
            assert all(window["split"] == split for window in windows)
            assert all(window["branch"] in expected_branches for window in windows)


def test_checked_in_remote_pilot_fixture_is_self_consistent():
    with open(FIXTURE_DIR / "audit.json", encoding="utf-8") as file:
        audit = json.load(file)
    with open(FIXTURE_DIR / "manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    scenario_dir = FIXTURE_DIR / "scenario_000000"
    with open(scenario_dir / "metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    assert audit["gate_a_pass"]
    assert manifest["scenario_count"] == 200
    assert manifest["git_commit"] == "a976e2c6f8cdee5c5470881f8703b7f68a642bc3"
    assert metadata["scenario_id"] == "scenario_000000"

    fixture_branches = tuple(manifest["branch_semantics"])
    branch_initials = []
    for branch in fixture_branches:
        assert (scenario_dir / f"{branch}.mp4").is_file()
        with np.load(scenario_dir / f"{branch}.npz") as record:
            assert len(record["oracle_state"]) == len(record["actions"]) + 1
            assert len(record["sim_state"]) == len(record["actions"]) + 1
            if branch != "S":
                branch_initials.append(record["sim_state"][0])
    np.testing.assert_array_equal(branch_initials[0], branch_initials[1])
    np.testing.assert_array_equal(branch_initials[0], branch_initials[2])
