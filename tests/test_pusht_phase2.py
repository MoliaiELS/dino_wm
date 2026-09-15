import json
from pathlib import Path

import numpy as np
import pytest
import torch

from compare_state_wm import paired_bootstrap
from models.state_world_model import StateWorldModel
from phase2.data import (
    NormalizationStats,
    PairedStateWindowDataset,
    branch_balanced_weights,
    compute_train_normalization,
)
from phase2.evaluation import (
    StateCEMPlanner,
    state_error_metrics,
    state_planning_cost,
    task_error,
)
from train_state_wm import _run_epoch


BRANCHES = ("S", "F1", "F2", "R")


def _write_phase2_fixture(root: Path):
    (root / "indices").mkdir(parents=True)
    (root / "scenarios").mkdir()
    scenario_splits = {
        "scenario_train": "train",
        "scenario_valid": "valid",
        "scenario_test": "test",
    }
    variants = {
        "D_SF_balanced": ["S", "F1", "F2"],
        "D_SFR_balanced": ["S", "F1", "R"],
    }
    window_indices = {variant: {} for variant in variants}
    for scenario_number, (scenario_id, split) in enumerate(scenario_splits.items()):
        scenario_dir = root / "scenarios" / scenario_id
        scenario_dir.mkdir()
        offset = scenario_number * 1000.0
        for branch_number, branch in enumerate(BRANCHES):
            states = np.full((6, 11), offset + branch_number, dtype=np.float32)
            states[:, 5] = 1.0
            actions = np.full((5, 2), 0.1 * branch_number, dtype=np.float32)
            coverage = np.linspace(0.0, 1.0, 6, dtype=np.float32)
            np.savez_compressed(
                scenario_dir / f"{branch}.npz",
                oracle_state=states,
                actions=actions,
                coverage=coverage,
            )
    for variant, branches in variants.items():
        for scenario_id, split in scenario_splits.items():
            path = root / "indices" / f"{variant}_{split}.jsonl"
            with open(path, "w", encoding="utf-8") as file:
                for branch in branches:
                    file.write(
                        json.dumps(
                            {
                                "scenario_id": scenario_id,
                                "pair_id": scenario_id.replace("scenario", "pair"),
                                "split": split,
                                "branch": branch,
                                "start": 0,
                                "end": 6,
                            }
                        )
                        + "\n"
                    )
            window_indices[variant][split] = {
                "path": str(path.relative_to(root)),
                "count": len(branches),
                "branches": branches,
            }
    manifest = {
        "schema_version": "pusht-recovery-pairs-v1",
        "gate_a_pass": True,
        "split_by_scenario_before_windows": True,
        "scenario_splits": scenario_splits,
        "variants": variants,
        "window_indices": window_indices,
        "action_convention": {
            "command_low": [-1.0, -1.0],
            "command_high": [1.0, 1.0],
        },
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file)


def _upgrade_fixture_with_nominal_control(root: Path):
    manifest_path = root / "manifest.json"
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    manifest["schema_version"] = "pusht-recovery-pairs-v2"
    manifest["variants"]["D_SFN_balanced"] = ["S", "F1", "N"]
    manifest["window_indices"]["D_SFN_balanced"] = {}
    for scenario_id in manifest["scenario_splits"]:
        scenario_dir = root / "scenarios" / scenario_id
        with np.load(scenario_dir / "R.npz") as record:
            np.savez_compressed(
                scenario_dir / "N.npz",
                **{key: record[key] for key in record.files},
            )
    for split in ("train", "valid", "test"):
        source_metadata = manifest["window_indices"]["D_SFR_balanced"][split]
        source = root / source_metadata["path"]
        target = root / "indices" / f"D_SFN_balanced_{split}.jsonl"
        rows = []
        with open(source, encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                if row["branch"] == "R":
                    row["branch"] = "N"
                rows.append(row)
        with open(target, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        manifest["window_indices"]["D_SFN_balanced"][split] = {
            "path": str(target.relative_to(root)),
            "count": len(rows),
            "branches": ["S", "F1", "N"],
        }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file)


def test_phase2_loader_uses_manifest_windows_and_train_only_stats(tmp_path):
    dataset_dir = tmp_path / "paired"
    _write_phase2_fixture(dataset_dir)
    stats = compute_train_normalization(dataset_dir, "D_SFR_balanced")
    # Validation/test states were shifted by 1000/2000 and cannot affect this mean.
    np.testing.assert_allclose(stats.state_mean[[0, 4, 6]], [4 / 3, 4 / 3, 4 / 3])
    assert stats.state_std[5] == 1.0
    assert stats.count == 18

    train = PairedStateWindowDataset(
        dataset_dir, "D_SFR_balanced", "train", stats=stats
    )
    valid = PairedStateWindowDataset(
        dataset_dir, "D_SFR_balanced", "valid", stats=stats
    )
    item = train[0]
    assert item["states"].shape == (6, 11)
    assert item["actions"].shape == (5, 2)
    assert item["coverage"].shape == (6,)
    assert set(train.branch_counts) == {"S", "F1", "R"}
    assert valid.stats is stats
    weights = branch_balanced_weights(train.entries)
    assert weights.shape == (len(train),)


def test_validation_loader_refuses_to_compute_its_own_stats(tmp_path):
    dataset_dir = tmp_path / "paired"
    _write_phase2_fixture(dataset_dir)
    with pytest.raises(ValueError, match="training statistics"):
        PairedStateWindowDataset(dataset_dir, "D_SF_balanced", "valid")


def test_loader_accepts_success_matched_v2_variant(tmp_path):
    dataset_dir = tmp_path / "paired_v2"
    _write_phase2_fixture(dataset_dir)
    _upgrade_fixture_with_nominal_control(dataset_dir)
    stats = compute_train_normalization(dataset_dir, "D_SFN_balanced")
    train = PairedStateWindowDataset(
        dataset_dir, "D_SFN_balanced", "train", stats=stats
    )
    assert set(train.branch_counts) == {"S", "F1", "N"}


def test_state_world_model_is_cpu_safe_causal_and_rolls_out():
    torch.manual_seed(3)
    model = StateWorldModel(
        max_context=5,
        model_dim=32,
        state_emb_dim=16,
        action_emb_dim=8,
        depth=2,
        heads=4,
        mlp_dim=48,
        dim_head=8,
        dropout=0.0,
    )
    model.eval()
    states = torch.randn(2, 6, 11)
    actions = torch.randn(2, 5, 2).clamp(-1, 1)
    predicted, loss = model(states, actions)
    assert predicted.shape == (2, 5, 11)
    assert loss.ndim == 0 and torch.isfinite(loss)

    changed_states = states[:, :-1].clone()
    changed_actions = actions.clone()
    changed_states[:, 3:] += 100
    changed_actions[:, 3:] -= 100
    original = model.predict_next(states[:, :-1], actions)
    changed = model.predict_next(changed_states, changed_actions)
    torch.testing.assert_close(original[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6)

    rollout = model.rollout(states[:, :1], actions[:, :4])
    assert rollout.shape == (2, 4, 11)


def test_task_error_and_state_metrics_have_physical_units():
    target = np.zeros((2, 11), dtype=np.float32)
    target[:, 5] = 1.0
    predicted = target.copy()
    predicted[:, 2] = 10.0
    assert np.allclose(task_error(predicted), 0.1)
    metrics = state_error_metrics(predicted, target)
    assert metrics["object_goal_position_rmse_px"] == pytest.approx(
        np.sqrt(50.0)
    )
    assert metrics["angle_mae_rad"] == 0.0
    assert metrics["sample_count"] == 2


def test_paired_bootstrap_reports_the_paired_mean():
    result = paired_bootstrap([1.0, 0.0, 1.0, 0.0], samples=200, seed=4)
    assert result["n_pairs"] == 4
    assert result["mean"] == 0.5
    assert result["bootstrap_95_ci"][0] <= 0.5
    assert result["bootstrap_95_ci"][1] >= 0.5


def test_state_cem_uses_bounded_low_frequency_action_blocks():
    class DummyWorldModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def rollout(self, initial_states, actions):
            states = initial_states[:, -1:].repeat(1, actions.shape[1], 1)
            states = states.clone()
            states[:, :, 2] += torch.cumsum(actions[:, :, 0], dim=1)
            states[:, :, 3] += torch.cumsum(actions[:, :, 1], dim=1)
            return states

    stats = NormalizationStats(
        state_mean=np.zeros(11, dtype=np.float32),
        state_std=np.ones(11, dtype=np.float32),
        count=1,
        variant="test",
    )
    planner = StateCEMPlanner(
        DummyWorldModel(),
        stats,
        "cpu",
        horizon=5,
        num_samples=16,
        topk=4,
        iterations=2,
        action_repeat=2,
        seed=7,
    )
    initial = np.zeros(11, dtype=np.float32)
    initial[2:4] = 5.0
    initial[5] = 1.0
    actions, cost = planner.plan(initial)
    assert actions.shape == (5, 2)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
    np.testing.assert_allclose(actions[0], actions[1])
    np.testing.assert_allclose(actions[2], actions[3])
    assert np.isfinite(cost)


def test_state_planning_cost_rewards_the_goal_aligned_staging_pose():
    states = torch.zeros(2, 11)
    states[:, 3] = 50.0
    states[:, 5] = 1.0
    states[0, 1] = 139.0
    states[1, 0] = 139.0
    costs = state_planning_cost(states, staging_weight=0.25)
    assert costs[0] < costs[1]


def test_training_epoch_combines_teacher_forced_and_rollout_losses():
    model = StateWorldModel(
        max_context=5,
        model_dim=32,
        state_emb_dim=16,
        action_emb_dim=8,
        depth=1,
        heads=4,
        mlp_dim=32,
        dim_head=8,
        dropout=0.0,
    )
    loader = [
        {
            "states": torch.randn(2, 6, 11),
            "actions": torch.randn(2, 5, 2).clamp(-1, 1),
        }
    ]
    metrics = _run_epoch(
        model,
        loader,
        torch.device("cpu"),
        rollout_horizon=3,
        rollout_weight=1.0,
    )
    assert metrics["loss"] == pytest.approx(
        metrics["one_step_loss"] + metrics["rollout_loss"]
    )
    assert metrics["rollout_loss"] > 0
