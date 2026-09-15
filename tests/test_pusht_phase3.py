import json

import numpy as np
import torch

from phase3.cache_dino import render_sim_state
from env.pusht.pusht_env import PushTEnv
from phase3.evaluate_probes import _fit_classifier, _fit_progress
from models.visual_temporal_model import VisualTemporalWorldModel
from phase1.pusht_dataset import snapshot_to_vector
from phase3.data import (
    ProprioStats,
    VisualFeatureWindowDataset,
    VisualProbeFrameDataset,
    compute_proprio_stats,
)


def _write_cache(tmp_path):
    variants = {
        "D_SF": ["S", "F1"],
        "D_SFN_balanced": ["S", "F1", "N"],
        "D_SFR_balanced": ["S", "F1", "R"],
    }
    scenario_splits = {"scenario_train": "train", "scenario_valid": "valid"}
    lengths = {}
    phases = {
        "S": "nominal_push",
        "N": "nominal_push",
        "F1": "open_loop_failure",
        "F2": "neutral",
        "R": "reposition",
    }
    for scenario_index, scenario_id in enumerate(scenario_splits):
        lengths[scenario_id] = {}
        scenario_dir = tmp_path / "scenarios" / scenario_id
        scenario_dir.mkdir(parents=True)
        for branch_index, branch in enumerate(("S", "N", "F1", "F2", "R")):
            observations = 6
            lengths[scenario_id][branch] = {"observations": observations, "actions": 5}
            base = float(scenario_index * 10 + branch_index)
            np.savez(
                scenario_dir / f"{branch}.npz",
                tokens=np.full((observations, 4, 8), base, dtype=np.float16),
                proprio=np.arange(observations * 4, dtype=np.float32).reshape(observations, 4) + base,
                actions=np.zeros((5, 2), dtype=np.float32),
                oracle_state=np.zeros((observations, 11), dtype=np.float32),
                coverage=np.linspace(0.2, 1.0, observations, dtype=np.float32),
                success=np.asarray([False] * 5 + [True]),
                phase=np.asarray([phases[branch]] * 5),
            )
    manifest = {
        "cache_schema": "pusht-recovery-dino-cache-v1",
        "source_gate_a_pass": True,
        "split_by_scenario_before_windows": True,
        "scenario_splits": scenario_splits,
        "variants": variants,
        "trajectory_lengths": lengths,
        "pooled_patch_count": 4,
        "embedding_dim": 8,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_visual_cache_datasets_and_labels(tmp_path):
    _write_cache(tmp_path)
    stats = compute_proprio_stats(tmp_path, "D_SF")
    train = VisualFeatureWindowDataset(
        tmp_path, "D_SFN_balanced", "train", window_actions=3, stats=stats
    )
    assert len(train) == 9
    sample = train[0]
    assert sample["tokens"].shape == (4, 4, 8)
    assert sample["proprio"].shape == (4, 4)
    assert sample["actions"].shape == (3, 2)

    probes = VisualProbeFrameDataset(
        tmp_path, "valid", stats, history=3, recoverability_horizon=2
    )
    rows = [probes[index] for index in range(len(probes))]
    f1_rows = [row for row in rows if row["branch"] == "F1"]
    assert f1_rows and all(not row["off_nominal_mask"] for row in f1_rows)
    recovery_rows = [row for row in rows if row["branch"] == "R"]
    nominal_rows = [row for row in rows if row["branch"] == "N"]
    assert recovery_rows and all(row["off_nominal"] == 1 for row in recovery_rows)
    assert nominal_rows and all(row["off_nominal"] == 0 for row in nominal_rows)
    assert all(not row["recoverability_mask"] for row in f1_rows)
    s_rows = [row for row in rows if row["branch"] == "S"]
    assert any(row["recoverable"] == 1 for row in s_rows)


def test_visual_temporal_model_forward_rollout_and_representation():
    model = VisualTemporalWorldModel(
        patch_count=4,
        visual_dim=8,
        max_context=3,
        model_dim=16,
        depth=2,
        heads=2,
        mlp_dim=32,
        dim_head=8,
        dropout=0.0,
    )
    visual = torch.randn(2, 4, 4, 8)
    proprio = torch.randn(2, 4, 4)
    actions = torch.randn(2, 3, 2)
    output = model(visual, proprio, actions)
    assert output["visual"].shape == (2, 3, 4, 8)
    assert output["proprio"].shape == (2, 3, 4)
    assert torch.isfinite(output["loss"])
    rollout_visual, rollout_proprio = model.rollout(
        visual[:, :1], proprio[:, :1], actions
    )
    assert rollout_visual.shape == (2, 3, 4, 8)
    assert rollout_proprio.shape == (2, 3, 4)
    representation = model.contextual_representation(
        visual[:, 1:], proprio[:, 1:], actions[:, 1:]
    )
    assert representation.shape == (2, 32)


def test_sim_state_render_is_deterministic():
    env = PushTEnv(with_velocity=True, with_target=True, render_size=96)
    env.reset()
    vector = snapshot_to_vector(env.get_sim_state())
    first = render_sim_state(env, vector)
    second = render_sim_state(env, vector)
    assert first.shape == (96, 96, 3)
    assert np.array_equal(first, second)


def test_probe_fitting_returns_sample_aligned_predictions():
    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.2], [0.8, 0.9], [1.0, 1.0]], dtype=np.float32
    )
    payload = {
        "features": features,
        "coverage": np.asarray([0.0, 0.1, 0.9, 1.0]),
        "off_nominal": np.asarray([0, 0, 1, 1]),
        "off_nominal_mask": np.ones(4, dtype=bool),
        "scenario_id": np.asarray(["a", "a", "b", "b"]),
    }
    progress = _fit_progress(payload, payload, payload, [0.1, 1.0])
    classifier = _fit_classifier(
        payload, payload, payload, "off_nominal", "off_nominal_mask", [0.1, 1.0]
    )
    assert len(progress["test_predictions"]) == 4
    assert progress["test"]["r2"] > 0.8
    assert len(classifier["test_probabilities"]) == 4
    assert classifier["test"]["roc_auc"] == 1.0
