import os
import pickle
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gym
import numpy as np
import pytest
import torch

import env
from datasets.pusht_dset import (
    compute_valid_stats,
    load_pusht_slice_train_val,
)
from env.pusht.action_utils import get_action_bounds
from env.pusht.pusht_env import PushTEnv
from env.pusht.pusht_wrapper import PushTWrapper
from preprocessor import Preprocessor


def test_pusht_registration_does_not_import_pointmaze():
    assert "env.pointmaze" not in sys.modules
    assert "env.pointmaze.maze_model" not in sys.modules

    registered = gym.make("pusht")
    try:
        observation, state = registered.reset()
        assert observation["visual"].shape == (224, 224, 3)
        assert state.shape == (7,)
    finally:
        registered.close()


def test_relative_action_convention_and_shared_clipping():
    env_instance = PushTEnv(
        with_velocity=True,
        reset_to_state=np.array([256, 400, 256, 300, 0, 0, 0]),
    )
    try:
        env_instance.reset()
        np.testing.assert_allclose(env_instance.action_space.low, [-1.0, -1.0])
        np.testing.assert_allclose(env_instance.action_space.high, [1.0, 1.0])

        _, _, _, info = env_instance.step(np.array([4.0, -3.0]))
        np.testing.assert_allclose(info["command_action"], [1.0, -1.0])
        with pytest.raises(ValueError, match="shape"):
            env_instance.step(np.array([0.0, 0.0, 0.0]))

        absolute_low, absolute_high = get_action_bounds(
            relative=False, action_scale=100.0, workspace_size=512.0
        )
        np.testing.assert_allclose(absolute_low, [0.0, 0.0])
        np.testing.assert_allclose(absolute_high, [5.12, 5.12])
    finally:
        env_instance.close()


def _rollout_from_snapshot(env_instance, snapshot, actions):
    env_instance.set_sim_state(snapshot)
    oracle_states = []
    legacy_states = []
    frames = []
    coverages = []
    for action in actions:
        observation, _, _, info = env_instance.step(action)
        oracle_states.append(env_instance.get_oracle_state())
        legacy_states.append(info["state"])
        frames.append(observation["visual"])
        coverages.append(info["final_coverage"])
    return (
        np.stack(oracle_states),
        np.stack(legacy_states),
        np.stack(frames),
        np.asarray(coverages),
    )


def test_complete_snapshot_restore_has_exact_branch_replay():
    env_instance = PushTEnv(
        with_velocity=True,
        reset_to_state=np.array([256, 400, 256, 300, 0, 0, 0]),
    )
    try:
        env_instance.seed(7)
        env_instance.reset()
        for action in ([0.0, -0.8], [0.0, -0.8], [0.1, -0.6]):
            env_instance.step(np.asarray(action))

        snapshot = env_instance.get_sim_state()
        assert {
            "position",
            "velocity",
            "angle",
            "angular_velocity",
            "force",
            "torque",
            "center_of_gravity",
        } <= snapshot["block"].keys()

        branch_actions = np.asarray(
            [[0.2, -0.5], [-0.3, -0.4], [0.4, 0.1], [-0.2, 0.3]],
            dtype=np.float64,
        )
        first = _rollout_from_snapshot(env_instance, snapshot, branch_actions)
        second = _rollout_from_snapshot(env_instance, snapshot, branch_actions)

        np.testing.assert_allclose(first[0], second[0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(first[1], second[1], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_allclose(first[3], second[3], rtol=0, atol=1e-12)
    finally:
        env_instance.close()


def test_success_depends_on_object_goal_coverage_not_agent_pose():
    env_instance = PushTWrapper()
    try:
        env_instance.reset()
        goal_pose = np.asarray(env_instance.goal_pose)
        perfect = env_instance.evaluate_task(
            block_pose=goal_pose, goal_pose=goal_pose
        )
        assert perfect["success"]
        assert perfect["coverage"] == pytest.approx(1.0, abs=1e-12)

        goal_state = np.array([30, 30, *goal_pose, 0, 0], dtype=np.float64)
        cur_state = np.array([480, 480, *goal_pose, 100, -100], dtype=np.float64)
        result = env_instance.eval_state(goal_state, cur_state)
        assert result["success"]
        assert result["coverage"] == pytest.approx(1.0, abs=1e-12)
    finally:
        env_instance.close()


def test_compute_valid_stats_ignores_padding():
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]],
            [[5.0, 6.0], [999.0, 999.0], [999.0, 999.0]],
        ]
    )
    mean, std = compute_valid_stats(values, [2, 1])
    valid = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    torch.testing.assert_close(mean, valid.mean(dim=0))
    torch.testing.assert_close(std, valid.std(dim=0, unbiased=False))


def _write_minimal_dataset(path, actions, states, velocities, seq_lengths):
    path.mkdir(parents=True)
    torch.save(actions, path / "rel_actions.pth")
    torch.save(states, path / "states.pth")
    torch.save(velocities, path / "velocities.pth")
    with open(path / "seq_lengths.pkl", "wb") as file:
        pickle.dump(seq_lengths, file)


def test_validation_dataset_reuses_dynamic_training_statistics(tmp_path):
    train_actions = torch.tensor(
        [
            [[100.0, -100.0], [0.0, 100.0], [99900.0, 99900.0]],
            [[-100.0, 0.0], [99900.0, 99900.0], [99900.0, 99900.0]],
        ]
    )
    train_states = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
    train_states[0, 2] = 99999
    train_states[1, 1:] = 99999
    train_velocities = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    train_velocities[0, 2] = 99999
    train_velocities[1, 1:] = 99999
    _write_minimal_dataset(
        tmp_path / "train",
        train_actions,
        train_states,
        train_velocities,
        [2, 1],
    )

    _write_minimal_dataset(
        tmp_path / "val",
        torch.full((2, 3, 2), 500.0),
        torch.full((2, 3, 5), 500.0),
        torch.full((2, 3, 2), 500.0),
        [2, 1],
    )

    _, trajectory_datasets = load_pusht_slice_train_val(
        transform=None,
        n_rollout=None,
        data_path=str(tmp_path),
        normalize_action=True,
        num_hist=1,
        num_pred=1,
        frameskip=1,
        with_velocity=True,
    )
    train = trajectory_datasets["train"]
    valid = trajectory_datasets["valid"]

    expected_actions = torch.tensor([[1.0, -1.0], [0.0, 1.0], [-1.0, 0.0]])
    torch.testing.assert_close(train.action_mean, expected_actions.mean(dim=0))
    torch.testing.assert_close(
        train.action_std, expected_actions.std(dim=0, unbiased=False)
    )
    torch.testing.assert_close(valid.action_mean, train.action_mean)
    torch.testing.assert_close(valid.action_std, train.action_std)
    torch.testing.assert_close(valid.state_mean, train.state_mean)
    torch.testing.assert_close(valid.proprio_mean, train.proprio_mean)


def test_preprocessor_clips_flattened_normalized_action_chunks():
    preprocessor = Preprocessor(
        action_mean=torch.tensor([0.2, -0.2]),
        action_std=torch.tensor([0.5, 0.25]),
        state_mean=torch.zeros(1),
        state_std=torch.ones(1),
        proprio_mean=torch.zeros(1),
        proprio_std=torch.ones(1),
        transform=None,
        action_low=torch.tensor([-1.0, -1.0]),
        action_high=torch.tensor([1.0, 1.0]),
    )
    normalized = torch.tensor([[[20.0, -20.0, -20.0, 20.0]]])
    clipped = preprocessor.clamp_normalized_actions(normalized)
    physical = preprocessor.denormalize_actions(clipped.reshape(1, 1, 2, 2))
    assert torch.all(physical >= -1.0)
    assert torch.all(physical <= 1.0)
    torch.testing.assert_close(
        physical,
        torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
    )
