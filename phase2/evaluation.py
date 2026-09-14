"""Held-out prediction, paired ranking and closed-loop recovery evaluation."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .data import NormalizationStats, load_manifest


EVAL_BRANCHES = ("S", "F1", "F2", "R")


def task_error(states):
    """Goal error from the oracle-state task coordinates; lower is better."""

    if isinstance(states, torch.Tensor):
        position = torch.linalg.vector_norm(states[..., 2:4], dim=-1) / 100.0
        angle = torch.abs(torch.atan2(states[..., 4], states[..., 5])) / math.pi
        return position + angle
    states = np.asarray(states)
    position = np.linalg.norm(states[..., 2:4], axis=-1) / 100.0
    angle = np.abs(np.arctan2(states[..., 4], states[..., 5])) / np.pi
    return position + angle


def _denormalize_tensor(states, stats: NormalizationStats):
    mean = torch.as_tensor(stats.state_mean, dtype=states.dtype, device=states.device)
    std = torch.as_tensor(stats.state_std, dtype=states.dtype, device=states.device)
    return states * std + mean


def _test_scenarios(manifest, max_scenarios=None):
    scenarios = sorted(
        scenario_id
        for scenario_id, split in manifest["scenario_splits"].items()
        if split == "test"
    )
    return scenarios if max_scenarios is None else scenarios[:max_scenarios]


def _load_branch(dataset_dir: Path, scenario_id: str, branch: str):
    with np.load(dataset_dir / "scenarios" / scenario_id / f"{branch}.npz") as record:
        return {
            "states": np.asarray(record["oracle_state"], dtype=np.float32).copy(),
            "actions": np.asarray(record["actions"], dtype=np.float32).copy(),
            "coverage": np.asarray(record["coverage"], dtype=np.float32).copy(),
            "success": np.asarray(record["success"], dtype=bool).copy(),
        }


def state_error_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    angle_pred = np.arctan2(predicted[..., 4], predicted[..., 5])
    angle_target = np.arctan2(target[..., 4], target[..., 5])
    angle_diff = (angle_pred - angle_target + np.pi) % (2 * np.pi) - np.pi

    def rmse(indices):
        return float(np.sqrt(np.mean(np.square(predicted[..., indices] - target[..., indices]))))

    return {
        "agent_object_position_rmse_px": rmse(slice(0, 2)),
        "object_goal_position_rmse_px": rmse(slice(2, 4)),
        "angle_mae_rad": float(np.mean(np.abs(angle_diff))),
        "agent_velocity_rmse": rmse(slice(6, 8)),
        "object_velocity_rmse": rmse(slice(8, 10)),
        "object_angular_velocity_rmse": rmse(slice(10, 11)),
        "task_error_mae": float(np.mean(np.abs(task_error(predicted) - task_error(target)))),
        "sample_count": int(predicted.shape[0]),
    }


@torch.no_grad()
def evaluate_prediction_horizons(
    model,
    dataset_dir,
    stats,
    device,
    horizons=(1, 5, 10, 20),
    batch_size=512,
    stride=1,
    max_scenarios=None,
):
    """Evaluate the same universal S/F1/F2/R test set for every model."""

    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    scenarios = _test_scenarios(manifest, max_scenarios)
    results = {}
    model.eval()
    for horizon in horizons:
        initial_states, action_sequences, targets = [], [], []
        for scenario_id in scenarios:
            for branch in EVAL_BRANCHES:
                record = _load_branch(dataset_dir, scenario_id, branch)
                for start in range(0, len(record["actions"]) - horizon + 1, stride):
                    initial_states.append(record["states"][start])
                    action_sequences.append(record["actions"][start : start + horizon])
                    targets.append(record["states"][start + horizon])
        predicted_batches = []
        for start in range(0, len(initial_states), batch_size):
            raw_initial = np.asarray(initial_states[start : start + batch_size])
            normalized_initial = stats.normalize(raw_initial).astype(np.float32)
            state_tensor = torch.from_numpy(normalized_initial[:, None]).to(device)
            action_tensor = torch.from_numpy(
                np.asarray(action_sequences[start : start + batch_size], dtype=np.float32)
            ).to(device)
            predicted_normalized = model.rollout(state_tensor, action_tensor)[:, -1]
            predicted_batches.append(
                _denormalize_tensor(predicted_normalized, stats).cpu().numpy()
            )
        predictions = np.concatenate(predicted_batches, axis=0)
        results[str(horizon)] = state_error_metrics(predictions, np.asarray(targets))
    return results


@torch.no_grad()
def evaluate_counterfactual_ranking(
    model,
    dataset_dir,
    stats,
    device,
    max_scenarios=None,
):
    """Rank the three continuations from each identical post-perturbation state."""

    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    per_pair = []
    model.eval()
    for scenario_id in _test_scenarios(manifest, max_scenarios):
        records = {branch: _load_branch(dataset_dir, scenario_id, branch) for branch in ("F1", "F2", "R")}
        initials = np.stack([records[branch]["states"][0] for branch in ("F1", "F2", "R")])
        if not np.array_equal(initials[0], initials[1]) or not np.array_equal(initials[0], initials[2]):
            raise ValueError(f"Counterfactual branches do not share a state in {scenario_id}")
        predicted_scores, actual_scores = {}, {}
        for branch in ("F1", "F2", "R"):
            record = records[branch]
            initial = stats.normalize(record["states"][0]).astype(np.float32)
            initial_tensor = torch.from_numpy(initial[None, None]).to(device)
            actions = torch.from_numpy(record["actions"][None]).to(device)
            predicted = model.rollout(initial_tensor, actions)[:, -1]
            predicted_raw = _denormalize_tensor(predicted, stats)[0]
            predicted_scores[branch] = float(task_error(predicted_raw).item())
            actual_scores[branch] = float(task_error(record["states"][-1]))
        selected = min(predicted_scores, key=predicted_scores.get)
        actual_best = min(actual_scores.values())
        margin = min(predicted_scores["F1"], predicted_scores["F2"]) - predicted_scores["R"]
        per_pair.append(
            {
                "scenario_id": scenario_id,
                "predicted_task_error": predicted_scores,
                "actual_task_error": actual_scores,
                "selected_branch": selected,
                "recovery_top1": selected == "R",
                "recovery_vs_f1": predicted_scores["R"] < predicted_scores["F1"],
                "recovery_vs_f2": predicted_scores["R"] < predicted_scores["F2"],
                "recovery_margin": margin,
                "selection_regret": actual_scores[selected] - actual_best,
            }
        )
    return {
        "pair_count": len(per_pair),
        "recovery_top1_accuracy": float(np.mean([row["recovery_top1"] for row in per_pair])),
        "recovery_vs_f1_accuracy": float(np.mean([row["recovery_vs_f1"] for row in per_pair])),
        "recovery_vs_f2_accuracy": float(np.mean([row["recovery_vs_f2"] for row in per_pair])),
        "mean_recovery_margin": float(np.mean([row["recovery_margin"] for row in per_pair])),
        "mean_selection_regret": float(np.mean([row["selection_regret"] for row in per_pair])),
        "per_pair": per_pair,
    }


class StateCEMPlanner:
    """Bounded state-space CEM using StateWorldModel rollouts."""

    def __init__(
        self,
        model,
        stats,
        device,
        horizon=12,
        num_samples=256,
        topk=32,
        iterations=4,
        initial_std=0.5,
        action_repeat=3,
        action_cost=0.02,
        smoothness_cost=0.01,
        seed=0,
    ):
        if not 1 < topk <= num_samples:
            raise ValueError("CEM requires 1 < topk <= num_samples")
        self.model = model
        self.stats = stats
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.topk = int(topk)
        self.iterations = int(iterations)
        self.initial_std = float(initial_std)
        self.action_repeat = int(action_repeat)
        self.action_cost = float(action_cost)
        self.smoothness_cost = float(smoothness_cost)
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.control_horizon = int(math.ceil(self.horizon / self.action_repeat))
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(seed)

    @torch.no_grad()
    def plan(self, raw_state, warm_start=None):
        self.model.eval()
        if warm_start is None:
            mean = torch.zeros(self.control_horizon, 2, device=self.device)
        else:
            warm_start = torch.as_tensor(
                warm_start, dtype=torch.float32, device=self.device
            )
            if warm_start.shape != (self.horizon, 2):
                raise ValueError("warm_start has the wrong shape")
            controls = []
            for start in range(0, self.horizon, self.action_repeat):
                controls.append(warm_start[start : start + self.action_repeat].mean(dim=0))
            mean = torch.stack(controls)
        std = torch.full_like(mean, self.initial_std)
        normalized = self.stats.normalize(np.asarray(raw_state, dtype=np.float32)).astype(np.float32)
        initial = torch.from_numpy(normalized).to(self.device)[None, None]
        best_cost = None
        for _ in range(self.iterations):
            noise = torch.randn(
                self.num_samples,
                self.control_horizon,
                2,
                device=self.device,
                generator=self.generator,
            )
            control_candidates = torch.clamp(
                mean[None] + std[None] * noise, -1.0, 1.0
            )
            control_candidates[0] = torch.clamp(mean, -1.0, 1.0)
            candidates = control_candidates.repeat_interleave(
                self.action_repeat, dim=1
            )[:, : self.horizon]
            states = initial.expand(self.num_samples, -1, -1)
            predicted = self.model.rollout(states, candidates)
            final_raw = _denormalize_tensor(predicted[:, -1], self.stats)
            effort = candidates.square().sum(dim=(1, 2))
            smoothness = (candidates[:, 1:] - candidates[:, :-1]).square().sum(
                dim=(1, 2)
            )
            costs = (
                task_error(final_raw)
                + self.action_cost * effort
                + self.smoothness_cost * smoothness
            )
            elite_costs, elite_indices = torch.topk(costs, self.topk, largest=False)
            elite = control_candidates[elite_indices]
            mean = torch.clamp(elite.mean(dim=0), -1.0, 1.0)
            std = torch.clamp(elite.std(dim=0, unbiased=False), min=0.03, max=1.0)
            best_cost = float(elite_costs[0].item())
        action_sequence = mean.repeat_interleave(self.action_repeat, dim=0)[
            : self.horizon
        ]
        return action_sequence.cpu().numpy(), best_cost


def evaluate_closed_loop_recovery(
    model,
    dataset_dir,
    stats,
    device,
    output_dir=None,
    max_scenarios=None,
    max_steps=35,
    save_videos=False,
    planner_kwargs=None,
):
    """Replan after every real simulator step from held-out failure snapshots."""

    # Import lazily so prediction/ranking evaluation remains simulator independent.
    from env.pusht.pusht_env import PushTEnv

    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    scenarios = _test_scenarios(manifest, max_scenarios)
    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    planner_kwargs = {} if planner_kwargs is None else dict(planner_kwargs)
    env = PushTEnv(with_velocity=True, with_target=True, render_size=224)
    rows = []
    try:
        for scenario_index, scenario_id in enumerate(scenarios):
            with open(
                dataset_dir / "scenarios" / scenario_id / "branch_snapshots.pkl", "rb"
            ) as file:
                snapshot = pickle.load(file)["post_perturbation"]
            observation, _ = env.set_sim_state(snapshot)
            frames = [np.asarray(observation["visual"], dtype=np.uint8)]
            planner = StateCEMPlanner(
                model,
                stats,
                device,
                seed=scenario_index,
                **planner_kwargs,
            )
            warm_start = None
            action_cost = 0.0
            coverages = [float(env.evaluate_task()["coverage"])]
            success = bool(env.evaluate_task()["success"])
            steps = 0
            while steps < max_steps and not success:
                sequence, _ = planner.plan(env.get_oracle_state(), warm_start)
                action = np.clip(sequence[0], -1.0, 1.0)
                observation, _, _, _ = env.step(action)
                action_cost += float(np.linalg.norm(action))
                steps += 1
                metrics = env.evaluate_task()
                coverages.append(float(metrics["coverage"]))
                success = bool(metrics["success"])
                if save_videos:
                    frames.append(np.asarray(observation["visual"], dtype=np.uint8))
                warm_start = np.concatenate(
                    [sequence[1:], np.zeros((1, 2), dtype=np.float32)], axis=0
                )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "success": success,
                    "steps": steps,
                    "initial_coverage": coverages[0],
                    "final_coverage": coverages[-1],
                    "max_coverage": max(coverages),
                    "action_cost": action_cost,
                }
            )
            if save_videos and output_dir is not None:
                imageio.mimsave(output_dir / f"{scenario_id}.mp4", frames, fps=10)
    finally:
        env.close()
    return {
        "scenario_count": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_coverage": float(np.mean([row["final_coverage"] for row in rows])),
        "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_action_cost": float(np.mean([row["action_cost"] for row in rows])),
        "per_scenario": rows,
    }


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    temporary.replace(path)
