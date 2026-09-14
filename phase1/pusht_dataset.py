"""Paired PushT counterfactual dataset generation and auditing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np

from env.pusht.pusht_env import PushTEnv
from .pusht_oracle import GeometricPushTOracle, OracleConfig


SCHEMA_VERSION = "pusht-recovery-pairs-v1"
BRANCHES = ("S", "F1", "F2", "R")
VARIANTS = {
    "D_S": ("S",),
    "D_SF": ("S", "F1"),
    "D_SFR": ("S", "F1", "R"),
    "D_SF_balanced": ("S", "F1", "F2"),
    "D_SFR_balanced": ("S", "F1", "R"),
}
PERTURBATION_LEVELS = {
    "agent_lateral": {"low": 30.0, "medium": 55.0, "high": 80.0},
    "agent_retreat": {"low": 30.0, "medium": 55.0, "high": 80.0},
}
SIM_STATE_FIELDS = (
    "agent.position.x",
    "agent.position.y",
    "agent.velocity.x",
    "agent.velocity.y",
    "agent.angle",
    "agent.angular_velocity",
    "agent.force.x",
    "agent.force.y",
    "agent.torque",
    "agent.center_of_gravity.x",
    "agent.center_of_gravity.y",
    "block.position.x",
    "block.position.y",
    "block.velocity.x",
    "block.velocity.y",
    "block.angle",
    "block.angular_velocity",
    "block.force.x",
    "block.force.y",
    "block.torque",
    "block.center_of_gravity.x",
    "block.center_of_gravity.y",
    "goal_pose.x",
    "goal_pose.y",
    "goal_pose.angle",
    "space_damping",
    "success_threshold",
)


@dataclass(frozen=True)
class Phase1Config:
    num_scenarios: int = 200
    seed: int = 20260914
    render_size: int = 224
    fps: int = 10
    nominal_horizon: int = 50
    branch_horizon: int = 35
    target_branch_progress: float = 0.35
    window_length: int = 21
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "valid": 0.15, "test": 0.15}
    )
    goal_x_range: tuple = (220.0, 292.0)
    goal_y_range: tuple = (165.0, 215.0)
    goal_angle_range: tuple = (-0.20, 0.20)
    initial_distance_range: tuple = (70.0, 100.0)
    # The T stem extends about 120 px behind the body origin. Together with
    # the 15 px agent radius and a small margin, 139 px is contact-safe.
    initial_agent_offset: float = 139.0
    save_videos: bool = True
    verify_videos: bool = True
    bootstrap_samples: int = 5000
    oracle: OracleConfig = field(default_factory=OracleConfig)

    def validate(self):
        if self.num_scenarios <= 0:
            raise ValueError("num_scenarios must be positive")
        if self.nominal_horizon < 2 or self.branch_horizon < 1:
            raise ValueError("trajectory horizons are too short")
        if self.window_length < 2:
            raise ValueError("window_length must include an action transition")
        if set(self.split_ratios) != {"train", "valid", "test"}:
            raise ValueError("split_ratios must define train, valid and test")
        if not np.isclose(sum(self.split_ratios.values()), 1.0):
            raise ValueError("split ratios must sum to one")
        if any(value < 0 for value in self.split_ratios.values()):
            raise ValueError("split ratios cannot be negative")


@dataclass
class TrajectoryRecord:
    actions: np.ndarray
    rgb: np.ndarray
    proprio: np.ndarray
    oracle_state: np.ndarray
    legacy_state: np.ndarray
    sim_state: np.ndarray
    coverage: np.ndarray
    success: np.ndarray
    contacts: np.ndarray
    snapshots: Optional[List[dict]] = None

    @property
    def action_count(self):
        return int(self.actions.shape[0])

    @property
    def first_success_step(self):
        indices = np.flatnonzero(self.success)
        return None if len(indices) == 0 else int(indices[0])


def _body_state_vector(body_state):
    return np.concatenate(
        [
            np.asarray(body_state["position"], dtype=np.float64),
            np.asarray(body_state["velocity"], dtype=np.float64),
            np.asarray(
                [body_state["angle"], body_state["angular_velocity"]],
                dtype=np.float64,
            ),
            np.asarray(body_state["force"], dtype=np.float64),
            np.asarray([body_state["torque"]], dtype=np.float64),
            np.asarray(body_state["center_of_gravity"], dtype=np.float64),
        ]
    )


def snapshot_to_vector(snapshot):
    vector = np.concatenate(
        [
            _body_state_vector(snapshot["agent"]),
            _body_state_vector(snapshot["block"]),
            np.asarray(snapshot["goal_pose"], dtype=np.float64),
            np.asarray(
                [snapshot["space_damping"], snapshot["success_threshold"]],
                dtype=np.float64,
            ),
        ]
    )
    if vector.shape != (len(SIM_STATE_FIELDS),):
        raise RuntimeError(f"Unexpected simulator vector shape {vector.shape}")
    return vector


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_default, sort_keys=True)
    temporary.replace(path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _split_scenarios(num_scenarios, ratios, seed):
    names = ("train", "valid", "test")
    raw_counts = np.asarray([ratios[name] * num_scenarios for name in names])
    counts = np.floor(raw_counts).astype(int)
    remainders = raw_counts - counts
    for index in np.argsort(-remainders)[: num_scenarios - int(counts.sum())]:
        counts[index] += 1

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_scenarios)
    split_map = {}
    cursor = 0
    for name, count in zip(names, counts):
        for scenario_index in permutation[cursor : cursor + count]:
            split_map[f"scenario_{scenario_index:06d}"] = name
        cursor += int(count)
    return split_map


def _capture_observation(env, observation, legacy_state):
    snapshot = env.get_sim_state()
    metrics = env.evaluate_task()
    return {
        "rgb": np.asarray(observation["visual"], dtype=np.uint8).copy(),
        "proprio": np.asarray(observation["proprio"], dtype=np.float32).copy(),
        "oracle_state": env.get_oracle_state().copy(),
        "legacy_state": np.asarray(legacy_state, dtype=np.float32).copy(),
        "sim_state": snapshot_to_vector(snapshot),
        "coverage": metrics["coverage"],
        "success": metrics["success"],
        "snapshot": snapshot,
    }


def rollout_from_snapshot(
    env,
    snapshot,
    horizon,
    controller=None,
    actions=None,
    capture_snapshots=False,
):
    """Roll out exactly ``horizon`` actions from one immutable snapshot."""
    if (controller is None) == (actions is None):
        raise ValueError("Provide exactly one of controller or actions")
    if actions is not None and len(actions) != horizon:
        raise ValueError("Fixed action sequence must match the requested horizon")

    observation, legacy_state = env.set_sim_state(snapshot)
    first = _capture_observation(env, observation, legacy_state)
    buffers = {key: [first[key]] for key in (
        "rgb", "proprio", "oracle_state", "legacy_state", "sim_state",
        "coverage", "success",
    )}
    snapshots = [first["snapshot"]] if capture_snapshots else None
    action_buffer = []
    contact_buffer = []

    for step in range(horizon):
        action = controller.act(env) if controller is not None else actions[step]
        observation, _, _, info = env.step(action)
        captured = _capture_observation(env, observation, info["state"])
        for key in buffers:
            buffers[key].append(captured[key])
        if capture_snapshots:
            snapshots.append(captured["snapshot"])
        action_buffer.append(np.asarray(info["command_action"], dtype=np.float32))
        contact_buffer.append(int(info["n_contacts"]))

    return TrajectoryRecord(
        actions=np.stack(action_buffer),
        rgb=np.stack(buffers["rgb"]),
        proprio=np.stack(buffers["proprio"]),
        oracle_state=np.stack(buffers["oracle_state"]),
        legacy_state=np.stack(buffers["legacy_state"]),
        sim_state=np.stack(buffers["sim_state"]),
        coverage=np.asarray(buffers["coverage"], dtype=np.float32),
        success=np.asarray(buffers["success"], dtype=bool),
        contacts=np.asarray(contact_buffer, dtype=np.int16),
        snapshots=snapshots,
    )


def _trajectory_metadata(record, video_name):
    return {
        "action_count": record.action_count,
        "observation_count": int(record.rgb.shape[0]),
        "video": video_name,
        "rgb_sha256_before_encoding": hashlib.sha256(record.rgb.tobytes()).hexdigest(),
        "initial_coverage": float(record.coverage[0]),
        "final_coverage": float(record.coverage[-1]),
        "max_coverage": float(record.coverage.max()),
        "final_success": bool(record.success[-1]),
        "ever_success": bool(record.success.any()),
        "first_success_step": record.first_success_step,
        "timeout": not bool(record.success.any()),
        "action_cost": float(np.linalg.norm(record.actions, axis=1).sum()),
        "contact_steps": int(np.count_nonzero(record.contacts)),
    }


def save_trajectory(directory, branch, record, fps, save_video):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{branch}.npz",
        actions=record.actions,
        proprio=record.proprio,
        oracle_state=record.oracle_state,
        legacy_state=record.legacy_state,
        sim_state=record.sim_state,
        coverage=record.coverage,
        success=record.success,
        contacts=record.contacts,
    )
    video_name = None
    if save_video:
        video_name = f"{branch}.mp4"
        imageio.mimsave(directory / video_name, record.rgb, fps=fps)
    return _trajectory_metadata(record, video_name)


class PushTPhase1Generator:
    def __init__(self, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.config = Phase1Config() if config is None else config
        self.config.validate()
        self.oracle = GeometricPushTOracle(self.config.oracle)

    def _scenario_spec(self, scenario_index):
        rng = np.random.default_rng(self.config.seed + scenario_index * 104729)
        goal_angle = rng.uniform(*self.config.goal_angle_range)
        backward = np.asarray([-np.sin(goal_angle), np.cos(goal_angle)])
        goal_pose = np.asarray(
            [
                rng.uniform(*self.config.goal_x_range),
                rng.uniform(*self.config.goal_y_range),
                goal_angle,
            ],
            dtype=np.float64,
        )
        initial_distance = rng.uniform(*self.config.initial_distance_range)
        block_position = goal_pose[:2] + backward * initial_distance
        agent_position = (
            block_position + backward * self.config.initial_agent_offset
        )
        initial_state = np.asarray(
            [*agent_position, *block_position, goal_angle, 0.0, 0.0],
            dtype=np.float64,
        )

        perturbation_types = tuple(PERTURBATION_LEVELS)
        severity_names = ("low", "medium", "high")
        perturbation_type = perturbation_types[scenario_index % len(perturbation_types)]
        severity = severity_names[scenario_index % len(severity_names)]
        sign = -1.0 if (scenario_index // 6) % 2 else 1.0
        return {
            "goal_pose": goal_pose,
            "initial_state": initial_state,
            "initial_distance": initial_distance,
            "perturbation_type": perturbation_type,
            "severity": severity,
            "requested_magnitude": PERTURBATION_LEVELS[perturbation_type][severity],
            "sign": sign,
        }

    @staticmethod
    def _select_branch_step(record, target_progress):
        distances = np.linalg.norm(record.oracle_state[:, 2:4], axis=1)
        initial_distance = max(float(distances[0]), 1e-9)
        progress = 1.0 - distances / initial_distance
        success_indices = np.flatnonzero(record.success)
        last_index = (
            int(success_indices[0]) - 2
            if len(success_indices)
            else record.action_count - 2
        )
        last_index = max(2, min(last_index, record.action_count - 2))
        candidates = np.arange(2, last_index + 1)
        selected = int(candidates[np.argmin(np.abs(progress[candidates] - target_progress))])
        return selected, float(progress[selected])

    @staticmethod
    def _apply_perturbation(snapshot, spec):
        perturbed = copy.deepcopy(snapshot)
        block_position = np.asarray(perturbed["block"]["position"], dtype=np.float64)
        goal_position = np.asarray(perturbed["goal_pose"][:2], dtype=np.float64)
        motion = goal_position - block_position
        motion /= max(float(np.linalg.norm(motion)), 1e-9)
        if spec["perturbation_type"] == "agent_lateral":
            perturbation_direction = np.asarray([-motion[1], motion[0]]) * spec["sign"]
        elif spec["perturbation_type"] == "agent_retreat":
            perturbation_direction = -motion
        else:
            raise ValueError(f"Unknown perturbation type {spec['perturbation_type']}")
        displacement = perturbation_direction * spec["requested_magnitude"]
        target = "agent"
        original = np.asarray(perturbed[target]["position"], dtype=np.float64)
        updated = np.clip(original + displacement, 25.0, 487.0)
        perturbed[target]["position"] = updated
        actual_displacement = updated - original
        return perturbed, {
            "target": target,
            "vector": actual_displacement,
            "magnitude": float(np.linalg.norm(actual_displacement)),
            "preserves_velocity": True,
            "transition_is_exogenous": True,
        }

    @staticmethod
    def _pad_actions(actions, horizon):
        actions = np.asarray(actions, dtype=np.float64)
        if len(actions) >= horizon:
            return actions[:horizon]
        padding = np.zeros((horizon - len(actions), 2), dtype=np.float64)
        return np.concatenate([actions, padding], axis=0)

    def _generate_scenario(self, env, scenario_index, split):
        scenario_id = f"scenario_{scenario_index:06d}"
        pair_id = f"pair_{scenario_index:06d}"
        spec = self._scenario_spec(scenario_index)
        env.seed(self.config.seed + scenario_index)
        env.reset_to_state = spec["initial_state"]
        env.reset()
        env.set_task_goal(spec["goal_pose"].copy())
        initial_snapshot = env.get_sim_state()

        nominal = rollout_from_snapshot(
            env,
            initial_snapshot,
            self.config.nominal_horizon,
            controller=self.oracle,
            capture_snapshots=True,
        )
        if not nominal.success[-1]:
            raise RuntimeError(
                f"Oracle failed nominal scenario {scenario_id}; "
                f"final coverage={nominal.coverage[-1]:.4f}"
            )

        branch_step, branch_progress = self._select_branch_step(
            nominal, self.config.target_branch_progress
        )
        pre_perturb_snapshot = nominal.snapshots[branch_step]
        post_perturb_snapshot, applied = self._apply_perturbation(
            pre_perturb_snapshot, spec
        )
        post_vector = snapshot_to_vector(post_perturb_snapshot)
        branch_hash = hashlib.sha256(post_vector.tobytes()).hexdigest()

        f1_actions = self._pad_actions(
            nominal.actions[branch_step:], self.config.branch_horizon
        )
        f2_actions = np.zeros((self.config.branch_horizon, 2), dtype=np.float64)
        f1 = rollout_from_snapshot(
            env, post_perturb_snapshot, self.config.branch_horizon, actions=f1_actions
        )
        f2 = rollout_from_snapshot(
            env, post_perturb_snapshot, self.config.branch_horizon, actions=f2_actions
        )
        recovery = rollout_from_snapshot(
            env,
            post_perturb_snapshot,
            self.config.branch_horizon,
            controller=self.oracle,
        )

        scenario_dir = self.output_dir / "scenarios" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=False)
        trajectory_metadata = {}
        for branch, record in {
            "S": nominal,
            "F1": f1,
            "F2": f2,
            "R": recovery,
        }.items():
            trajectory_metadata[branch] = save_trajectory(
                scenario_dir,
                branch,
                record,
                fps=self.config.fps,
                save_video=self.config.save_videos,
            )

        with open(scenario_dir / "branch_snapshots.pkl", "wb") as file:
            pickle.dump(
                {
                    "pre_perturbation": pre_perturb_snapshot,
                    "post_perturbation": post_perturb_snapshot,
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        metadata = {
            "scenario_id": scenario_id,
            "pair_id": pair_id,
            "split": split,
            "seed": self.config.seed + scenario_index,
            "shape": env.shape,
            "goal_pose": spec["goal_pose"],
            "initial_state": spec["initial_state"],
            "branch_step_in_nominal": branch_step,
            "branch_progress": branch_progress,
            "branch_state_sha256": branch_hash,
            "perturbation": {
                "type": spec["perturbation_type"],
                "severity": spec["severity"],
                "requested_magnitude": spec["requested_magnitude"],
                **applied,
            },
            "branches": trajectory_metadata,
        }
        _write_json(scenario_dir / "metadata.json", metadata)
        return metadata

    def _build_window_indices(self, scenario_metadata, split_map):
        index_dir = self.output_dir / "indices"
        index_dir.mkdir()
        index_manifest = {}
        metadata_by_id = {entry["scenario_id"]: entry for entry in scenario_metadata}

        for variant, branches in VARIANTS.items():
            index_manifest[variant] = {}
            for split in ("train", "valid", "test"):
                index_path = index_dir / f"{variant}_{split}.jsonl"
                count = 0
                with open(index_path, "w", encoding="utf-8") as file:
                    for scenario_id in sorted(split_map):
                        if split_map[scenario_id] != split:
                            continue
                        metadata = metadata_by_id[scenario_id]
                        for branch in branches:
                            action_count = metadata["branches"][branch]["action_count"]
                            for start in range(
                                max(0, action_count - self.config.window_length + 2)
                            ):
                                entry = {
                                    "scenario_id": scenario_id,
                                    "pair_id": metadata["pair_id"],
                                    "split": split,
                                    "branch": branch,
                                    "start": start,
                                    "end": start + self.config.window_length,
                                }
                                file.write(json.dumps(entry, sort_keys=True) + "\n")
                                count += 1
                index_manifest[variant][split] = {
                    "path": str(index_path.relative_to(self.output_dir)),
                    "count": count,
                    "branches": list(branches),
                }
        return index_manifest

    def generate(self):
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty dataset directory {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scenarios").mkdir()

        split_map = _split_scenarios(
            self.config.num_scenarios,
            self.config.split_ratios,
            self.config.seed,
        )
        env = PushTEnv(with_velocity=True, with_target=True, render_size=self.config.render_size)
        scenario_metadata = []
        try:
            for scenario_index in range(self.config.num_scenarios):
                scenario_id = f"scenario_{scenario_index:06d}"
                metadata = self._generate_scenario(
                    env, scenario_index, split_map[scenario_id]
                )
                scenario_metadata.append(metadata)
                print(
                    f"[{scenario_index + 1}/{self.config.num_scenarios}] "
                    f"{scenario_id}: S={metadata['branches']['S']['final_coverage']:.3f} "
                    f"F1={metadata['branches']['F1']['final_coverage']:.3f} "
                    f"R={metadata['branches']['R']['final_coverage']:.3f}",
                    flush=True,
                )
        finally:
            env.close()

        windows = self._build_window_indices(scenario_metadata, split_map)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "config": asdict(self.config),
            "action_convention": {
                "kind": "bounded_relative_agent_displacement",
                "command_low": [-1.0, -1.0],
                "command_high": [1.0, 1.0],
                "simulator_pixels_per_command_unit": 100.0,
            },
            "oracle_scope": "goal-aligned translation pilot",
            "sim_state_fields": list(SIM_STATE_FIELDS),
            "branch_semantics": {
                "S": "closed-loop oracle nominal success trajectory",
                "F1": "post-perturbation open-loop nominal continuation",
                "F2": "post-perturbation equal-horizon neutral continuation",
                "R": "post-perturbation closed-loop oracle replanning",
            },
            "perturbation_levels": PERTURBATION_LEVELS,
            "split_by_scenario_before_windows": True,
            "scenario_splits": split_map,
            "variants": {key: list(value) for key, value in VARIANTS.items()},
            "window_indices": windows,
            "scenario_count": len(scenario_metadata),
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        audit = audit_dataset(
            self.output_dir,
            verify_videos=self.config.verify_videos,
            bootstrap_samples=self.config.bootstrap_samples,
        )
        manifest["audit_path"] = "audit.json"
        manifest["gate_a_pass"] = audit["gate_a_pass"]
        _write_json(self.output_dir / "manifest.json", manifest)
        return manifest, audit


def _video_frame_count(path):
    reader = imageio.get_reader(path)
    try:
        count = reader.count_frames()
        if not np.isfinite(count):
            count = sum(1 for _ in reader)
        return int(count)
    finally:
        reader.close()


def _paired_statistics(values, bootstrap_samples, seed=0):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) and bootstrap_samples:
        rng = np.random.default_rng(seed)
        samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
        confidence_interval = np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()
    else:
        confidence_interval = [mean, mean]
    if abs(mean) < 1e-12 or std == 0:
        estimated_n = None if abs(mean) < 1e-12 else 1
    else:
        estimated_n = int(math.ceil(((1.96 + 0.8416) * std / abs(mean)) ** 2))
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "bootstrap_95_ci": confidence_interval,
        "estimated_pairs_for_80_percent_power_at_observed_effect": estimated_n,
    }


def audit_dataset(dataset_dir, verify_videos=True, bootstrap_samples=5000):
    dataset_dir = Path(dataset_dir)
    manifest = _read_json(dataset_dir / "manifest.json")
    errors = []
    warnings = []
    scenario_metrics = []
    perturbation_types = Counter()
    severities = Counter()
    split_counts = Counter()
    action_violations = 0
    temporal_violations = 0
    branch_state_max_error = 0.0

    scenario_dirs = sorted((dataset_dir / "scenarios").glob("scenario_*"))
    if len(scenario_dirs) != manifest["scenario_count"]:
        errors.append(
            f"Scenario directory count {len(scenario_dirs)} != manifest "
            f"count {manifest['scenario_count']}"
        )

    seen_pairs = set()
    for scenario_dir in scenario_dirs:
        metadata = _read_json(scenario_dir / "metadata.json")
        scenario_id = metadata["scenario_id"]
        pair_id = metadata["pair_id"]
        if pair_id in seen_pairs:
            errors.append(f"Duplicate pair_id {pair_id}")
        seen_pairs.add(pair_id)
        expected_split = manifest["scenario_splits"].get(scenario_id)
        if metadata["split"] != expected_split:
            errors.append(f"Split mismatch for {scenario_id}")
        split_counts[metadata["split"]] += 1
        perturbation_types[metadata["perturbation"]["type"]] += 1
        severities[metadata["perturbation"]["severity"]] += 1

        records = {}
        for branch in BRANCHES:
            npz_path = scenario_dir / f"{branch}.npz"
            if not npz_path.exists():
                errors.append(f"Missing {npz_path}")
                continue
            with np.load(npz_path) as loaded:
                records[branch] = {key: loaded[key] for key in loaded.files}
            record = records[branch]
            action_count = len(record["actions"])
            observation_count = len(record["oracle_state"])
            aligned_keys = (
                "proprio", "legacy_state", "sim_state", "coverage", "success"
            )
            if observation_count != action_count + 1 or any(
                len(record[key]) != observation_count for key in aligned_keys
            ) or len(record["contacts"]) != action_count:
                temporal_violations += 1
                errors.append(f"Temporal alignment failure in {scenario_id}/{branch}")
            if not all(np.all(np.isfinite(record[key])) for key in (
                "actions", "proprio", "oracle_state", "legacy_state", "sim_state", "coverage"
            )):
                errors.append(f"Non-finite value in {scenario_id}/{branch}")
            action_violations += int(
                np.count_nonzero((record["actions"] < -1.000001) | (record["actions"] > 1.000001))
            )

            video_name = metadata["branches"][branch]["video"]
            if verify_videos and video_name is not None:
                frame_count = _video_frame_count(scenario_dir / video_name)
                if frame_count != observation_count:
                    errors.append(
                        f"Video alignment failure in {scenario_id}/{branch}: "
                        f"{frame_count} != {observation_count}"
                    )

        if not all(branch in records for branch in BRANCHES):
            continue
        branch_initials = np.stack(
            [records[branch]["sim_state"][0] for branch in ("F1", "F2", "R")]
        )
        branch_error = float(np.max(np.abs(branch_initials - branch_initials[0])))
        branch_state_max_error = max(branch_state_max_error, branch_error)
        if branch_error != 0:
            errors.append(f"Counterfactual branch mismatch in {scenario_id}")
        if not (
            len(records["F1"]["actions"])
            == len(records["F2"]["actions"])
            == len(records["R"]["actions"])
        ):
            errors.append(f"Unequal counterfactual budget in {scenario_id}")

        scenario_metrics.append(
            {
                "scenario_id": scenario_id,
                "S_success": bool(records["S"]["success"][-1]),
                "F1_success": bool(records["F1"]["success"][-1]),
                "F2_success": bool(records["F2"]["success"][-1]),
                "R_success": bool(records["R"]["success"][-1]),
                "F1_coverage": float(records["F1"]["coverage"][-1]),
                "F2_coverage": float(records["F2"]["coverage"][-1]),
                "R_coverage": float(records["R"]["coverage"][-1]),
                "R_recovery_steps": metadata["branches"]["R"]["first_success_step"],
            }
        )

    if action_violations:
        errors.append(f"Found {action_violations} out-of-bounds action components")
    if max(perturbation_types.values(), default=0) - min(
        perturbation_types.values(), default=0
    ) > 1:
        errors.append("Perturbation type counts are not balanced")
    if max(severities.values(), default=0) - min(severities.values(), default=0) > 1:
        errors.append("Severity counts are not balanced")

    for variant, split_entries in manifest["window_indices"].items():
        for split, entry in split_entries.items():
            count = 0
            with open(dataset_dir / entry["path"], "r", encoding="utf-8") as file:
                for line in file:
                    window = json.loads(line)
                    count += 1
                    if window["split"] != split:
                        errors.append(f"Window split field mismatch in {entry['path']}")
                    if manifest["scenario_splits"].get(window["scenario_id"]) != split:
                        errors.append(f"Scenario leakage in {entry['path']}")
                    if window["branch"] not in manifest["variants"][variant]:
                        errors.append(f"Unexpected branch in {entry['path']}")
            if count != entry["count"]:
                errors.append(f"Window count mismatch in {entry['path']}")

    if not scenario_metrics:
        errors.append("No complete scenarios were available for metrics")
        rates = {key: 0.0 for key in ("S", "F1", "F2", "R")}
        coverage_effect = success_effect = _paired_statistics([], 0)
    else:
        rates = {
            branch: float(np.mean([row[f"{branch}_success"] for row in scenario_metrics]))
            for branch in ("S", "F1", "F2", "R")
        }
        coverage_effect = _paired_statistics(
            [row["R_coverage"] - row["F1_coverage"] for row in scenario_metrics],
            bootstrap_samples,
            seed=int(manifest["config"]["seed"]),
        )
        success_effect = _paired_statistics(
            [float(row["R_success"]) - float(row["F1_success"]) for row in scenario_metrics],
            bootstrap_samples,
            seed=int(manifest["config"]["seed"]) + 1,
        )
        if rates["S"] < 0.95:
            errors.append(f"Nominal oracle success rate is too low: {rates['S']:.3f}")
        if coverage_effect["mean"] <= 0:
            warnings.append("Recovery did not improve mean final coverage over F1")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenario_metrics),
        "split_counts": dict(split_counts),
        "perturbation_type_counts": dict(perturbation_types),
        "severity_counts": dict(severities),
        "branch_final_success_rates": rates,
        "branch_state_max_abs_error": branch_state_max_error,
        "action_bound_violations": action_violations,
        "temporal_alignment_violations": temporal_violations,
        "paired_recovery_minus_f1_final_coverage": coverage_effect,
        "paired_recovery_minus_f1_success": success_effect,
        "errors": errors,
        "warnings": warnings,
        "gate_a_pass": not errors,
    }
    _write_json(dataset_dir / "audit.json", audit)
    return audit
