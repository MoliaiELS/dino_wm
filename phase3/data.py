"""Leakage-safe cached-DINO datasets for Experiment B."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_CACHE_SCHEMA = "pusht-recovery-dino-cache-v1"
OFF_NOMINAL_PHASES = {
    "reposition",
    "recontact",
    "open_loop_failure",
    "neutral",
}
NOMINAL_PHASES = {"nominal_push", "hold"}
RECOVERABILITY_BRANCHES = {"S", "N", "R"}


def _read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_cache_manifest(cache_dir):
    manifest = _read_json(Path(cache_dir) / "manifest.json")
    if manifest.get("cache_schema") != SUPPORTED_CACHE_SCHEMA:
        raise ValueError(f"Unsupported visual cache schema: {manifest.get('cache_schema')}")
    if not manifest.get("source_gate_a_pass", False):
        raise ValueError("Visual cache source dataset did not pass Gate A")
    if not manifest.get("split_by_scenario_before_windows", False):
        raise ValueError("Visual cache does not preserve scenario-disjoint splits")
    return manifest


@dataclass(frozen=True)
class ProprioStats:
    mean: np.ndarray
    std: np.ndarray
    count: int
    variant: str

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("PushT proprio statistics must have four dimensions")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Proprio statistics must be finite")
        if np.any(std <= 0):
            raise ValueError("Proprio standard deviations must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def normalize(self, values):
        return (values - self.mean) / self.std

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": int(self.count),
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            payload["mean"], payload["std"], payload["count"], payload["variant"]
        )


def _trajectory_paths(cache_dir, manifest, variant, split):
    if variant not in manifest["variants"]:
        raise ValueError(f"Unknown variant {variant}")
    for scenario_id, assigned_split in sorted(manifest["scenario_splits"].items()):
        if assigned_split != split:
            continue
        for branch in manifest["variants"][variant]:
            yield Path(cache_dir) / "scenarios" / scenario_id / f"{branch}.npz"


def compute_proprio_stats(cache_dir, variant, min_std=1.0):
    cache_dir = Path(cache_dir)
    manifest = load_cache_manifest(cache_dir)
    count = 0
    total = np.zeros(4, dtype=np.float64)
    squared_total = np.zeros(4, dtype=np.float64)
    for path in _trajectory_paths(cache_dir, manifest, variant, "train"):
        with np.load(path) as record:
            values = np.asarray(record["proprio"], dtype=np.float64)
        count += len(values)
        total += values.sum(axis=0)
        squared_total += np.square(values).sum(axis=0)
    if count == 0:
        raise ValueError(f"No train proprio frames for {variant}")
    mean = total / count
    variance = np.maximum(squared_total / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), float(min_std))
    return ProprioStats(mean, std, count, variant)


def _stratified_limit(entries, maximum: Optional[int], seed: int):
    if maximum is None or maximum >= len(entries):
        return entries
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["branch"]].append(entry)
    rng = np.random.default_rng(seed)
    for values in grouped.values():
        rng.shuffle(values)
    result = []
    while len(result) < maximum:
        advanced = False
        for branch in sorted(grouped):
            if grouped[branch] and len(result) < maximum:
                result.append(grouped[branch].pop())
                advanced = True
        if not advanced:
            break
    rng.shuffle(result)
    return result


class _CachedTrajectoryMixin:
    def __init__(self, cache_dir, stats):
        self.cache_dir = Path(cache_dir)
        self.manifest = load_cache_manifest(cache_dir)
        self.stats = stats
        self.token_count = int(self.manifest["pooled_patch_count"])
        self.token_dim = int(self.manifest["embedding_dim"])
        self.proprio_dim = 4
        self.action_dim = 2

    @lru_cache(maxsize=512)
    def _load_branch(self, scenario_id, branch):
        path = self.cache_dir / "scenarios" / scenario_id / f"{branch}.npz"
        with np.load(path) as record:
            return {key: np.asarray(record[key]).copy() for key in record.files}

    def _tensor_frame_data(self, record, frame_indices):
        tokens = np.asarray(record["tokens"][frame_indices], dtype=np.float32)
        proprio = self.stats.normalize(record["proprio"][frame_indices]).astype(
            np.float32
        )
        return torch.from_numpy(tokens), torch.from_numpy(proprio)


class VisualFeatureWindowDataset(_CachedTrajectoryMixin, Dataset):
    """Teacher-forced visual dynamics windows with T actions and T+1 frames."""

    def __init__(
        self,
        cache_dir,
        variant,
        split,
        window_actions=3,
        stats: Optional[ProprioStats] = None,
        max_windows=None,
        seed=0,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {split}")
        if window_actions < 1:
            raise ValueError("window_actions must be positive")
        manifest = load_cache_manifest(cache_dir)
        if stats is None:
            if split != "train":
                raise ValueError("Validation/test visual data require train statistics")
            stats = compute_proprio_stats(cache_dir, variant)
        super().__init__(cache_dir, stats)
        self.variant = variant
        self.split = split
        self.window_actions = int(window_actions)
        entries = []
        for scenario_id, assigned_split in sorted(self.manifest["scenario_splits"].items()):
            if assigned_split != split:
                continue
            for branch in self.manifest["variants"][variant]:
                observations = int(
                    self.manifest["trajectory_lengths"][scenario_id][branch][
                        "observations"
                    ]
                )
                for start in range(observations - window_actions):
                    entries.append(
                        {"scenario_id": scenario_id, "branch": branch, "start": start}
                    )
        self.entries = _stratified_limit(entries, max_windows, seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        record = self._load_branch(entry["scenario_id"], entry["branch"])
        start = entry["start"]
        stop = start + self.window_actions + 1
        tokens, proprio = self._tensor_frame_data(record, slice(start, stop))
        actions = torch.from_numpy(
            np.asarray(record["actions"][start : stop - 1], dtype=np.float32)
        )
        return {
            "tokens": tokens,
            "proprio": proprio,
            "actions": actions,
            "scenario_id": entry["scenario_id"],
            "branch": entry["branch"],
            "start": start,
        }

    @property
    def branch_counts(self):
        return dict(Counter(entry["branch"] for entry in self.entries))


class VisualProbeFrameDataset(_CachedTrajectoryMixin, Dataset):
    """Causal history samples and oracle-only labels for frozen probes."""

    def __init__(
        self,
        cache_dir,
        split,
        stats,
        history=3,
        recoverability_horizon=10,
        branches=("S", "N", "F1", "F2", "R"),
        max_samples=None,
        seed=0,
    ):
        if history < 1 or recoverability_horizon < 1:
            raise ValueError("history and recoverability horizon must be positive")
        super().__init__(cache_dir, stats)
        self.split = split
        self.history = int(history)
        self.recoverability_horizon = int(recoverability_horizon)
        self.entries = []
        for scenario_id, assigned_split in sorted(self.manifest["scenario_splits"].items()):
            if assigned_split != split:
                continue
            for branch in branches:
                observations = int(
                    self.manifest["trajectory_lengths"][scenario_id][branch][
                        "observations"
                    ]
                )
                for frame in range(history - 1, observations):
                    self.entries.append(
                        {"scenario_id": scenario_id, "branch": branch, "frame": frame}
                    )
        self.entries = _stratified_limit(self.entries, max_samples, seed)

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _phase_at(record, frame):
        phases = record["phase"]
        if len(phases) == 0:
            return ""
        value = phases[min(frame, len(phases) - 1)]
        return str(value.item() if getattr(value, "ndim", 0) == 0 else value)

    def __getitem__(self, index):
        entry = self.entries[index]
        record = self._load_branch(entry["scenario_id"], entry["branch"])
        frame = entry["frame"]
        start = frame - self.history + 1
        tokens, proprio = self._tensor_frame_data(record, slice(start, frame + 1))
        actions = np.asarray(record["actions"][start:frame], dtype=np.float32)
        phase = self._phase_at(record, frame)
        if phase in OFF_NOMINAL_PHASES:
            off_nominal, off_mask = 1, True
        elif phase in NOMINAL_PHASES:
            off_nominal, off_mask = 0, True
        else:
            off_nominal, off_mask = 0, False
        recoverability_mask = entry["branch"] in RECOVERABILITY_BRANCHES
        horizon_stop = min(len(record["success"]), frame + self.recoverability_horizon + 1)
        recoverable = bool(np.any(record["success"][frame:horizon_stop]))
        return {
            "tokens": tokens,
            "proprio": proprio,
            "history_actions": torch.from_numpy(actions),
            "oracle_state": torch.from_numpy(
                np.asarray(record["oracle_state"][frame], dtype=np.float32)
            ),
            "coverage": float(record["coverage"][frame]),
            "off_nominal": int(off_nominal),
            "off_nominal_mask": bool(off_mask),
            "recoverable": int(recoverable),
            "recoverability_mask": bool(recoverability_mask),
            "scenario_id": entry["scenario_id"],
            "branch": entry["branch"],
            "frame": frame,
            "phase": phase,
        }


def branch_balanced_weights(entries):
    counts = Counter(entry["branch"] for entry in entries)
    if not counts:
        raise ValueError("Cannot weight an empty visual dataset")
    return torch.as_tensor(
        [1.0 / counts[entry["branch"]] for entry in entries], dtype=torch.double
    )
