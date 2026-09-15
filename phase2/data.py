"""Leakage-safe loading for the Phase 1 paired PushT state dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_SCHEMAS = {
    "pusht-recovery-pairs-v1",
    "pusht-recovery-pairs-v2",
}
ORACLE_STATE_DIM = 11
ACTION_DIM = 2
# The controlled translation pilot has nearly constant angle dimensions in
# S/F1/F2.  Never divide those physical coordinates by numerical noise.  A
# unit floor also makes a shared normalizer safe when R contains small contact
# rotations that are absent from the common S+F1 core.
DEFAULT_STATE_STD_FLOOR = np.ones(ORACLE_STATE_DIM, dtype=np.float64)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


@dataclass(frozen=True)
class NormalizationStats:
    """State statistics computed from complete training trajectories only."""

    state_mean: np.ndarray
    state_std: np.ndarray
    count: int
    variant: str

    def __post_init__(self):
        mean = np.asarray(self.state_mean, dtype=np.float32)
        std = np.asarray(self.state_std, dtype=np.float32)
        if mean.shape != (ORACLE_STATE_DIM,) or std.shape != (ORACLE_STATE_DIM,):
            raise ValueError("Oracle-state normalization must have 11 dimensions")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Normalization statistics must be finite")
        if np.any(std <= 0):
            raise ValueError("Normalization standard deviations must be positive")
        object.__setattr__(self, "state_mean", mean)
        object.__setattr__(self, "state_std", std)

    def normalize(self, states):
        return (states - self.state_mean) / self.state_std

    def denormalize(self, states):
        return states * self.state_std + self.state_mean

    def to_dict(self):
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "count": int(self.count),
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            state_mean=np.asarray(payload["state_mean"], dtype=np.float32),
            state_std=np.asarray(payload["state_std"], dtype=np.float32),
            count=int(payload["count"]),
            variant=str(payload["variant"]),
        )


def load_manifest(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest = _read_json(dataset_dir / "manifest.json")
    if manifest.get("schema_version") not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"Expected one of {sorted(SUPPORTED_SCHEMAS)}, "
            f"got {manifest.get('schema_version')}"
        )
    if not manifest.get("gate_a_pass", False):
        raise ValueError("Refusing to train from a dataset that has not passed Gate A")
    if not manifest.get("split_by_scenario_before_windows", False):
        raise ValueError("Dataset does not guarantee scenario-level splitting")
    return manifest


def _trajectory_paths(dataset_dir: Path, manifest: dict, variant: str, split: str):
    if variant not in manifest["variants"]:
        raise ValueError(f"Unknown dataset variant {variant}")
    branches = tuple(manifest["variants"][variant])
    for scenario_id, assigned_split in sorted(manifest["scenario_splits"].items()):
        if assigned_split != split:
            continue
        for branch in branches:
            yield dataset_dir / "scenarios" / scenario_id / f"{branch}.npz"


def compute_train_normalization(dataset_dir, variant, min_std=None):
    """Compute statistics once per physical frame, never from valid/test data."""

    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    count = 0
    total = np.zeros(ORACLE_STATE_DIM, dtype=np.float64)
    squared_total = np.zeros(ORACLE_STATE_DIM, dtype=np.float64)
    for path in _trajectory_paths(dataset_dir, manifest, variant, "train"):
        with np.load(path) as record:
            states = np.asarray(record["oracle_state"], dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != ORACLE_STATE_DIM:
            raise ValueError(f"Unexpected oracle-state shape {states.shape} in {path}")
        if not np.all(np.isfinite(states)):
            raise ValueError(f"Non-finite oracle state in {path}")
        count += len(states)
        total += states.sum(axis=0)
        squared_total += np.square(states).sum(axis=0)
    if count == 0:
        raise ValueError(f"No training states found for {variant}")
    mean = total / count
    variance = np.maximum(squared_total / count - np.square(mean), 0.0)
    floor = (
        DEFAULT_STATE_STD_FLOOR
        if min_std is None
        else np.broadcast_to(np.asarray(min_std, dtype=np.float64), (ORACLE_STATE_DIM,))
    )
    if np.any(floor <= 0):
        raise ValueError("Normalization standard-deviation floors must be positive")
    std = np.maximum(np.sqrt(variance), floor)
    return NormalizationStats(mean, std, count, variant)


def _stratified_limit(entries, max_windows: Optional[int], seed: int):
    if max_windows is None or max_windows >= len(entries):
        return entries
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["branch"]].append(entry)
    rng = np.random.default_rng(seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected = []
    branch_names = sorted(grouped)
    while len(selected) < max_windows:
        made_progress = False
        for branch in branch_names:
            if grouped[branch] and len(selected) < max_windows:
                selected.append(grouped[branch].pop())
                made_progress = True
        if not made_progress:
            break
    rng.shuffle(selected)
    return selected


class PairedStateWindowDataset(Dataset):
    """State/action windows named by the immutable Phase 1 index files."""

    def __init__(
        self,
        dataset_dir,
        variant,
        split,
        stats: Optional[NormalizationStats] = None,
        max_windows: Optional[int] = None,
        seed: int = 0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.manifest = load_manifest(self.dataset_dir)
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {split}")
        if variant not in self.manifest["window_indices"]:
            raise ValueError(f"No window index for variant {variant}")
        index_metadata = self.manifest["window_indices"][variant][split]
        self.entries = _read_jsonl(self.dataset_dir / index_metadata["path"])
        if len(self.entries) != int(index_metadata["count"]):
            raise ValueError("Window index count disagrees with the manifest")
        expected_branches = set(self.manifest["variants"][variant])
        for entry in self.entries:
            if entry["split"] != split:
                raise ValueError("Window split disagrees with its index file")
            if entry["branch"] not in expected_branches:
                raise ValueError("Window branch is not part of the requested variant")
            if self.manifest["scenario_splits"].get(entry["scenario_id"]) != split:
                raise ValueError("Scenario split disagrees with window split")
        self.entries = _stratified_limit(self.entries, max_windows, seed)
        if stats is None:
            if split != "train":
                raise ValueError("Validation/test datasets require training statistics")
            stats = compute_train_normalization(self.dataset_dir, variant)
        self.stats = stats
        self.variant = variant
        self.split = split
        self.state_dim = ORACLE_STATE_DIM
        self.action_dim = ACTION_DIM
        action_convention = self.manifest["action_convention"]
        self.action_low = np.asarray(action_convention["command_low"], dtype=np.float32)
        self.action_high = np.asarray(action_convention["command_high"], dtype=np.float32)

    @lru_cache(maxsize=1024)
    def _load_branch(self, scenario_id: str, branch: str):
        path = self.dataset_dir / "scenarios" / scenario_id / f"{branch}.npz"
        with np.load(path) as record:
            states = np.asarray(record["oracle_state"], dtype=np.float32).copy()
            actions = np.asarray(record["actions"], dtype=np.float32).copy()
            coverage = np.asarray(record["coverage"], dtype=np.float32).copy()
        return states, actions, coverage

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        states, actions, coverage = self._load_branch(
            entry["scenario_id"], entry["branch"]
        )
        start, end = int(entry["start"]), int(entry["end"])
        state_window = states[start:end]
        action_window = actions[start : end - 1]
        coverage_window = coverage[start:end]
        if len(state_window) != end - start or len(action_window) != end - start - 1:
            raise IndexError(f"Window exceeds trajectory bounds: {entry}")
        if state_window.shape[1:] != (ORACLE_STATE_DIM,):
            raise ValueError(f"Unexpected state window shape {state_window.shape}")
        if action_window.shape[1:] != (ACTION_DIM,):
            raise ValueError(f"Unexpected action window shape {action_window.shape}")
        if np.any(action_window < self.action_low) or np.any(action_window > self.action_high):
            raise ValueError(f"Out-of-bounds action in {entry}")
        normalized_states = self.stats.normalize(state_window).astype(np.float32)
        return {
            "states": torch.from_numpy(normalized_states),
            # Command-space actions already use the shared physical [-1, 1] bounds.
            "actions": torch.from_numpy(action_window),
            "coverage": torch.from_numpy(coverage_window),
            "scenario_id": entry["scenario_id"],
            "pair_id": entry["pair_id"],
            "branch": entry["branch"],
            "start": start,
        }

    @property
    def branch_counts(self) -> Dict[str, int]:
        return dict(Counter(entry["branch"] for entry in self.entries))


def branch_balanced_weights(entries: Iterable[dict]):
    """Return inverse-frequency weights so every included branch is sampled equally."""

    entries = list(entries)
    counts = Counter(entry["branch"] for entry in entries)
    if not counts:
        raise ValueError("Cannot weight an empty index")
    return torch.as_tensor(
        [1.0 / counts[entry["branch"]] for entry in entries], dtype=torch.double
    )
