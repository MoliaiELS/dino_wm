import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Any, Callable, Mapping, Optional

from env.pusht.action_utils import (
    DEFAULT_ACTION_SCALE,
    DEFAULT_RELATIVE_ACTION_LIMIT,
    DEFAULT_WORKSPACE_SIZE,
    get_action_bounds,
)
from .traj_dset import TrajDataset, TrajSlicerDataset
decord.bridge.set_bridge("torch")


def compute_valid_stats(values, seq_lengths, min_std=1e-6):
    """Compute feature statistics without including padded trajectory frames."""
    valid_chunks = []
    for rollout_idx, seq_length in enumerate(seq_lengths):
        seq_length = int(seq_length)
        if seq_length < 0 or seq_length > values.shape[1]:
            raise ValueError(
                f"Invalid sequence length {seq_length} for padded length "
                f"{values.shape[1]} at rollout {rollout_idx}"
            )
        if seq_length:
            valid_chunks.append(values[rollout_idx, :seq_length])

    if not valid_chunks:
        raise ValueError("Cannot compute normalization statistics without valid frames")

    valid_values = torch.cat(valid_chunks, dim=0).float()
    mean = valid_values.mean(dim=0)
    std = valid_values.std(dim=0, unbiased=False)
    std = torch.where(std < min_std, torch.ones_like(std), std)
    return mean, std

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=DEFAULT_ACTION_SCALE,
        relative_action_limit=DEFAULT_RELATIVE_ACTION_LIMIT,
        workspace_size=DEFAULT_WORKSPACE_SIZE,
        with_velocity: bool = True, # agent's velocity
        normalization_stats: Optional[Mapping[str, Any]] = None,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.action_scale = float(action_scale)
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / self.action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.shapes = self.shapes[:n]
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        action_low, action_high = get_action_bounds(
            relative=self.relative,
            action_scale=self.action_scale,
            relative_action_limit=relative_action_limit,
            workspace_size=workspace_size,
        )
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32)

        if normalize_action:
            if normalization_stats is None:
                self.action_mean, self.action_std = compute_valid_stats(
                    self.actions, self.seq_lengths
                )
                self.state_mean, self.state_std = compute_valid_stats(
                    self.states, self.seq_lengths
                )
                self.proprio_mean, self.proprio_std = compute_valid_stats(
                    self.proprios, self.seq_lengths
                )
            else:
                self.action_mean, self.action_std = self._read_stats(
                    normalization_stats, "action", self.action_dim
                )
                self.state_mean, self.state_std = self._read_stats(
                    normalization_stats, "state", self.state_dim
                )
                self.proprio_mean, self.proprio_std = self._read_stats(
                    normalization_stats, "proprio", self.proprio_dim
                )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.normalization_stats = {
            "action_mean": self.action_mean.clone(),
            "action_std": self.action_std.clone(),
            "state_mean": self.state_mean.clone(),
            "state_std": self.state_std.clone(),
            "proprio_mean": self.proprio_mean.clone(),
            "proprio_std": self.proprio_std.clone(),
        }

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    @staticmethod
    def _read_stats(stats, prefix, expected_dim):
        mean = torch.as_tensor(stats[f"{prefix}_mean"], dtype=torch.float32).clone()
        std = torch.as_tensor(stats[f"{prefix}_std"], dtype=torch.float32).clone()
        if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
            raise ValueError(
                f"{prefix} statistics must have shape ({expected_dim},), got "
                f"mean={tuple(mean.shape)} and std={tuple(std.shape)}"
            )
        if torch.any(std <= 0):
            raise ValueError(f"{prefix} standard deviation must be positive")
        return mean, std

    def get_normalization_stats(self):
        """Return a defensive copy suitable for validation/test datasets."""
        return {key: value.clone() for key, value in self.normalization_stats.items()}

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {'shape': shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        normalization_stats=train_dset.get_normalization_stats(),
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset
