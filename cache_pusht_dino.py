"""Render paired simulator states and cache frozen DINOv2 patch tokens."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env.pusht.pusht_env import PushTEnv
from models.dino import DinoV2Encoder
from phase2.data import load_manifest
from phase3.data import SUPPORTED_CACHE_SCHEMA


DINO_REVISION = "b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"


def _default_source():
    root = os.environ.get("DATASET_DIR")
    return None if root is None else str(Path(root) / "pusht_recovery_phase1_pilot_v2")


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _body_from_vector(vector):
    return {
        "position": vector[0:2],
        "velocity": vector[2:4],
        "angle": float(vector[4]),
        "angular_velocity": float(vector[5]),
        "force": vector[6:8],
        "torque": float(vector[8]),
        "center_of_gravity": vector[9:11],
    }


def render_sim_state(env, vector, shape="T"):
    """Render one serialized Phase-1 state without advancing physics."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (27,):
        raise ValueError(f"Expected a 27-D simulator vector, got {vector.shape}")
    if env.space is None or env.shape != shape:
        env.shape = shape
        env.reset()
    env.space.damping = float(vector[25])
    env.goal_pose = vector[22:25].copy()
    env.success_threshold = float(vector[26])
    env._restore_body(env.agent, _body_from_vector(vector[0:11]))
    env._restore_body(env.block, _body_from_vector(vector[11:22]))
    env.space.reindex_shapes_for_body(env.agent)
    env.space.reindex_shapes_for_body(env.block)
    return np.asarray(env.render("rgb_array"), dtype=np.uint8)


def _encode_frames(encoder, frames, grid_size, batch_size, device):
    outputs = []
    for start in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[start : start + batch_size]).to(device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = batch.sub_(0.5).div_(0.5)
        with torch.inference_mode():
            tokens = encoder(batch)
            side = int(round(tokens.shape[1] ** 0.5))
            if side * side != tokens.shape[1]:
                raise ValueError("DINO patch tokens do not form a square grid")
            tokens = tokens.reshape(
                tokens.shape[0], side, side, tokens.shape[-1]
            ).permute(0, 3, 1, 2)
            tokens = F.adaptive_avg_pool2d(tokens, (grid_size, grid_size))
            tokens = tokens.permute(0, 2, 3, 1).reshape(
                tokens.shape[0], grid_size * grid_size, tokens.shape[1]
            )
        outputs.append(tokens.cpu().to(torch.float16).numpy())
    return np.concatenate(outputs, axis=0)


def _atomic_savez(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as file:
        np.savez(file, **arrays)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=_default_source())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pooled-grid-size", type=int, default=4)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source_dir is None:
        raise ValueError("Set DATASET_DIR or pass --source-dir")
    if args.batch_size <= 0 or args.pooled_grid_size <= 0:
        raise ValueError("batch size and pooled grid size must be positive")
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    source_manifest = load_manifest(source_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").to(device)
    encoder.requires_grad_(False).eval()
    env = PushTEnv(with_velocity=True, with_target=True, render_size=224)

    scenario_ids = sorted(source_manifest["scenario_splits"])
    if args.max_scenarios is not None:
        scenario_ids = scenario_ids[: args.max_scenarios]
    lengths = {}
    total_frames = 0
    pixel_replay_max_error = 0
    token_abs_max = 0.0
    branches = sorted({b for values in source_manifest["variants"].values() for b in values})
    for scenario_index, scenario_id in enumerate(scenario_ids):
        source_scenario = source_dir / "scenarios" / scenario_id
        target_scenario = output_dir / "scenarios" / scenario_id
        target_scenario.mkdir(parents=True, exist_ok=True)
        metadata = json.load(open(source_scenario / "metadata.json", encoding="utf-8"))
        shape = metadata.get("shape", "T")
        lengths[scenario_id] = {}
        for branch in branches:
            source_path = source_scenario / f"{branch}.npz"
            target_path = target_scenario / f"{branch}.npz"
            with np.load(source_path) as record:
                arrays = {key: np.asarray(record[key]).copy() for key in record.files}
            observations = len(arrays["sim_state"])
            lengths[scenario_id][branch] = {
                "observations": observations,
                "actions": len(arrays["actions"]),
            }
            if target_path.exists() and args.resume:
                with np.load(target_path) as cached:
                    if len(cached["tokens"]) != observations:
                        raise ValueError(f"Resume cache length mismatch in {target_path}")
                    token_abs_max = max(token_abs_max, float(np.abs(cached["tokens"]).max()))
                total_frames += observations
                continue
            frames = np.stack(
                [render_sim_state(env, state, shape) for state in arrays["sim_state"]]
            )
            replay = render_sim_state(env, arrays["sim_state"][0], shape)
            pixel_replay_max_error = max(
                pixel_replay_max_error,
                int(np.abs(replay.astype(np.int16) - frames[0].astype(np.int16)).max()),
            )
            tokens = _encode_frames(
                encoder, frames, args.pooled_grid_size, args.batch_size, device
            )
            if not np.all(np.isfinite(tokens)):
                raise ValueError(f"Non-finite DINO tokens in {scenario_id}/{branch}")
            token_abs_max = max(token_abs_max, float(np.abs(tokens).max()))
            _atomic_savez(target_path, tokens=tokens, **arrays)
            total_frames += observations
        print(
            json.dumps(
                {
                    "scenario": scenario_id,
                    "completed": scenario_index + 1,
                    "total": len(scenario_ids),
                    "frames": total_frames,
                }
            ),
            flush=True,
        )

    selected = set(scenario_ids)
    manifest = {
        "cache_schema": SUPPORTED_CACHE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_dir": str(source_dir.resolve()),
        "source_schema": source_manifest["schema_version"],
        "source_gate_a_pass": bool(source_manifest["gate_a_pass"]),
        "split_by_scenario_before_windows": bool(
            source_manifest["split_by_scenario_before_windows"]
        ),
        "scenario_splits": {
            key: value
            for key, value in source_manifest["scenario_splits"].items()
            if key in selected
        },
        "variants": source_manifest["variants"],
        "trajectory_lengths": lengths,
        "dino_model": "dinov2_vits14",
        "dino_revision": DINO_REVISION,
        "dino_feature_key": "x_norm_patchtokens",
        "input_normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "render_size": 224,
        "original_patch_count": 256,
        "pooled_grid_size": args.pooled_grid_size,
        "pooled_patch_count": args.pooled_grid_size ** 2,
        "embedding_dim": int(encoder.emb_dim),
        "storage_dtype": "float16",
        "total_frames": total_frames,
        "audit": {
            "pixel_deterministic_replay_max_error": pixel_replay_max_error,
            "token_abs_max": token_abs_max,
            "finite_tokens": True,
        },
    }
    temporary = output_dir / "manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    temporary.replace(output_dir / "manifest.json")
    print(json.dumps({"status": "completed", **manifest["audit"]}), flush=True)


if __name__ == "__main__":
    main()
