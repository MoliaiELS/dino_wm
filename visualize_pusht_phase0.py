"""Generate a deterministic PushT Phase 0 visual smoke-test artifact."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v2 as imageio
import numpy as np

from env.pusht.pusht_env import PushTEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plan_outputs/phase0_pusht_smoke"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def replay_branch(env_instance, snapshot, actions):
    env_instance.set_sim_state(snapshot)
    frames = []
    oracle_states = []
    coverages = []
    success = []
    for action in actions:
        observation, _, _, info = env_instance.step(action)
        frames.append(observation["visual"])
        oracle_states.append(env_instance.get_oracle_state())
        coverages.append(info["final_coverage"])
        success.append(info["success"])
    return np.stack(frames), np.stack(oracle_states), coverages, success


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_instance = PushTEnv(
        with_velocity=True,
        with_target=True,
        render_size=224,
        reset_to_state=np.array([256, 400, 256, 300, 0, 0, 0]),
    )
    try:
        env_instance.seed(args.seed)
        initial_observation, _ = env_instance.reset()
        warmup_frames = [initial_observation["visual"]]
        for action in ([0.0, -0.8], [0.0, -0.8], [0.1, -0.6]):
            observation, _, _, _ = env_instance.step(np.asarray(action))
            warmup_frames.append(observation["visual"])

        snapshot = env_instance.get_sim_state()
        branch_actions = np.asarray(
            [
                [0.2, -0.5],
                [-0.3, -0.4],
                [0.4, 0.1],
                [-0.2, 0.3],
                [0.0, -0.6],
                [0.3, -0.2],
            ],
            dtype=np.float64,
        )
        first = replay_branch(env_instance, snapshot, branch_actions)
        second = replay_branch(env_instance, snapshot, branch_actions)

        frames = np.concatenate([np.stack(warmup_frames), first[0]], axis=0)
        imageio.imwrite(args.output_dir / "initial.png", frames[0])
        imageio.imwrite(args.output_dir / "final.png", frames[-1])
        imageio.mimsave(args.output_dir / "rollout.mp4", frames, fps=args.fps)

        summary = {
            "seed": args.seed,
            "frame_count": int(len(frames)),
            "frame_shape": list(frames.shape[1:]),
            "action_convention": "relative command; target displacement = command * 100 pixels",
            "action_low": env_instance.action_space.low.tolist(),
            "action_high": env_instance.action_space.high.tolist(),
            "max_oracle_replay_error": float(np.max(np.abs(first[1] - second[1]))),
            "max_rgb_replay_error": int(np.max(np.abs(first[0].astype(int) - second[0].astype(int)))),
            "final_coverage": float(first[2][-1]),
            "success": bool(first[3][-1]),
        }
        with open(args.output_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        env_instance.close()


if __name__ == "__main__":
    main()
