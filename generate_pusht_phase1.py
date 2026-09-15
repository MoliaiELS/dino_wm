"""Command-line entry point for the Phase 1 PushT paired-data pilot."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from phase1.pusht_dataset import Phase1Config, PushTPhase1Generator, audit_dataset


def default_output_dir():
    dataset_root = os.environ.get("DATASET_DIR")
    if not dataset_root:
        raise RuntimeError("DATASET_DIR is not set; source bash.sh before running")
    return Path(dataset_root) / "pusht_recovery_phase1_pilot_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--nominal-horizon", type=int, default=50)
    parser.add_argument("--branch-horizon", type=int, default=35)
    parser.add_argument("--window-length", type=int, default=21)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Assign every generated scenario to test for a locked confirmatory set",
    )
    parser.add_argument("--skip-video-verification", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = default_output_dir() if args.output_dir is None else args.output_dir
    if args.audit_only:
        audit = audit_dataset(
            output_dir, verify_videos=not args.skip_video_verification
        )
    else:
        config = replace(
            Phase1Config(),
            num_scenarios=args.num_scenarios,
            seed=args.seed,
            render_size=args.render_size,
            nominal_horizon=args.nominal_horizon,
            branch_horizon=args.branch_horizon,
            window_length=args.window_length,
            split_ratios=(
                {"train": 0.0, "valid": 0.0, "test": 1.0}
                if args.all_test else Phase1Config().split_ratios
            ),
            save_videos=not args.no_videos,
            verify_videos=not args.skip_video_verification,
        )
        _, audit = PushTPhase1Generator(output_dir, config).generate()
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["gate_a_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
