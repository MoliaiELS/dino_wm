"""Evaluate a Phase 2 StateWorldModel checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from models.state_world_model import StateWorldModel
from phase2.data import NormalizationStats
from phase2.evaluation import (
    dump_json,
    evaluate_closed_loop_recovery,
    evaluate_counterfactual_ranking,
    evaluate_prediction_horizons,
)


def _default_dataset_dir():
    root = os.environ.get("DATASET_DIR")
    return None if root is None else str(Path(root) / "pusht_recovery_phase1_pilot_v1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default=_default_dataset_dir())
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--prediction-stride", type=int, default=1)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--cem-horizon", type=int, default=12)
    parser.add_argument("--cem-samples", type=int, default=256)
    parser.add_argument("--cem-topk", type=int, default=32)
    parser.add_argument("--cem-iterations", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_dir is None:
        raise ValueError("Set DATASET_DIR or pass --dataset-dir")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = StateWorldModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "training_variant": checkpoint["variant"],
        "training_seed": checkpoint["seed"],
        "training_epoch": checkpoint["epoch"],
        "git_commit": checkpoint["git_commit"],
    }
    if not args.skip_prediction:
        results["prediction"] = evaluate_prediction_horizons(
            model,
            args.dataset_dir,
            stats,
            device,
            batch_size=args.batch_size,
            stride=args.prediction_stride,
            max_scenarios=args.max_scenarios,
        )
    if not args.skip_ranking:
        results["counterfactual_ranking"] = evaluate_counterfactual_ranking(
            model,
            args.dataset_dir,
            stats,
            device,
            max_scenarios=args.max_scenarios,
        )
    if args.closed_loop:
        output_path = Path(args.output)
        results["closed_loop_recovery"] = evaluate_closed_loop_recovery(
            model,
            args.dataset_dir,
            stats,
            device,
            output_dir=output_path.parent / f"{output_path.stem}_videos",
            max_scenarios=args.max_scenarios,
            save_videos=args.save_videos,
            planner_kwargs={
                "horizon": args.cem_horizon,
                "num_samples": args.cem_samples,
                "topk": args.cem_topk,
                "iterations": args.cem_iterations,
            },
        )
    dump_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
