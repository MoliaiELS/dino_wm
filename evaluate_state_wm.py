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
    evaluate_prediction_by_branch,
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
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--cem-horizon", type=int, default=12)
    parser.add_argument("--cem-samples", type=int, default=256)
    parser.add_argument("--cem-topk", type=int, default=32)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-action-repeat", type=int, default=3)
    parser.add_argument("--cem-initial-std", type=float, default=0.5)
    parser.add_argument("--cem-action-cost", type=float, default=0.02)
    parser.add_argument("--cem-smoothness-cost", type=float, default=0.01)
    parser.add_argument("--cem-staging-weight", type=float, default=0.25)
    parser.add_argument("--cem-staging-distance", type=float, default=139.0)
    parser.add_argument("--cem-max-action-norm", type=float)
    parser.add_argument("--cem-trajectory-cost-weight", type=float, default=0.0)
    parser.add_argument("--cem-progress-regression-weight", type=float, default=0.0)
    parser.add_argument("--cem-object-speed-weight", type=float, default=0.0)
    parser.add_argument("--cem-min-predicted-improvement", type=float)
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
        "evaluation_config": {
            "device": str(device),
            "batch_size": args.batch_size,
            "prediction_stride": args.prediction_stride,
            "max_scenarios": args.max_scenarios,
            "split": args.split,
            "cem_horizon": args.cem_horizon,
            "cem_samples": args.cem_samples,
            "cem_topk": args.cem_topk,
            "cem_iterations": args.cem_iterations,
            "cem_action_repeat": args.cem_action_repeat,
            "cem_initial_std": args.cem_initial_std,
            "cem_action_cost": args.cem_action_cost,
            "cem_smoothness_cost": args.cem_smoothness_cost,
            "cem_staging_weight": args.cem_staging_weight,
            "cem_staging_distance": args.cem_staging_distance,
            "cem_max_action_norm": args.cem_max_action_norm,
            "cem_trajectory_cost_weight": args.cem_trajectory_cost_weight,
            "cem_progress_regression_weight": args.cem_progress_regression_weight,
            "cem_object_speed_weight": args.cem_object_speed_weight,
            "cem_min_predicted_improvement": args.cem_min_predicted_improvement,
        },
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
            split=args.split,
        )
        results["prediction_by_branch"] = evaluate_prediction_by_branch(
            model,
            args.dataset_dir,
            stats,
            device,
            batch_size=args.batch_size,
            stride=args.prediction_stride,
            max_scenarios=args.max_scenarios,
            split=args.split,
        )
        results["recovery_prefix_prediction"] = evaluate_prediction_horizons(
            model,
            args.dataset_dir,
            stats,
            device,
            batch_size=args.batch_size,
            stride=args.prediction_stride,
            max_scenarios=args.max_scenarios,
            branches=("R",),
            start_phases={"reposition", "recontact"},
            split=args.split,
        )
    if not args.skip_ranking:
        results["counterfactual_ranking"] = evaluate_counterfactual_ranking(
            model,
            args.dataset_dir,
            stats,
            device,
            max_scenarios=args.max_scenarios,
            split=args.split,
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
            split=args.split,
            planner_kwargs={
                "horizon": args.cem_horizon,
                "num_samples": args.cem_samples,
                "topk": args.cem_topk,
                "iterations": args.cem_iterations,
                "action_repeat": args.cem_action_repeat,
                "initial_std": args.cem_initial_std,
                "action_cost": args.cem_action_cost,
                "smoothness_cost": args.cem_smoothness_cost,
                "staging_weight": args.cem_staging_weight,
                "staging_distance": args.cem_staging_distance,
                "max_action_norm": args.cem_max_action_norm,
                "trajectory_cost_weight": args.cem_trajectory_cost_weight,
                "progress_regression_weight": args.cem_progress_regression_weight,
                "object_speed_weight": args.cem_object_speed_weight,
                "min_predicted_improvement": args.cem_min_predicted_improvement,
            },
        )
    dump_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
