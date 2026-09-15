"""Train the Experiment B temporal adapter on cached frozen DINO tokens."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.visual_temporal_model import VisualTemporalWorldModel
from phase2.evaluation import dump_json
from phase3.data import (
    VisualFeatureWindowDataset,
    branch_balanced_weights,
    compute_proprio_stats,
)


def _default_cache_dir():
    root = os.environ.get("DATASET_DIR")
    return None if root is None else str(Path(root) / "pusht_recovery_dino_cache_v2")


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, device, optimizer=None, rollout_weight=0.25):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "one_step": 0.0, "rollout": 0.0, "visual": 0.0, "proprio": 0.0}
    count = 0
    for batch in loader:
        visual = batch["tokens"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = model(visual, proprio, actions)
            rollout_visual, rollout_proprio = model.rollout(
                visual[:, :1], proprio[:, :1], actions
            )
            rollout_visual_loss = torch.nn.functional.mse_loss(
                rollout_visual, visual[:, 1:]
            )
            rollout_proprio_loss = torch.nn.functional.mse_loss(
                rollout_proprio, proprio[:, 1:]
            )
            rollout_loss = rollout_visual_loss + model.proprio_loss_weight * rollout_proprio_loss
            loss = output["loss"] + float(rollout_weight) * rollout_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = visual.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["one_step"] += float(output["loss"].detach()) * batch_size
        totals["rollout"] += float(rollout_loss.detach()) * batch_size
        totals["visual"] += float(output["visual_loss"].detach()) * batch_size
        totals["proprio"] += float(output["proprio_loss"].detach()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("No visual batches were processed")
    return {key: value / count for key, value in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=_default_cache_dir())
    parser.add_argument("--variant", required=True)
    parser.add_argument("--normalization-variant", default="D_SF")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-actions", type=int, default=3)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-valid-windows", type=int)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--proprio-loss-weight", type=float, default=0.25)
    parser.add_argument("--rollout-weight", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cache_dir is None:
        raise ValueError("Set DATASET_DIR or pass --cache-dir")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_proprio_stats(args.cache_dir, args.normalization_variant)
    train_dataset = VisualFeatureWindowDataset(
        args.cache_dir,
        args.variant,
        "train",
        window_actions=args.window_actions,
        stats=stats,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    valid_dataset = VisualFeatureWindowDataset(
        args.cache_dir,
        args.variant,
        "valid",
        window_actions=args.window_actions,
        stats=stats,
        max_windows=args.max_valid_windows,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        branch_balanced_weights(train_dataset.entries),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    model = VisualTemporalWorldModel(
        patch_count=train_dataset.token_count,
        visual_dim=train_dataset.token_dim,
        proprio_dim=train_dataset.proprio_dim,
        action_dim=train_dataset.action_dim,
        max_context=args.window_actions,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.model_dim // args.heads,
        dropout=args.dropout,
        proprio_loss_weight=args.proprio_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config = {
        **vars(args),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "git_commit": _git_commit(),
        "model": model.config_dict(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "proprio_stats": stats.to_dict(),
        "train_windows": len(train_dataset),
        "valid_windows": len(valid_dataset),
        "train_branch_counts": train_dataset.branch_counts,
        "valid_branch_counts": valid_dataset.branch_counts,
    }
    dump_json(output_dir / "config.json", config)
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(
            model, train_loader, device, optimizer, rollout_weight=args.rollout_weight
        )
        valid = _run_epoch(
            model, valid_loader, device, optimizer=None, rollout_weight=args.rollout_weight
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}, **{f"valid_{k}": v for k, v in valid.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "proprio_stats": stats.to_dict(),
            "variant": args.variant,
            "seed": args.seed,
            "epoch": epoch,
            "valid_loss": valid["loss"],
            "git_commit": config["git_commit"],
        }
        torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
        if valid["loss"] < best_loss:
            best_loss = valid["loss"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
        dump_json(output_dir / "metrics.json", {"best_valid_loss": best_loss, "history": history})
    print(json.dumps({"status": "completed", "best_valid_loss": best_loss}), flush=True)


if __name__ == "__main__":
    main()
