"""Train the Phase 2 oracle-state dynamics model."""

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

from models.state_world_model import StateWorldModel
from phase2.data import (
    PairedStateWindowDataset,
    branch_balanced_weights,
    compute_train_normalization,
)
from phase2.evaluation import dump_json


def _default_dataset_dir():
    root = os.environ.get("DATASET_DIR")
    return None if root is None else str(Path(root) / "pusht_recovery_phase1_pilot_v1")


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _run_epoch(model, loader, device, optimizer=None, max_steps=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        with torch.set_grad_enabled(training):
            _, loss = model(states, actions)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = states.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No batches were processed")
    return total_loss / sample_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=_default_dataset_dir())
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--normalization-variant",
        help="Training-only reference variant; use D_SF for the primary fixed-budget pair",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-valid-windows", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--state-emb-dim", type=int, default=64)
    parser.add_argument("--action-emb-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_dir is None:
        raise ValueError("Set DATASET_DIR or pass --dataset-dir")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    _seed_everything(args.seed)
    device = _device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization_variant = args.normalization_variant or args.variant
    stats = compute_train_normalization(args.dataset_dir, normalization_variant)
    train_dataset = PairedStateWindowDataset(
        args.dataset_dir,
        args.variant,
        "train",
        stats=stats,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    valid_dataset = PairedStateWindowDataset(
        args.dataset_dir,
        args.variant,
        "valid",
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    max_context = int(train_dataset[0]["actions"].shape[0])
    model_config = {
        "state_dim": train_dataset.state_dim,
        "action_dim": train_dataset.action_dim,
        "max_context": max_context,
        "model_dim": args.model_dim,
        "state_emb_dim": args.state_emb_dim,
        "action_emb_dim": args.action_emb_dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.mlp_dim,
        "dim_head": args.model_dim // args.heads,
        "dropout": args.dropout,
    }
    model = StateWorldModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        **vars(args),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "device_resolved": str(device),
        "git_commit": _git_commit(),
        "model": model_config,
        "parameter_count": parameter_count,
        "normalization": stats.to_dict(),
        "normalization_variant": normalization_variant,
        "train_windows": len(train_dataset),
        "valid_windows": len(valid_dataset),
        "train_branch_counts": train_dataset.branch_counts,
        "valid_branch_counts": valid_dataset.branch_counts,
    }
    dump_json(output_dir / "config.json", config)

    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_steps=args.max_steps_per_epoch,
        )
        valid_loss = _run_epoch(
            model,
            valid_loader,
            device,
            optimizer=None,
            max_steps=args.max_steps_per_epoch,
        )
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "normalization": stats.to_dict(),
            "variant": args.variant,
            "seed": args.seed,
            "epoch": epoch,
            "valid_loss": valid_loss,
            "git_commit": config["git_commit"],
        }
        torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
        dump_json(output_dir / "metrics.json", {"best_valid_loss": best_loss, "history": history})
    print(
        json.dumps(
            {
                "status": "completed",
                "best_valid_loss": best_loss,
                "checkpoint": str(output_dir / "checkpoint_best.pt"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
