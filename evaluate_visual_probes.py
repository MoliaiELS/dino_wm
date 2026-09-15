"""Train frozen probes on raw, random or learned visual representations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models.visual_temporal_model import VisualTemporalWorldModel
from phase2.evaluation import dump_json
from phase3.data import ProprioStats, VisualProbeFrameDataset


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(checkpoint_path, representation, seed, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = VisualTemporalWorldModel(**checkpoint["model_config"])
    if representation == "trained_adapter":
        model.load_state_dict(checkpoint["model_state_dict"])
    elif representation != "random_adapter":
        raise ValueError(f"Model loading is not used for {representation}")
    model.requires_grad_(False).eval().to(device)
    return model, ProprioStats.from_dict(checkpoint["proprio_stats"]), checkpoint


def _make_loader(cache_dir, split, stats, args, shuffle=False):
    dataset = VisualProbeFrameDataset(
        cache_dir,
        split,
        stats,
        history=args.history,
        recoverability_horizon=args.recoverability_horizon,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    return dataset, DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )


def _extract(loader, representation, model, device):
    features = []
    labels = defaultdict_list()
    with torch.inference_mode():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            history_actions = batch["history_actions"].to(device, non_blocking=True)
            if representation in {"trained_adapter", "random_adapter"}:
                values = model.contextual_representation(
                    tokens, proprio, history_actions
                )
            elif representation == "raw_dino":
                # Standard frozen-DINO baseline: mean pooled current patch tokens,
                # with the same proprioception and past actions available to B.
                values = torch.cat(
                    [
                        tokens[:, -1].mean(dim=1),
                        proprio[:, -1],
                        history_actions.flatten(start_dim=1),
                    ],
                    dim=-1,
                )
            elif representation == "oracle_state":
                values = batch["oracle_state"].to(device)
            else:
                raise ValueError(f"Unknown representation {representation}")
            features.append(values.cpu().numpy())
            for key in (
                "coverage",
                "off_nominal",
                "off_nominal_mask",
                "recoverable",
                "recoverability_mask",
                "frame",
            ):
                value = batch[key]
                labels[key].extend(
                    value.cpu().numpy().tolist() if torch.is_tensor(value) else list(value)
                )
            labels["scenario_id"].extend(list(batch["scenario_id"]))
            labels["branch"].extend(list(batch["branch"]))
            labels["phase"].extend(list(batch["phase"]))
    result = {key: np.asarray(value) for key, value in labels.items()}
    result["features"] = np.concatenate(features, axis=0).astype(np.float32)
    return result


def defaultdict_list():
    from collections import defaultdict

    return defaultdict(list)


def _regression_metrics(target, prediction):
    correlation = spearmanr(target, prediction).statistic
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": float(correlation),
    }


def _classification_metrics(target, probability, threshold=0.5):
    predicted = probability >= threshold
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(np.mean(np.square(probability - target))),
        "positive_fraction": float(np.mean(target)),
    }


def _fit_progress(train, valid, test, alphas):
    best = None
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train["features"], train["coverage"])
        prediction = model.predict(valid["features"])
        score = mean_absolute_error(valid["coverage"], prediction)
        if best is None or score < best[0]:
            best = (score, alpha, model)
    _, alpha, model = best
    prediction = model.predict(test["features"])
    return {
        "selected_alpha": alpha,
        "validation_mae": best[0],
        "test": _regression_metrics(test["coverage"], prediction),
        "test_predictions": prediction.astype(float).tolist(),
        "test_targets": test["coverage"].astype(float).tolist(),
        "test_scenario_ids": test["scenario_id"].tolist(),
    }


def _fit_classifier(train, valid, test, label_key, mask_key, cs):
    train_mask = train[mask_key].astype(bool)
    valid_mask = valid[mask_key].astype(bool)
    test_mask = test[mask_key].astype(bool)
    if len(np.unique(train[label_key][train_mask])) != 2:
        raise ValueError(f"Training label {label_key} does not contain two classes")
    best = None
    for c_value in cs:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=1000,
                random_state=0,
            ),
        )
        model.fit(train["features"][train_mask], train[label_key][train_mask])
        probability = model.predict_proba(valid["features"][valid_mask])[:, 1]
        score = roc_auc_score(valid[label_key][valid_mask], probability)
        if best is None or score > best[0]:
            best = (score, c_value, model)
    _, c_value, model = best
    probability = model.predict_proba(test["features"][test_mask])[:, 1]
    target = test[label_key][test_mask].astype(int)
    return {
        "selected_c": c_value,
        "validation_roc_auc": best[0],
        "test": _classification_metrics(target, probability),
        "test_probabilities": probability.astype(float).tolist(),
        "test_targets": target.tolist(),
        "test_scenario_ids": test["scenario_id"][test_mask].tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--test-cache-dir", required=True)
    parser.add_argument(
        "--representation",
        choices=("raw_dino", "random_adapter", "trained_adapter", "oracle_state"),
        required=True,
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--recoverability-horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--logistic-c", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    return parser.parse_args()


def main():
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if args.representation in {"trained_adapter", "random_adapter"}:
        if not args.checkpoint:
            raise ValueError("Adapter representations require --checkpoint")
        model, stats, checkpoint = _load_model(
            args.checkpoint, args.representation, args.seed, device
        )
        source_variant = checkpoint["variant"]
    else:
        # Use the same train-only common-core normalization as both adapters.
        from phase3.data import compute_proprio_stats

        stats = compute_proprio_stats(args.train_cache_dir, "D_SF")
        model = None
        source_variant = None
    train_dataset, train_loader = _make_loader(
        args.train_cache_dir, "train", stats, args
    )
    valid_dataset, valid_loader = _make_loader(
        args.train_cache_dir, "valid", stats, args
    )
    test_dataset, test_loader = _make_loader(
        args.test_cache_dir, "test", stats, args
    )
    train = _extract(train_loader, args.representation, model, device)
    valid = _extract(valid_loader, args.representation, model, device)
    test = _extract(test_loader, args.representation, model, device)
    report = {
        "representation": args.representation,
        "source_variant": source_variant,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "train_cache_dir": str(Path(args.train_cache_dir).resolve()),
        "test_cache_dir": str(Path(args.test_cache_dir).resolve()),
        "history": args.history,
        "recoverability_definition": (
            f"success under recorded S/N/R oracle continuation within {args.recoverability_horizon} steps"
        ),
        "off_nominal_definition": {
            "positive_phases": sorted(["reposition", "recontact", "open_loop_failure", "neutral"]),
            "negative_phases": sorted(["nominal_push", "hold"]),
            "excluded_ambiguous_phase": "corrective_push",
        },
        "sample_counts": {
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
            "feature_dim": int(train["features"].shape[1]),
        },
        "task_progress": _fit_progress(
            train, valid, test, args.ridge_alphas
        ),
        "off_nominal": _fit_classifier(
            train, valid, test, "off_nominal", "off_nominal_mask", args.logistic_c
        ),
        "recoverability": _fit_classifier(
            train,
            valid,
            test,
            "recoverable",
            "recoverability_mask",
            args.logistic_c,
        ),
    }
    dump_json(args.output, report)
    summary = {
        "output": args.output,
        "representation": args.representation,
        "progress": report["task_progress"]["test"],
        "off_nominal": report["off_nominal"]["test"],
        "recoverability": report["recoverability"]["test"],
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
