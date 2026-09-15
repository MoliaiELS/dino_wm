#!/usr/bin/env python
"""Generate static scientific figures for the Phase 1/2 experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#777777"
BRANCH_COLORS = [BLUE, RED, GRAY, GREEN]
VARIANTS = ("D_SF_balanced", "D_SFR_balanced")
VARIANT_LABELS = ("SF (S+F1+F2)", "SFR (S+F1+R)")


def _read(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _annotate_bars(ax, bars, digits=2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def branch_montage(dataset_dir, output_dir, scenario_id):
    scenario_dir = dataset_dir / "scenarios" / scenario_id
    branches = ("S", "F1", "F2", "R")
    descriptions = (
        "Nominal oracle success",
        "Open-loop nominal continuation",
        "Zero-action neutral continuation",
        "Oracle replanning recovery",
    )
    fractions = (0.0, 0.5, 1.0)
    row_labels = ("Start", "Middle", "Final")
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.6))
    for column, (branch, description) in enumerate(zip(branches, descriptions)):
        reader = imageio.get_reader(scenario_dir / f"{branch}.mp4")
        try:
            frames = [np.asarray(frame) for frame in reader]
        finally:
            reader.close()
        with np.load(scenario_dir / f"{branch}.npz") as record:
            coverage = np.asarray(record["coverage"])
        for row, fraction in enumerate(fractions):
            index = int(round(fraction * (len(frames) - 1)))
            axes[row, column].imshow(frames[index])
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].text(
                0.03,
                0.96,
                f"coverage={coverage[index]:.3f}",
                transform=axes[row, column].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if column == 0:
                axes[row, column].set_ylabel(row_labels[row], fontsize=10)
        axes[0, column].set_title(f"{branch}\n{description}", fontsize=10)
    fig.suptitle(
        f"Paired PushT branches — {scenario_id}", fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5,
        0.01,
        "S starts from the original scenario; F1, F2 and R share the exact post-perturbation state.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    _save(fig, output_dir / "phase1-branch-montage.png")


def dataset_summary(dataset_dir, output_dir):
    manifest = _read(dataset_dir / "manifest.json")
    audit = _read(dataset_dir / "audit.json")
    branches = ("S", "F1", "F2", "R")
    final_coverage = {branch: [] for branch in branches}
    for scenario_dir in sorted((dataset_dir / "scenarios").glob("scenario_*")):
        for branch in branches:
            with np.load(scenario_dir / f"{branch}.npz") as record:
                final_coverage[branch].append(float(record["coverage"][-1]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    split_names = ("Train", "Validation", "Test")
    split_values = [audit["split_counts"][name] for name in ("train", "valid", "test")]
    bars = axes[0].bar(split_names, split_values, color=[BLUE, ORANGE, GREEN])
    _annotate_bars(axes[0], bars, digits=0)
    axes[0].set_title("Scenario-disjoint split")
    axes[0].set_ylabel("Number of paired scenarios")
    axes[0].set_ylim(0, max(split_values) * 1.18)

    cell_order = (
        "agent_lateral:low",
        "agent_lateral:medium",
        "agent_lateral:high",
        "agent_retreat:low",
        "agent_retreat:medium",
        "agent_retreat:high",
    )
    cell_labels = ("Lat-L", "Lat-M", "Lat-H", "Ret-L", "Ret-M", "Ret-H")
    cell_values = [
        audit["perturbation_type_severity_counts"][key] for key in cell_order
    ]
    bars = axes[1].bar(cell_labels, cell_values, color=[BLUE] * 3 + [ORANGE] * 3)
    _annotate_bars(axes[1], bars, digits=0)
    axes[1].set_title("Perturbation cells")
    axes[1].set_ylabel("Number of paired scenarios")
    axes[1].set_ylim(0, max(cell_values) * 1.22)

    box = axes[2].boxplot(
        [final_coverage[branch] for branch in branches],
        labels=branches,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for patch, color in zip(box["boxes"], BRANCH_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[2].axhline(0.95, color=RED, linestyle="--", linewidth=1, label="Success=0.95")
    axes[2].set_title("Final coverage by branch")
    axes[2].set_ylabel("Geometric coverage [0, 1]")
    axes[2].set_ylim(-0.03, 1.08)
    axes[2].legend(loc="lower right", fontsize=8)
    success_rates = audit["branch_final_success_rates"]
    axes[2].text(
        0.03,
        0.04,
        "  ".join(f"{branch}: {success_rates[branch]:.0%}" for branch in branches),
        transform=axes[2].transAxes,
        fontsize=8,
    )

    fig.suptitle(
        f"Phase 1 dataset audit — {manifest['scenario_count']} paired scenarios",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, output_dir / "phase1-dataset-summary.png")


def training_curves(run_dir, output_dir, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, variant, label, color in zip(
        axes, VARIANTS, VARIANT_LABELS, (BLUE, ORANGE)
    ):
        train_curves = []
        valid_curves = []
        for seed in seeds:
            metrics = _read(run_dir / variant / f"seed_{seed}" / "metrics.json")
            train = np.asarray([row["train_loss"] for row in metrics["history"]])
            valid = np.asarray([row["valid_loss"] for row in metrics["history"]])
            train_curves.append(train)
            valid_curves.append(valid)
            ax.plot(
                np.arange(1, len(valid) + 1),
                valid,
                color=color,
                alpha=0.22,
                linewidth=1,
            )
        train_curves = np.stack(train_curves)
        valid_curves = np.stack(valid_curves)
        epochs = np.arange(1, train_curves.shape[1] + 1)
        ax.plot(epochs, train_curves.mean(axis=0), color=GRAY, label="Train mean")
        ax.plot(epochs, valid_curves.mean(axis=0), color=color, label="Validation mean")
        ax.fill_between(
            epochs,
            valid_curves.min(axis=0),
            valid_curves.max(axis=0),
            color=color,
            alpha=0.16,
            label="Validation seed range",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("One-step + 5-step rollout MSE (log scale)")
    fig.suptitle(
        "Optimization curves across three training seeds",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, output_dir / "phase2-training-curves.png")


def main_results(run_dir, output_dir, seeds):
    aggregate = _read(run_dir / "three_seed_aggregate.json")
    ranking = aggregate["counterfactual_ranking"]["recovery_top1_accuracy"]
    prediction = aggregate["prediction"]
    closed_loop = aggregate["closed_loop_recovery"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))

    x = np.arange(len(seeds) + 1)
    baseline = np.asarray(ranking["baseline_per_seed"] + [ranking["baseline_mean"]])
    recovery = np.asarray(
        ranking["recovery_rich_per_seed"] + [ranking["recovery_rich_mean"]]
    )
    width = 0.34
    bars_a = axes[0, 0].bar(x - width / 2, baseline, width, color=BLUE, label="SF")
    bars_b = axes[0, 0].bar(x + width / 2, recovery, width, color=ORANGE, label="SFR")
    _annotate_bars(axes[0, 0], bars_a, digits=2)
    _annotate_bars(axes[0, 0], bars_b, digits=2)
    axes[0, 0].set_xticks(x, [f"seed {seed}" for seed in seeds] + ["Mean"])
    axes[0, 0].set_ylim(0, 1.12)
    axes[0, 0].set_ylabel("Recovery top-1 accuracy")
    axes[0, 0].set_title("Counterfactual action ranking")
    axes[0, 0].legend()

    horizons = [1, 5, 10, 20]
    for label, field, color, marker in (
        ("SF", "baseline", BLUE, "o"),
        ("SFR", "recovery_rich", ORANGE, "s"),
    ):
        means = []
        lows = []
        highs = []
        for horizon in horizons:
            metric = prediction[str(horizon)]["object_goal_position_rmse_px"]
            values = np.asarray(metric[f"{field}_per_seed"])
            means.append(values.mean())
            lows.append(values.min())
            highs.append(values.max())
        means = np.asarray(means)
        axes[0, 1].plot(horizons, means, color=color, marker=marker, label=label)
        axes[0, 1].fill_between(horizons, lows, highs, color=color, alpha=0.15)
    axes[0, 1].set_xticks(horizons)
    axes[0, 1].set_xlabel("Prediction horizon [steps]")
    axes[0, 1].set_ylabel("Object-goal position RMSE [px]")
    axes[0, 1].set_title("Multi-step task-state prediction")
    axes[0, 1].legend()

    coverage_names = ("Final", "Maximum")
    baseline_coverage = [
        closed_loop["final_coverage"]["baseline_mean"],
        closed_loop["max_coverage"]["baseline_mean"],
    ]
    recovery_coverage = [
        closed_loop["final_coverage"]["recovery_rich_mean"],
        closed_loop["max_coverage"]["recovery_rich_mean"],
    ]
    cx = np.arange(2)
    bars_a = axes[1, 0].bar(
        cx - width / 2, baseline_coverage, width, color=BLUE, label="SF"
    )
    bars_b = axes[1, 0].bar(
        cx + width / 2, recovery_coverage, width, color=ORANGE, label="SFR"
    )
    _annotate_bars(axes[1, 0], bars_a, digits=3)
    _annotate_bars(axes[1, 0], bars_b, digits=3)
    axes[1, 0].axhline(0.95, color=RED, linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(cx, coverage_names)
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_ylabel("Coverage [0, 1]")
    axes[1, 0].set_title("Closed-loop coverage (three-seed mean)")
    axes[1, 0].legend()
    axes[1, 0].text(
        0.02,
        0.87,
        "Success totals: SF 2/90, SFR 5/90",
        transform=axes[1, 0].transAxes,
        fontsize=9,
    )

    cost = closed_loop["action_cost"]
    bars = axes[1, 1].bar(
        (0, 1),
        (cost["baseline_mean"], cost["recovery_rich_mean"]),
        color=(BLUE, ORANGE),
        width=0.55,
    )
    _annotate_bars(axes[1, 1], bars, digits=2)
    for variant_index, key in enumerate(("baseline_per_seed", "recovery_rich_per_seed")):
        values = cost[key]
        axes[1, 1].scatter(
            np.full(len(values), variant_index),
            values,
            color="black",
            s=22,
            zorder=3,
            label="Training seeds" if variant_index == 0 else None,
        )
    axes[1, 1].set_xticks((0, 1), ("SF", "SFR"))
    axes[1, 1].set_ylabel("Cumulative action norm")
    axes[1, 1].set_title("Closed-loop action cost")
    axes[1, 1].set_ylim(0, max(cost["baseline_per_seed"]) * 1.2)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        "Corrected fixed-budget comparison — seeds 0, 1 and 2",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, output_dir / "phase2-main-results.png")


def coverage_dynamics(run_dir, output_dir, seeds):
    traces = {variant: [] for variant in VARIANTS}
    drops_by_seed = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in seeds:
            report = _read(
                run_dir / variant / f"seed_{seed}" / "closed_loop_short_shaped.json"
            )["closed_loop_recovery"]
            seed_drops = []
            for row in report["per_scenario"]:
                trace = np.asarray(row["coverage_trace"], dtype=np.float64)
                if len(trace) < 36:
                    trace = np.pad(trace, (0, 36 - len(trace)), mode="edge")
                traces[variant].append(trace[:36])
                seed_drops.append(row["max_coverage"] - row["final_coverage"])
            drops_by_seed[variant].append(float(np.mean(seed_drops)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant, label, color in zip(VARIANTS, ("SF", "SFR"), (BLUE, ORANGE)):
        values = np.stack(traces[variant])
        steps = np.arange(values.shape[1])
        axes[0].plot(steps, values.mean(axis=0), color=color, label=label)
        axes[0].fill_between(
            steps,
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
            color=color,
            alpha=0.16,
        )
    axes[0].axhline(0.95, color=RED, linestyle="--", linewidth=1, label="Success")
    axes[0].set_xlabel("Executed environment step")
    axes[0].set_ylabel("Coverage [0, 1]")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Coverage trajectory (mean and interquartile range)")
    axes[0].legend(fontsize=8)

    means = [np.mean(drops_by_seed[variant]) for variant in VARIANTS]
    bars = axes[1].bar((0, 1), means, color=(BLUE, ORANGE), width=0.55)
    _annotate_bars(axes[1], bars, digits=3)
    for index, variant in enumerate(VARIANTS):
        axes[1].scatter(
            np.full(len(seeds), index),
            drops_by_seed[variant],
            color="black",
            s=22,
            zorder=3,
        )
    axes[1].set_xticks((0, 1), ("SF", "SFR"))
    axes[1].set_ylabel("Maximum coverage − final coverage")
    axes[1].set_title("Progress lost before episode end")
    axes[1].set_ylim(0, max(means) * 1.4)

    fig.suptitle(
        "Closed-loop recovery dynamics — 90 rollouts per condition",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, output_dir / "phase2-coverage-dynamics.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-id", default="scenario_000000")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    branch_montage(args.dataset_dir, args.output_dir, args.scenario_id)
    dataset_summary(args.dataset_dir, args.output_dir)
    training_curves(args.run_dir, args.output_dir, args.seeds)
    main_results(args.run_dir, args.output_dir, args.seeds)
    coverage_dynamics(args.run_dir, args.output_dir, args.seeds)
    for path in sorted(args.output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
