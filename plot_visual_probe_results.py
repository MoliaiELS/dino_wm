"""Plot the locked three-seed Experiment B probe comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "raw_dino",
    "random_adapter",
    "temporal_sfn",
    "temporal_sfr",
    "oracle_state_mlp_upper",
)
LABELS = ("Raw DINO", "Random\nadapter", "Temporal\nSFN", "Temporal\nSFR", "Oracle-state\nMLP")
COLORS = ("#8c8c8c", "#b8b8b8", "#4c78a8", "#e45756", "#54a24b")


def _panel(ax, report, probe, metric, title, ylabel):
    means = [report["representations"][name][probe][metric]["mean"] for name in ORDER]
    x = np.arange(len(ORDER))
    ax.bar(x, means, color=COLORS, width=0.72, edgecolor="white", linewidth=0.7)
    for index, name in enumerate(ORDER):
        values = report["representations"][name][probe][metric]["per_seed"]
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(index + np.asarray(jitter), values, color="#202020", s=16, zorder=3)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot(input_path, output_path):
    with open(input_path, encoding="utf-8") as file:
        report = json.load(file)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    _panel(axes[0, 0], report, "task_progress", "mae", "A  Task progress", "MAE (lower is better)")
    _panel(axes[0, 1], report, "off_nominal", "brier", "B  Off-nominal state", "Brier error (lower is better)")
    _panel(axes[1, 0], report, "recoverability", "brier", "C  Five-step recoverability", "Brier error (lower is better)")

    ax = axes[1, 1]
    probes = ("task_progress", "off_nominal", "recoverability")
    labels = ("Progress MAE", "Off-nominal Brier", "Recoverability Brier")
    effects = report["paired_recovery_effect"]
    means = np.asarray([effects[key]["mean"] for key in probes])
    cis = np.asarray([effects[key]["crossed_bootstrap_95_ci"] for key in probes])
    positions = np.arange(len(probes))
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack([means - cis[:, 0], cis[:, 1] - means]),
        fmt="o",
        color="#e45756",
        ecolor="#555555",
        capsize=4,
    )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("SFN error minus SFR error (positive favors SFR)")
    ax.set_title("D  Paired recovery-data effect (95% CI)", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Experiment B: frozen-DINO representation probes on 60 fresh scenarios",
        fontsize=15,
        fontweight="bold",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
