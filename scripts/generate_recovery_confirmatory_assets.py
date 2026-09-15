#!/usr/bin/env python
"""Plot the locked-planner confirmatory recovery evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#4C78A8"
ORANGE = "#F58518"
RED = "#E45756"


def _read(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _reports(run_dir, variant, seeds, report_name):
    return [
        _read(run_dir / variant / f"seed_{seed}" / report_name)[
            "closed_loop_recovery"
        ]
        for seed in seeds
    ]


def _bar_panel(ax, baseline, recovery, ylabel, title, ylim=None):
    means = (np.mean(baseline), np.mean(recovery))
    bars = ax.bar((0, 1), means, color=(BLUE, ORANGE), width=0.55)
    for index, values in enumerate((baseline, recovery)):
        ax.scatter(
            np.full(len(values), index), values, color="black", s=24, zorder=3
        )
    for bar, value in zip(bars, means):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks((0, 1), ("SFN", "SFR"))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-variant", default="D_SFN_balanced")
    parser.add_argument("--recovery-variant", default="D_SFR_balanced")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    baseline = _reports(
        args.run_dir, args.baseline_variant, args.seeds, args.report_name
    )
    recovery = _reports(
        args.run_dir, args.recovery_variant, args.seeds, args.report_name
    )
    aggregate = _read(args.aggregate)["closed_loop_recovery"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    panels = (
        ("success_rate", "Success rate", "95% coverage success", (0, 1.03)),
        ("mean_final_coverage", "Coverage", "Final coverage", (0, 1.03)),
        ("mean_max_coverage", "Coverage", "Maximum coverage", (0, 1.03)),
        (
            "mean_coverage_retention_loss",
            "Maximum − final coverage",
            "Retention loss (lower is better)",
            (0, None),
        ),
        (
            "mean_action_cost",
            "Cumulative action norm",
            "Action cost (lower is better)",
            (0, None),
        ),
    )
    for ax, (metric, ylabel, title, ylim) in zip(axes.flat[:5], panels):
        _bar_panel(
            ax,
            [report[metric] for report in baseline],
            [report[metric] for report in recovery],
            ylabel,
            title,
            ylim,
        )
        if metric == "success_rate":
            totals = aggregate["success_rate"]
            ax.text(
                0.02,
                0.93,
                f"Successes: {totals['baseline_total_successes']}/180 vs "
                f"{totals['recovery_rich_total_successes']}/180",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )

    trace_ax = axes.flat[5]
    for reports, label, color in (
        (baseline, "SFN", BLUE),
        (recovery, "SFR", ORANGE),
    ):
        traces = np.asarray(
            [
                row["coverage_trace"]
                for report in reports
                for row in report["per_scenario"]
            ],
            dtype=np.float64,
        )
        steps = np.arange(traces.shape[1])
        trace_ax.plot(steps, traces.mean(axis=0), color=color, label=label)
        trace_ax.fill_between(
            steps,
            np.quantile(traces, 0.25, axis=0),
            np.quantile(traces, 0.75, axis=0),
            color=color,
            alpha=0.16,
        )
    trace_ax.axhline(0.95, color=RED, linestyle="--", linewidth=1)
    trace_ax.set_xlabel("Executed environment step")
    trace_ax.set_ylabel("Coverage")
    trace_ax.set_ylim(0, 1.03)
    trace_ax.set_title("Coverage dynamics (mean and IQR)")
    trace_ax.legend()

    fig.suptitle(
        "Fresh 60-scenario confirmatory evaluation — locked P3 planner",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
