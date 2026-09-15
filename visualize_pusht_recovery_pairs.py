"""Visualize all perturbation cells and the R-versus-N distinction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CELL_ORDER = (
    "agent_lateral:low",
    "agent_lateral:medium",
    "agent_lateral:high",
    "agent_retreat:low",
    "agent_retreat:medium",
    "agent_retreat:high",
)
CELL_LABELS = {
    "agent_lateral:low": "Lateral / low",
    "agent_lateral:medium": "Lateral / medium",
    "agent_lateral:high": "Lateral / high",
    "agent_retreat:low": "Retreat / low",
    "agent_retreat:medium": "Retreat / medium",
    "agent_retreat:high": "Retreat / high",
}


def _read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _default_dataset_dir():
    root = os.environ.get("DATASET_DIR")
    return None if root is None else Path(root) / "pusht_recovery_phase1_pilot_v2"


def _representatives(dataset_dir):
    representatives = {}
    for scenario_dir in sorted((dataset_dir / "scenarios").glob("scenario_*")):
        metadata = _read_json(scenario_dir / "metadata.json")
        cell = (
            f"{metadata['perturbation']['type']}:"
            f"{metadata['perturbation']['severity']}"
        )
        representatives.setdefault(cell, (scenario_dir, metadata))
    missing = set(CELL_ORDER) - set(representatives)
    if missing:
        raise ValueError(f"Dataset is missing perturbation cells: {sorted(missing)}")
    return representatives


def _video_frames(path):
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def six_cell_montage(dataset_dir, output_dir, representatives):
    columns = ("Nominal N start", "Recovery R start", "R recontact", "R final")
    fig, axes = plt.subplots(6, 4, figsize=(11.5, 15.5))
    for row, cell in enumerate(CELL_ORDER):
        scenario_dir, metadata = representatives[cell]
        n_frames = _video_frames(scenario_dir / "N.mp4")
        r_frames = _video_frames(scenario_dir / "R.mp4")
        first_contact = metadata["branches"]["R"]["first_contact_step"]
        recontact_frame = (
            len(r_frames) // 2
            if first_contact is None
            else min(int(first_contact) + 1, len(r_frames) - 1)
        )
        frames = (n_frames[0], r_frames[0], r_frames[recontact_frame], r_frames[-1])
        coverages = []
        with np.load(scenario_dir / "N.npz") as nominal, np.load(
            scenario_dir / "R.npz"
        ) as recovery:
            coverages.extend(
                (
                    float(nominal["coverage"][0]),
                    float(recovery["coverage"][0]),
                    float(recovery["coverage"][recontact_frame]),
                    float(recovery["coverage"][-1]),
                )
            )
        for column, (frame, coverage) in enumerate(zip(frames, coverages)):
            ax = axes[row, column]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.03,
                0.96,
                f"coverage={coverage:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if row == 0:
                ax.set_title(columns[column], fontsize=10)
            if column == 0:
                ax.set_ylabel(CELL_LABELS[cell], fontsize=9)
    fig.suptitle(
        "Success-matched nominal continuation versus recovery",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "N starts before the exogenous perturbation; R starts after it. "
        "Each row is one perturbation type/severity cell.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(
        output_dir / "phase1-six-cell-recovery-montage.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def trajectory_overlays(dataset_dir, output_dir, representatives):
    audit = _read_json(dataset_dir / "audit.json")
    per_scenario = {
        row["scenario_id"]: row
        for row in audit["recovery_vs_nominal_continuation"]["per_scenario"]
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    for ax, cell in zip(axes.flat, CELL_ORDER):
        scenario_dir, metadata = representatives[cell]
        with np.load(scenario_dir / "N.npz") as nominal, np.load(
            scenario_dir / "R.npz"
        ) as recovery:
            n_state = np.asarray(nominal["sim_state"])
            r_state = np.asarray(recovery["sim_state"])
        ax.plot(
            n_state[:, 0], n_state[:, 1], "--", color="#4C78A8", label="N agent"
        )
        ax.plot(r_state[:, 0], r_state[:, 1], color="#F58518", label="R agent")
        ax.plot(
            n_state[:, 11], n_state[:, 12], "--", color="#72B7B2", label="N object"
        )
        ax.plot(r_state[:, 11], r_state[:, 12], color="#E45756", label="R object")
        ax.scatter(r_state[0, 0], r_state[0, 1], marker="x", color="#F58518")
        ax.scatter(r_state[0, 11], r_state[0, 12], marker="s", color="#E45756")
        ax.scatter(r_state[0, 22], r_state[0, 23], marker="*", s=80, color="#54A24B")
        metrics = per_scenario[metadata["scenario_id"]]
        ax.set_title(
            f"{CELL_LABELS[cell]}\n"
            f"prefix={metrics['recovery_prefix_steps']} steps, "
            f"action RMSE={metrics['action_rmse_command']:.2f}",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_xlabel("Simulator x [px]")
        ax.set_ylabel("Simulator y [px]")
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Aligned N and R trajectories across perturbation cells",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(
        output_dir / "phase1-six-cell-trajectory-overlays.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("report_assets"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_dir is None:
        raise ValueError("Set DATASET_DIR or pass --dataset-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representatives = _representatives(args.dataset_dir)
    six_cell_montage(args.dataset_dir, args.output_dir, representatives)
    trajectory_overlays(args.dataset_dir, args.output_dir, representatives)
    print(args.output_dir / "phase1-six-cell-recovery-montage.png")
    print(args.output_dir / "phase1-six-cell-trajectory-overlays.png")


if __name__ == "__main__":
    main()
