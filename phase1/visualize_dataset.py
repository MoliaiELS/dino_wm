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
    audit = _read_json(dataset_dir / "audit.json")
    audit_rows = {
        row["scenario_id"]: row
        for row in audit["recovery_vs_nominal_continuation"]["per_scenario"]
    }
    representatives = {}
    for scenario_dir in sorted((dataset_dir / "scenarios").glob("scenario_*")):
        metadata = _read_json(scenario_dir / "metadata.json")
        cell = (
            f"{metadata['perturbation']['type']}:"
            f"{metadata['perturbation']['severity']}"
        )
        metrics = audit_rows[metadata["scenario_id"]]
        # Deliberately choose the most visibly recovery-rich example in each
        # cell instead of the first scenario, which can make N/S and R look
        # nearly identical in a qualitative audit.
        score = (
            metrics["recovery_prefix_steps"],
            metrics["agent_object_position_rmse_px"],
            metrics["action_rmse_command"],
        )
        current = representatives.get(cell)
        if current is None or score > current[3]:
            representatives[cell] = (scenario_dir, metadata, metrics, score)
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
        scenario_dir, metadata, _, _ = representatives[cell]
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
        scenario_dir, metadata, _, _ = representatives[cell]
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


def _overlay_path(ax, states, stop, frame_shape):
    height, width = frame_shape[:2]
    scale_x, scale_y = width / 512.0, height / 512.0
    stop = max(1, min(int(stop), len(states)))
    ax.plot(
        states[:stop, 0] * scale_x,
        states[:stop, 1] * scale_y,
        color="#00A6D6",
        linewidth=1.8,
        label="agent path",
    )
    ax.plot(
        states[:stop, 11] * scale_x,
        states[:stop, 12] * scale_y,
        color="#F58518",
        linewidth=1.8,
        label="object path",
    )


def recovery_storyboard(dataset_dir, output_dir, representatives):
    """Show where nominal success ends and true post-perturbation recovery starts."""

    columns = (
        "S initial",
        "S at branch point",
        "R post-perturbation start",
        "R first recontact",
        "R first success",
    )
    fig, axes = plt.subplots(6, 5, figsize=(14.5, 16.2))
    for row, cell in enumerate(CELL_ORDER):
        scenario_dir, metadata, metrics, _ = representatives[cell]
        s_frames = _video_frames(scenario_dir / "S.mp4")
        r_frames = _video_frames(scenario_dir / "R.mp4")
        with np.load(scenario_dir / "S.npz") as nominal, np.load(
            scenario_dir / "R.npz"
        ) as recovery:
            s_state = np.asarray(nominal["sim_state"])
            r_state = np.asarray(recovery["sim_state"])
            s_coverage = np.asarray(nominal["coverage"])
            r_coverage = np.asarray(recovery["coverage"])
        branch_frame = min(int(metadata["branch_step_in_nominal"]), len(s_frames) - 1)
        contact_step = metadata["branches"]["R"]["first_contact_step"]
        success_step = metadata["branches"]["R"]["first_success_step"]
        contact_frame = min(
            len(r_frames) - 1,
            len(r_frames) // 2 if contact_step is None else int(contact_step) + 1,
        )
        success_frame = min(
            len(r_frames) - 1,
            len(r_frames) - 1 if success_step is None else int(success_step) + 1,
        )
        frames = (
            s_frames[0],
            s_frames[branch_frame],
            r_frames[0],
            r_frames[contact_frame],
            r_frames[success_frame],
        )
        coverages = (
            s_coverage[0],
            s_coverage[branch_frame],
            r_coverage[0],
            r_coverage[contact_frame],
            r_coverage[success_frame],
        )
        for column, (frame, coverage) in enumerate(zip(frames, coverages)):
            ax = axes[row, column]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            if column >= 2:
                path_stop = (1, contact_frame + 1, success_frame + 1)[column - 2]
                _overlay_path(ax, r_state, path_stop, frame.shape)
            if column == 2:
                height, width = frame.shape[:2]
                scale_x, scale_y = width / 512.0, height / 512.0
                start = s_state[branch_frame, :2] * (scale_x, scale_y)
                end = r_state[0, :2] * (scale_x, scale_y)
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#D81B60", "lw": 2.4},
                )
            ax.text(
                0.03,
                0.96,
                f"cov={float(coverage):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            if row == 0:
                ax.set_title(columns[column], fontsize=9.5)
            if column == 0:
                ax.set_ylabel(
                    f"{CELL_LABELS[cell]}\n"
                    f"prefix={metrics['recovery_prefix_steps']} steps",
                    fontsize=8.5,
                )
    fig.suptitle(
        "Nominal success S versus post-perturbation recovery R",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "Magenta arrow: exogenous agent displacement. Cyan/orange: R agent/object paths. "
        "Representatives maximize recovery-prefix distinctiveness within each cell.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(
        output_dir / "phase1-s-versus-r-recovery-storyboard.png",
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
    recovery_storyboard(args.dataset_dir, args.output_dir, representatives)
    print(args.output_dir / "phase1-six-cell-recovery-montage.png")
    print(args.output_dir / "phase1-six-cell-trajectory-overlays.png")
    print(args.output_dir / "phase1-s-versus-r-recovery-storyboard.png")


if __name__ == "__main__":
    main()
