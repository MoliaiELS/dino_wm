"""Aggregate paired Phase 2 results across training seeds and test scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase2.evaluation import dump_json


def _read(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _rows_by_id(rows, id_key):
    result = {row[id_key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {id_key} in evaluation rows")
    return result


def seed_bootstrap(values, samples=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("seed_bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "seed_bootstrap_95_ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def crossed_paired_bootstrap(values, samples=10000, seed=0):
    """Bootstrap crossed training-seed and scenario effects for paired deltas."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("crossed_paired_bootstrap requires a non-empty matrix")
    num_seeds, num_scenarios = values.shape
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, num_seeds, size=(samples, num_seeds))
    scenario_indices = rng.integers(
        0, num_scenarios, size=(samples, num_scenarios)
    )
    sampled = values[
        seed_indices[:, :, None], scenario_indices[:, None, :]
    ].mean(axis=(1, 2))
    return {
        "n_training_seeds": int(num_seeds),
        "n_scenarios": int(num_scenarios),
        "mean": float(values.mean()),
        "per_seed_means": values.mean(axis=1).tolist(),
        "crossed_bootstrap_95_ci": np.quantile(
            sampled, [0.025, 0.975]
        ).tolist(),
    }


def _paired_row_matrices(baseline_row_sets, recovery_row_sets, id_key):
    baseline_rows = []
    recovery_rows = []
    reference_ids = None
    for baseline_rows_for_seed, recovery_rows_for_seed in zip(
        baseline_row_sets, recovery_row_sets
    ):
        baseline_by_id = _rows_by_id(baseline_rows_for_seed, id_key)
        recovery_by_id = _rows_by_id(recovery_rows_for_seed, id_key)
        if set(baseline_by_id) != set(recovery_by_id):
            raise ValueError("Mismatched row IDs within a training seed")
        ids = sorted(baseline_by_id)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("Mismatched row IDs across training seeds")
        baseline_rows.append([baseline_by_id[key] for key in ids])
        recovery_rows.append([recovery_by_id[key] for key in ids])
    return baseline_rows, recovery_rows


def _aggregate_ranking(baseline_reports, recovery_reports, samples, seed):
    baseline_rows, recovery_rows = _paired_row_matrices(
        [report["counterfactual_ranking"]["per_pair"] for report in baseline_reports],
        [report["counterfactual_ranking"]["per_pair"] for report in recovery_reports],
        id_key="scenario_id",
    )
    top1_delta = np.asarray(
        [
            [float(r["recovery_top1"]) - float(b["recovery_top1"])
             for b, r in zip(baseline_seed, recovery_seed)]
            for baseline_seed, recovery_seed in zip(baseline_rows, recovery_rows)
        ]
    )
    margin_delta = np.asarray(
        [
            [r["recovery_margin"] - b["recovery_margin"]
             for b, r in zip(baseline_seed, recovery_seed)]
            for baseline_seed, recovery_seed in zip(baseline_rows, recovery_rows)
        ]
    )
    regret_reduction = np.asarray(
        [
            [b["selection_regret"] - r["selection_regret"]
             for b, r in zip(baseline_seed, recovery_seed)]
            for baseline_seed, recovery_seed in zip(baseline_rows, recovery_rows)
        ]
    )
    baseline_accuracy = [
        report["counterfactual_ranking"]["recovery_top1_accuracy"]
        for report in baseline_reports
    ]
    recovery_accuracy = [
        report["counterfactual_ranking"]["recovery_top1_accuracy"]
        for report in recovery_reports
    ]
    return {
        "recovery_top1_accuracy": {
            "baseline_mean": float(np.mean(baseline_accuracy)),
            "recovery_rich_mean": float(np.mean(recovery_accuracy)),
            "baseline_per_seed": baseline_accuracy,
            "recovery_rich_per_seed": recovery_accuracy,
        },
        "paired_top1_accuracy_delta": crossed_paired_bootstrap(
            top1_delta, samples, seed
        ),
        "paired_margin_delta": crossed_paired_bootstrap(
            margin_delta, samples, seed + 1
        ),
        "paired_selection_regret_reduction": crossed_paired_bootstrap(
            regret_reduction, samples, seed + 2
        ),
    }


def _aggregate_prediction(baseline_reports, recovery_reports, samples, seed):
    horizons = sorted(baseline_reports[0]["prediction"], key=int)
    result = {}
    offset = 0
    for horizon in horizons:
        metrics = baseline_reports[0]["prediction"][horizon]
        result[horizon] = {}
        for metric in metrics:
            baseline_values = [
                report["prediction"][horizon][metric]
                for report in baseline_reports
            ]
            recovery_values = [
                report["prediction"][horizon][metric]
                for report in recovery_reports
            ]
            if metric == "sample_count":
                if len(set(baseline_values + recovery_values)) != 1:
                    raise ValueError("Prediction reports used different sample counts")
                result[horizon][metric] = baseline_values[0]
                continue
            deltas = np.asarray(baseline_values) - np.asarray(recovery_values)
            delta_summary = seed_bootstrap(deltas, samples, seed + 10 + offset)
            offset += 1
            result[horizon][metric] = {
                "baseline_mean": float(np.mean(baseline_values)),
                "recovery_rich_mean": float(np.mean(recovery_values)),
                "baseline_per_seed": baseline_values,
                "recovery_rich_per_seed": recovery_values,
                "reduction_per_seed": deltas.tolist(),
                "mean_reduction": delta_summary["mean"],
                "seed_bootstrap_95_ci": delta_summary["seed_bootstrap_95_ci"],
            }
    return result


def _aggregate_closed_loop(baseline_reports, recovery_reports, samples, seed):
    baseline_rows, recovery_rows = _paired_row_matrices(
        [report["closed_loop_recovery"]["per_scenario"] for report in baseline_reports],
        [report["closed_loop_recovery"]["per_scenario"] for report in recovery_reports],
        id_key="scenario_id",
    )

    def matrix(metric, sign=1.0):
        return np.asarray(
            [
                [sign * (float(r[metric]) - float(b[metric]))
                 for b, r in zip(baseline_seed, recovery_seed)]
                for baseline_seed, recovery_seed in zip(baseline_rows, recovery_rows)
            ]
        )

    def levels(metric):
        baseline = [report["closed_loop_recovery"][metric] for report in baseline_reports]
        recovery = [report["closed_loop_recovery"][metric] for report in recovery_reports]
        return {
            "baseline_mean": float(np.mean(baseline)),
            "recovery_rich_mean": float(np.mean(recovery)),
            "baseline_per_seed": baseline,
            "recovery_rich_per_seed": recovery,
        }

    success_delta = matrix("success")
    return {
        "success_rate": {
            **levels("success_rate"),
            "baseline_total_successes": int(sum(
                row["success"] for seed_rows in baseline_rows for row in seed_rows
            )),
            "recovery_rich_total_successes": int(sum(
                row["success"] for seed_rows in recovery_rows for row in seed_rows
            )),
        },
        "final_coverage": levels("mean_final_coverage"),
        "max_coverage": levels("mean_max_coverage"),
        "action_cost": levels("mean_action_cost"),
        "paired_success_rate_delta": crossed_paired_bootstrap(
            success_delta, samples, seed + 100
        ),
        "paired_final_coverage_delta": crossed_paired_bootstrap(
            matrix("final_coverage"), samples, seed + 101
        ),
        "paired_max_coverage_delta": crossed_paired_bootstrap(
            matrix("max_coverage"), samples, seed + 102
        ),
        "paired_action_cost_delta": crossed_paired_bootstrap(
            matrix("action_cost"), samples, seed + 103
        ),
    }


def aggregate_run(
    run_dir,
    seeds,
    baseline_variant="D_SF_balanced",
    recovery_variant="D_SFR_balanced",
    closed_loop_name="closed_loop_short_shaped.json",
    bootstrap_samples=10000,
    bootstrap_seed=0,
):
    run_dir = Path(run_dir)
    baseline_evaluations = []
    recovery_evaluations = []
    baseline_closed_loop = []
    recovery_closed_loop = []
    for training_seed in seeds:
        baseline_dir = run_dir / baseline_variant / f"seed_{training_seed}"
        recovery_dir = run_dir / recovery_variant / f"seed_{training_seed}"
        baseline_evaluation = _read(baseline_dir / "evaluation.json")
        recovery_evaluation = _read(recovery_dir / "evaluation.json")
        if baseline_evaluation["training_seed"] != training_seed:
            raise ValueError("Baseline report training seed mismatch")
        if recovery_evaluation["training_seed"] != training_seed:
            raise ValueError("Recovery-rich report training seed mismatch")
        baseline_evaluations.append(baseline_evaluation)
        recovery_evaluations.append(recovery_evaluation)
        baseline_closed_loop.append(_read(baseline_dir / closed_loop_name))
        recovery_closed_loop.append(_read(recovery_dir / closed_loop_name))
    return {
        "run_dir": str(run_dir.resolve()),
        "training_seeds": list(seeds),
        "baseline_variant": baseline_variant,
        "recovery_rich_variant": recovery_variant,
        "bootstrap": {
            "method": "crossed resampling of training seeds and scenario IDs",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "counterfactual_ranking": _aggregate_ranking(
            baseline_evaluations,
            recovery_evaluations,
            bootstrap_samples,
            bootstrap_seed,
        ),
        "prediction": _aggregate_prediction(
            baseline_evaluations,
            recovery_evaluations,
            bootstrap_samples,
            bootstrap_seed,
        ),
        "closed_loop_recovery": _aggregate_closed_loop(
            baseline_closed_loop,
            recovery_closed_loop,
            bootstrap_samples,
            bootstrap_seed,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--baseline-variant", default="D_SF_balanced")
    parser.add_argument("--recovery-variant", default="D_SFR_balanced")
    parser.add_argument("--closed-loop-name", default="closed_loop_short_shaped.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = aggregate_run(
        args.run_dir,
        args.seeds,
        baseline_variant=args.baseline_variant,
        recovery_variant=args.recovery_variant,
        closed_loop_name=args.closed_loop_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    dump_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
