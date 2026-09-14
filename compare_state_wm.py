"""Paired comparison of two Phase 2 evaluation reports."""

from __future__ import annotations

import argparse
import json

import numpy as np

from phase2.evaluation import dump_json


def _read(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def paired_bootstrap(values, samples=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("paired_bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "n_pairs": int(len(values)),
        "mean": float(values.mean()),
        "bootstrap_95_ci": np.quantile(
            bootstrap_means, [0.025, 0.975]
        ).tolist(),
    }


def _rows_by_id(rows, id_key):
    result = {row[id_key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {id_key} in evaluation rows")
    return result


def compare_ranking(baseline, recovery, bootstrap_samples, seed):
    baseline_rows = _rows_by_id(
        baseline["counterfactual_ranking"]["per_pair"], "scenario_id"
    )
    recovery_rows = _rows_by_id(
        recovery["counterfactual_ranking"]["per_pair"], "scenario_id"
    )
    if set(baseline_rows) != set(recovery_rows):
        raise ValueError("Ranking reports do not contain the same scenario pairs")
    ids = sorted(baseline_rows)
    top1_delta = [
        float(recovery_rows[key]["recovery_top1"])
        - float(baseline_rows[key]["recovery_top1"])
        for key in ids
    ]
    margin_delta = [
        recovery_rows[key]["recovery_margin"]
        - baseline_rows[key]["recovery_margin"]
        for key in ids
    ]
    regret_reduction = [
        baseline_rows[key]["selection_regret"]
        - recovery_rows[key]["selection_regret"]
        for key in ids
    ]
    return {
        "recovery_top1_accuracy": {
            "baseline": baseline["counterfactual_ranking"]["recovery_top1_accuracy"],
            "recovery_rich": recovery["counterfactual_ranking"]["recovery_top1_accuracy"],
        },
        "paired_top1_accuracy_delta": paired_bootstrap(
            top1_delta, bootstrap_samples, seed
        ),
        "paired_margin_delta": paired_bootstrap(
            margin_delta, bootstrap_samples, seed + 1
        ),
        "paired_selection_regret_reduction": paired_bootstrap(
            regret_reduction, bootstrap_samples, seed + 2
        ),
    }


def compare_prediction(baseline, recovery):
    result = {}
    for horizon in sorted(baseline["prediction"], key=int):
        if horizon not in recovery["prediction"]:
            raise ValueError(f"Missing prediction horizon {horizon}")
        result[horizon] = {}
        for metric, baseline_value in baseline["prediction"][horizon].items():
            recovery_value = recovery["prediction"][horizon][metric]
            if metric == "sample_count":
                if baseline_value != recovery_value:
                    raise ValueError("Prediction evaluations used different samples")
                result[horizon][metric] = baseline_value
            else:
                result[horizon][metric] = {
                    "baseline": baseline_value,
                    "recovery_rich": recovery_value,
                    "reduction": baseline_value - recovery_value,
                }
    return result


def compare_closed_loop(baseline, recovery, bootstrap_samples, seed):
    baseline_rows = _rows_by_id(
        baseline["closed_loop_recovery"]["per_scenario"], "scenario_id"
    )
    recovery_rows = _rows_by_id(
        recovery["closed_loop_recovery"]["per_scenario"], "scenario_id"
    )
    if set(baseline_rows) != set(recovery_rows):
        raise ValueError("Closed-loop reports do not contain the same scenarios")
    ids = sorted(baseline_rows)
    success_delta = [
        float(recovery_rows[key]["success"]) - float(baseline_rows[key]["success"])
        for key in ids
    ]
    coverage_delta = [
        recovery_rows[key]["final_coverage"] - baseline_rows[key]["final_coverage"]
        for key in ids
    ]
    action_cost_delta = [
        recovery_rows[key]["action_cost"] - baseline_rows[key]["action_cost"]
        for key in ids
    ]
    return {
        "success_rate": {
            "baseline": baseline["closed_loop_recovery"]["success_rate"],
            "recovery_rich": recovery["closed_loop_recovery"]["success_rate"],
        },
        "paired_success_rate_delta": paired_bootstrap(
            success_delta, bootstrap_samples, seed
        ),
        "paired_final_coverage_delta": paired_bootstrap(
            coverage_delta, bootstrap_samples, seed + 1
        ),
        "paired_action_cost_delta": paired_bootstrap(
            action_cost_delta, bootstrap_samples, seed + 2
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--recovery-rich", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = _read(args.baseline)
    recovery = _read(args.recovery_rich)
    report = {
        "baseline_report": args.baseline,
        "recovery_rich_report": args.recovery_rich,
        "baseline_variant": baseline["training_variant"],
        "recovery_rich_variant": recovery["training_variant"],
        "training_seed": baseline["training_seed"],
    }
    if baseline["training_seed"] != recovery["training_seed"]:
        raise ValueError("Reports must use the same training seed")
    if "prediction" in baseline and "prediction" in recovery:
        report["prediction"] = compare_prediction(baseline, recovery)
    if "counterfactual_ranking" in baseline and "counterfactual_ranking" in recovery:
        report["counterfactual_ranking"] = compare_ranking(
            baseline, recovery, args.bootstrap_samples, args.seed
        )
    if "closed_loop_recovery" in baseline and "closed_loop_recovery" in recovery:
        report["closed_loop_recovery"] = compare_closed_loop(
            baseline, recovery, args.bootstrap_samples, args.seed
        )
    dump_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
