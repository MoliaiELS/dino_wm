"""Aggregate Experiment B probes across training seeds and paired scenarios."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase2.aggregate import crossed_paired_bootstrap
from phase2.evaluation import dump_json


def _read(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _levels(reports, probe, metric):
    values = [report[probe]["test"][metric] for report in reports]
    return {"mean": float(np.mean(values)), "per_seed": values}


def _scenario_matrix(baseline_reports, recovery_reports, probe, error_fn):
    seed_rows = []
    reference_ids = None
    for baseline, recovery in zip(baseline_reports, recovery_reports):
        if probe == "task_progress":
            target_key, prediction_key = "test_targets", "test_predictions"
        else:
            target_key, prediction_key = "test_targets", "test_probabilities"
        baseline_targets = np.asarray(baseline[probe][target_key], dtype=np.float64)
        recovery_targets = np.asarray(recovery[probe][target_key], dtype=np.float64)
        baseline_predictions = np.asarray(
            baseline[probe][prediction_key], dtype=np.float64
        )
        recovery_predictions = np.asarray(
            recovery[probe][prediction_key], dtype=np.float64
        )
        baseline_ids = np.asarray(baseline[probe]["test_scenario_ids"])
        recovery_ids = np.asarray(recovery[probe]["test_scenario_ids"])
        if not np.array_equal(baseline_ids, recovery_ids) or not np.allclose(
            baseline_targets, recovery_targets
        ):
            raise ValueError(f"Unpaired {probe} test rows")
        grouped = defaultdict(list)
        for scenario_id, target, baseline_prediction, recovery_prediction in zip(
            baseline_ids,
            baseline_targets,
            baseline_predictions,
            recovery_predictions,
        ):
            grouped[str(scenario_id)].append(
                error_fn(target, baseline_prediction)
                - error_fn(target, recovery_prediction)
            )
        ids = sorted(grouped)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("Scenario IDs differ across visual training seeds")
        seed_rows.append([float(np.mean(grouped[key])) for key in ids])
    return np.asarray(seed_rows, dtype=np.float64)


def aggregate(baseline_reports, recovery_reports, raw_report, random_reports, oracle_report, samples, seed):
    report = {
        "training_seeds": [item["seed"] for item in baseline_reports],
        "baseline_variant": "D_SFN_balanced",
        "recovery_variant": "D_SFR_balanced",
        "bootstrap": {
            "method": "crossed resampling of training seeds and fresh scenario IDs",
            "samples": samples,
            "seed": seed,
        },
        "representations": {},
        "paired_recovery_effect": {},
    }
    metric_map = {
        "task_progress": ("mae", "r2", "spearman"),
        "off_nominal": ("balanced_accuracy", "roc_auc", "average_precision", "brier"),
        "recoverability": ("balanced_accuracy", "roc_auc", "average_precision", "brier"),
    }
    representation_reports = {
        "raw_dino": [raw_report],
        "random_adapter": random_reports,
        "temporal_sfn": baseline_reports,
        "temporal_sfr": recovery_reports,
        "oracle_state_mlp_upper": [oracle_report],
    }
    for name, reports in representation_reports.items():
        report["representations"][name] = {
            probe: {metric: _levels(reports, probe, metric) for metric in metrics}
            for probe, metrics in metric_map.items()
        }
    error_functions = {
        "task_progress": lambda target, prediction: abs(prediction - target),
        "off_nominal": lambda target, probability: (probability - target) ** 2,
        "recoverability": lambda target, probability: (probability - target) ** 2,
    }
    for offset, (probe, error_fn) in enumerate(error_functions.items()):
        matrix = _scenario_matrix(
            baseline_reports, recovery_reports, probe, error_fn
        )
        report["paired_recovery_effect"][probe] = {
            "error_reduction_definition": (
                "SFN absolute error minus SFR absolute error"
                if probe == "task_progress"
                else "SFN Brier error minus SFR Brier error"
            ),
            **crossed_paired_bootstrap(matrix, samples, seed + offset),
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    probe_dir = args.run_dir / "probes"
    baseline = [
        _read(probe_dir / "trained_adapter" / f"seed_{seed}_D_SFN_balanced.json")
        for seed in args.seeds
    ]
    recovery = [
        _read(probe_dir / "trained_adapter" / f"seed_{seed}_D_SFR_balanced.json")
        for seed in args.seeds
    ]
    random_reports = [
        _read(probe_dir / "random_adapter" / f"seed_{seed}_D_SFN_balanced.json")
        for seed in args.seeds
    ]
    report = aggregate(
        baseline,
        recovery,
        _read(probe_dir / "raw_dino" / "seed_0.json"),
        random_reports,
        _read(probe_dir / "oracle_state" / "seed_0.json"),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    dump_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
