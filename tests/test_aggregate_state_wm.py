import json

import numpy as np

from phase2.aggregate import aggregate_run, crossed_paired_bootstrap


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(training_seed, recovery_rich):
    offset = 1.0 if recovery_rich else 0.0
    prediction = {
        "1": {
            "sample_count": 2,
            "object_goal_position_rmse_px": 1.0 if recovery_rich else 3.0,
        }
    }
    return {
        "training_seed": training_seed,
        "counterfactual_ranking": {
            "recovery_top1_accuracy": 1.0 if recovery_rich else 0.5,
            "per_pair": [
                {
                    "scenario_id": "a",
                    "recovery_top1": recovery_rich,
                    "recovery_margin": offset,
                    "selection_regret": 0.0 if recovery_rich else 1.0,
                },
                {
                    "scenario_id": "b",
                    "recovery_top1": True,
                    "recovery_margin": 1.0 + offset,
                    "selection_regret": 0.0,
                },
            ],
        },
        "prediction": prediction,
        "prediction_by_branch": {"R": prediction},
        "recovery_prefix_prediction": prediction,
    }


def _closed_loop(training_seed, recovery_rich):
    coverage = 0.8 if recovery_rich else 0.4
    retention_loss = 0.05 if recovery_rich else 0.2
    rows = [
        {
            "scenario_id": scenario_id,
            "success": recovery_rich,
            "final_coverage": coverage,
            "max_coverage": coverage + 0.1,
            "coverage_retention_loss": retention_loss,
            "action_cost": 1.0 if recovery_rich else 2.0,
        }
        for scenario_id in ("a", "b")
    ]
    return {
        "training_seed": training_seed,
        "closed_loop_recovery": {
            "success_rate": float(recovery_rich),
            "mean_final_coverage": coverage,
            "mean_max_coverage": coverage + 0.1,
            "mean_coverage_retention_loss": retention_loss,
            "mean_action_cost": 1.0 if recovery_rich else 2.0,
            "per_scenario": rows,
        },
    }


def test_crossed_bootstrap_constant_matrix():
    result = crossed_paired_bootstrap(np.ones((3, 4)), samples=100, seed=7)
    assert result["mean"] == 1.0
    assert result["crossed_bootstrap_95_ci"] == [1.0, 1.0]


def test_aggregate_run(tmp_path):
    for training_seed in (0, 1):
        for variant, recovery_rich in (
            ("D_SF_balanced", False),
            ("D_SFR_balanced", True),
        ):
            seed_dir = tmp_path / variant / f"seed_{training_seed}"
            _write(seed_dir / "evaluation.json", _evaluation(training_seed, recovery_rich))
            _write(
                seed_dir / "closed_loop_short_shaped.json",
                _closed_loop(training_seed, recovery_rich),
            )
    report = aggregate_run(tmp_path, [0, 1], bootstrap_samples=100, bootstrap_seed=3)
    ranking = report["counterfactual_ranking"]
    assert ranking["paired_top1_accuracy_delta"]["mean"] == 0.5
    assert ranking["paired_selection_regret_reduction"]["mean"] == 0.5
    prediction = report["prediction"]["1"]["object_goal_position_rmse_px"]
    assert prediction["mean_reduction"] == 2.0
    branch_prediction = report["prediction_by_branch"]["R"]["1"]
    assert branch_prediction["object_goal_position_rmse_px"]["mean_reduction"] == 2.0
    prefix_prediction = report["recovery_prefix_prediction"]["1"]
    assert prefix_prediction["object_goal_position_rmse_px"]["mean_reduction"] == 2.0
    closed_loop = report["closed_loop_recovery"]
    assert closed_loop["success_rate"]["baseline_total_successes"] == 0
    assert closed_loop["success_rate"]["recovery_rich_total_successes"] == 4
    assert closed_loop["paired_final_coverage_delta"]["mean"] == 0.4
    assert np.isclose(
        closed_loop["paired_coverage_retention_loss_delta"]["mean"], -0.15
    )
