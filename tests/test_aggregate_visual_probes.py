from aggregate_visual_probes import aggregate


def _report(seed, recovery=False):
    scenario_ids = ["a", "a", "b", "b"]
    targets = [0.0, 0.25, 0.75, 1.0]
    regression = targets if recovery else [0.2, 0.4, 0.6, 0.8]
    classes = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9] if recovery else [0.4, 0.4, 0.6, 0.6]
    classification_metrics = {
        "balanced_accuracy": 1.0,
        "roc_auc": 1.0,
        "average_precision": 1.0,
        "brier": 0.025 if recovery else 0.16,
    }
    return {
        "seed": seed,
        "task_progress": {
            "test": {"mae": 0.0 if recovery else 0.15, "r2": 1.0, "spearman": 1.0},
            "test_predictions": regression,
            "test_targets": targets,
            "test_scenario_ids": scenario_ids,
        },
        "off_nominal": {
            "test": classification_metrics,
            "test_probabilities": probabilities,
            "test_targets": classes,
            "test_scenario_ids": scenario_ids,
        },
        "recoverability": {
            "test": classification_metrics,
            "test_probabilities": probabilities,
            "test_targets": classes,
            "test_scenario_ids": scenario_ids,
        },
    }


def test_aggregate_visual_probe_paired_effects():
    baseline = [_report(0), _report(1)]
    recovery = [_report(0, True), _report(1, True)]
    result = aggregate(
        baseline,
        recovery,
        raw_report=_report(0),
        random_reports=baseline,
        oracle_report=_report(0, True),
        samples=100,
        seed=7,
    )
    assert result["representations"]["temporal_sfr"]["task_progress"]["mae"]["mean"] == 0.0
    assert "oracle_state_mlp_upper" in result["representations"]
    assert result["paired_recovery_effect"]["task_progress"]["mean"] > 0
    assert result["paired_recovery_effect"]["off_nominal"]["mean"] > 0
