"""Regression against the recovered artifact, not values copied from the paper."""
import json
from pathlib import Path

import numpy as np
import pytest

from mas_sim.experiments import ExperimentConfig, run_single_trial
from mas_sim.scenario import ScenarioConfig, generate_scenario
from mas_sim.simulation import simulate
from mas_sim.stats import _bootstrap_ci


def test_all_recovered_trials_match():
    expected = json.loads((Path(__file__).parents[1] / "outputs/recovered_original/results_raw.json").read_text())
    config = ExperimentConfig()
    for seed in range(30):
        actual = run_single_trial(config, seed)
        for mechanism, rows in expected.items():
            for metric, value in rows[seed].items():
                assert actual[mechanism][metric] == pytest.approx(value, abs=1e-10), (seed, mechanism, metric)


def test_ssi_vector_messages_and_batch_causality():
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=20, horizon=40), 7)
    for name in ("ssi", "cbba"):
        result = simulate(scenario, name)
        assert all(a.dispatch >= a.decision >= a.arrival for a in result.assignments)
        if name == "ssi":
            assert len(result.ledger.messages) == len(scenario.tasks) * 9
            assert any(len(m.task_ids) > 1 for m in result.ledger.messages if m.type == "bid")
        else:
            assert len(result.ledger.messages) % (4 * 3) == 0
            assert all(m.sender != m.receiver for m in result.ledger.messages)


def test_paired_bootstrap_constant_and_reproducible():
    result = _bootstrap_ci(np.array([2, 2, 2]))
    assert result["ci95"] == [2, 2]
    assert result["mean_difference"] == 2
    assert _bootstrap_ci([1, 4, -2]) == _bootstrap_ci([1, 4, -2])
