"""Reproducibility Tests"""

from __future__ import annotations

from mas_sim.experiments import ExperimentConfig, run_single_trial



def test_single_trial_is_reproducible_for_seed() -> None:

    config = ExperimentConfig(n_trials=1, n_agents=6, expected_tasks=25, k_invite=3)

    assert run_single_trial(config, seed=7) == run_single_trial(config, seed=7)
