"""Metrics Tests"""

from __future__ import annotations

import numpy as np

from mas_sim.metrics import jains_fairness

from mas_sim.models import Agent, Task

from mas_sim.strategies import run_round_robin



def test_jains_fairness_balanced_and_unbalanced() -> None:

    assert jains_fairness(np.array([1.0, 1.0, 1.0])) == 1.0

    assert jains_fairness(np.array([3.0, 0.0, 0.0])) < 0.34



def test_busy_time_includes_travel_and_execution() -> None:

    agent = Agent(agent_id=0, position=np.array([0.0, 0.0]), skills=np.array([1.0]), speed=1.0)

    task = Task(task_id=0, skill_required=0, workload=2.0, position=np.array([3.0, 4.0]), arrival_time=0.0)

    result = run_round_robin([agent], [task])

    assert result["loads"] == [7.0]

    assert result["fairness"] == 1.0
