"""Message Tests"""

from __future__ import annotations

import numpy as np

from mas_sim.models import Agent, Task

from mas_sim.strategies import RobustnessConfig, run_contract_net, run_nearest_agent



def _agents(n: int) -> list[Agent]:

    return [

        Agent(

            agent_id=i,

            position=np.array([float(i), 0.0]),

            skills=np.array([1.0]),

            speed=1.0,

        )

        for i in range(n)

    ]



def _tasks(n: int) -> list[Task]:

    return [

        Task(

            task_id=i,

            skill_required=0,

            workload=1.0,

            position=np.array([float(i), 0.0]),

            arrival_time=float(i),

        )

        for i in range(n)

    ]



def test_contract_net_message_count_without_failures() -> None:

    result = run_contract_net(

        _agents(4),

        _tasks(3),

        k_invite=2,

        rng=np.random.default_rng(1),

        robustness=RobustnessConfig(),

    )

    assert result["messages"] == 3 * (2 + 2 + 1)

    assert result["message_count"] == result["messages"]



def test_nearest_agent_message_count_queries_all_agents() -> None:

    result = run_nearest_agent(_agents(4), _tasks(3))

    assert result["messages"] == 3 * 2 * 4

    assert result["estimated_token_cost"] > 0
