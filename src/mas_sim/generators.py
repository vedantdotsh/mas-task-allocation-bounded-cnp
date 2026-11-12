"""Generators"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .models import Agent, Task

ArrivalMode = Literal["poisson", "uniform_fixed"]



def generate_tasks(

    expected_tasks: int,

    n_skills: int,

    horizon: float,

    workspace: float,

    rng: np.random.Generator,

    *,

    arrival_mode: ArrivalMode = "poisson",
    force_nonempty: bool = True,

) -> list[Task]:

    if expected_tasks <= 0:

        return []

    if arrival_mode == "poisson":

        rate = expected_tasks / horizon

        arrivals: list[float] = []

        t = 0.0

        while True:

            t += float(rng.exponential(1.0 / rate))

            if t > horizon:

                break

            arrivals.append(t)

        if not arrivals and force_nonempty:

            arrivals.append(float(rng.uniform(0, horizon)))

        arrival_times = np.array(arrivals, dtype=float)

    elif arrival_mode == "uniform_fixed":

        arrival_times = np.sort(rng.uniform(0, horizon, expected_tasks))

    else:

        raise ValueError(f"Unknown arrival_mode: {arrival_mode}")

    tasks: list[Task] = []

    for i, arrival_time in enumerate(arrival_times):

        tasks.append(

            Task(

                task_id=i,

                skill_required=int(rng.integers(0, n_skills)),

                workload=float(rng.uniform(2.0, 10.0)),

                position=rng.uniform(0, workspace, 2),

                arrival_time=float(arrival_time),

            )

        )

    return tasks



def generate_agents(

    n_agents: int,

    n_skills: int,

    workspace: float,

    rng: np.random.Generator,

) -> list[Agent]:

    agents: list[Agent] = []

    for i in range(n_agents):

        skills = rng.uniform(0.2, 0.5, n_skills)

        n_strong = int(rng.integers(1, min(3, n_skills + 1)))

        strong_idx = rng.choice(n_skills, size=n_strong, replace=False)

        skills[strong_idx] = rng.uniform(0.8, 1.2, n_strong)

        agents.append(

            Agent(

                agent_id=i,

                position=rng.uniform(0, workspace, 2),

                skills=skills,

                speed=float(rng.uniform(0.8, 1.3)),

            )

        )

    return agents
