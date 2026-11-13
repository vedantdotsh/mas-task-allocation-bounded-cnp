"""Ground-truth optimum for small instances via exhaustive search.

In this testbed a schedule is fully determined by the assignment vector
(which agent serves each task, in arrival order), so the true optimum can
be found by enumerating all n^T assignments with branch-and-bound on
makespan. Feasible only for small instances (n <= 4, T <= ~12), which is
exactly how we use it: as an optimality-gap reference, mirroring the
MILP-reference methodology of recent MRTA work.
"""

from __future__ import annotations

import numpy as np

from .core import Agent, Task, schedule_task
from .metrics import collect_metrics


def run_exhaustive_optimal(agents: list[Agent], tasks: list[Task], **_) -> dict:
    n = len(agents)
    T = len(tasks)
    if n**T > 20_000_000:
        raise ValueError(f"Instance too large for exhaustive search: {n}^{T}")

    positions = [a.position.copy() for a in agents]
    speeds = [a.speed for a in agents]
    execs = [
        [t.workload / max(float(a.skills[t.skill_type]), 0.05) for a in agents]
        for t in tasks
    ]

    best_makespan = [np.inf]
    best_assignment = [None]

    # State per agent: (busy_until, position)
    def dfs(idx: int, busy: list[float], pos: list[np.ndarray], assign: list[int], cur_max: float):
        if cur_max >= best_makespan[0]:
            return  # bound: cannot improve
        if idx == T:
            best_makespan[0] = cur_max
            best_assignment[0] = assign.copy()
            return
        task = tasks[idx]
        # Order agents by finish time to reach good incumbents early.
        order = sorted(
            range(n),
            key=lambda i: max(task.arrival_time, busy[i])
            + np.linalg.norm(pos[i] - task.position) / speeds[i]
            + execs[idx][i],
        )
        for i in order:
            start = max(task.arrival_time, busy[i])
            travel = float(np.linalg.norm(pos[i] - task.position) / speeds[i])
            finish = start + travel + execs[idx][i]
            old_busy, old_pos = busy[i], pos[i]
            busy[i], pos[i] = finish, task.position
            assign.append(i)
            dfs(idx + 1, busy, pos, assign, max(cur_max, finish))
            assign.pop()
            busy[i], pos[i] = old_busy, old_pos

    dfs(0, [a.busy_until for a in agents], positions, [], 0.0)

    # Replay the optimal assignment on the real agent objects.
    for task, agent_idx in zip(tasks, best_assignment[0]):
        schedule_task(agents[agent_idx], task)
    return collect_metrics(agents, tasks, 0, 0.0)
