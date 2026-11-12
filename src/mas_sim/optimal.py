"""Exact offline makespan: enumerate subset routes, then branch-and-bound partitions.

Unlike an arrival-order assignment search, this also bounds batching mechanisms
which may reorder tasks. Limits are explicit: at most four agents and 11 tasks.
"""
from dataclasses import dataclass
from math import dist, inf

import numpy as np

from .domain import execution_time, travel_time
from .scenario import Scenario


@dataclass
class Optimum:
    makespan: float
    routes: dict[int, list[int]]
    search_nodes: int


def _subset_routes(agent, tasks):
    size = len(tasks)
    total = 1 << size
    finishes = np.full((total, size), inf)
    parents = np.full((total, size), -1, dtype=np.int16)
    execution = [execution_time(agent, task) for task in tasks]
    travel = [[dist(a.position, b.position) / agent.speed for b in tasks] for a in tasks]
    for j, task in enumerate(tasks):
        finishes[1 << j, j] = max(agent.busy_until, task.arrival) + travel_time(agent, task) + execution[j]
    for mask in range(1, total):
        ends = [j for j in range(size) if mask & (1 << j)]
        for j in ends:
            previous = mask ^ (1 << j)
            if not previous:
                continue
            value, parent = min((max(float(finishes[previous, k]), tasks[j].arrival)
                                  + travel[k][j] + execution[j], k) for k in ends if k != j)
            finishes[mask, j] = value
            parents[mask, j] = parent
    best = np.min(finishes, axis=1)
    best[0] = agent.busy_until
    return best, finishes, parents


def exact_optimum(scenario: Scenario) -> Optimum:
    agents, tasks = scenario.agents, scenario.tasks
    if len(agents) > 4 or len(tasks) > 11:
        raise ValueError("Exact search limited to four agents and eleven tasks")
    if not tasks:
        return Optimum(max(a.busy_until for a in agents), {a.id: [] for a in agents}, 1)
    tables = [_subset_routes(agent, tasks) for agent in agents]
    all_tasks = (1 << len(tasks)) - 1
    incumbent = inf
    best_partition = None
    seen = {}
    nodes = 0

    def search(index, remaining, current, partition):
        nonlocal incumbent, best_partition, nodes
        nodes += 1
        if current >= incumbent or current >= seen.get((index, remaining), inf):
            return
        seen[index, remaining] = current
        if index == len(agents) - 1:
            candidate = max(current, float(tables[index][0][remaining]))
            if candidate < incumbent:
                incumbent = candidate
                best_partition = partition + [remaining]
            return
        # Enumerate every subset for this agent, pruning dominated partial makespans.
        subset = remaining
        while True:
            value = max(current, float(tables[index][0][subset]))
            if value < incumbent:
                search(index + 1, remaining ^ subset, value, partition + [subset])
            if subset == 0:
                break
            subset = (subset - 1) & remaining

    search(0, all_tasks, 0.0, [])
    routes = {}
    for agent, mask, (_, finishes, parents) in zip(agents, best_partition, tables):
        route = []
        end = int(np.argmin(finishes[mask])) if mask else -1
        while mask:
            route.append(tasks[end].id)
            parent = int(parents[mask, end])
            mask ^= 1 << end
            end = parent
        routes[agent.id] = list(reversed(route))
    return Optimum(incumbent, routes, nodes)
