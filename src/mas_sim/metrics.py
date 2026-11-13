"""Metrics layer: outcome measures computed identically for every mechanism."""

from __future__ import annotations

import numpy as np

from .core import Agent, Task
from .llm_proxy import proxy_metrics


def jains_fairness(loads: np.ndarray) -> float:
    """Jain's fairness index over per-agent total busy time (Eq. 4)."""
    if loads.size == 0 or np.all(loads == 0):
        return 1.0
    return float((np.sum(loads) ** 2) / (loads.size * np.sum(loads**2)))


def collect_metrics(
    agents: list[Agent],
    tasks: list[Task],
    total_messages: int,
    token_units: float,
    extra: dict | None = None,
) -> dict:
    finishes = np.array([t.finish_time for t in tasks])
    waits = np.array([t.wait_time for t in tasks])
    loads = np.array([a.total_busy_time for a in agents])
    makespan = float(finishes.max()) if finishes.size else 0.0
    out = {
        "makespan": makespan,
        "mean_wait": float(waits.mean()) if waits.size else 0.0,
        "p95_wait": float(np.percentile(waits, 95)) if waits.size else 0.0,
        "fairness": jains_fairness(loads),
        "messages": int(total_messages),
        "throughput": (len(tasks) / makespan) if makespan > 0 else 0.0,
        "n_tasks": len(tasks),
    }
    out.update(proxy_metrics(token_units, total_messages))
    if extra:
        out.update(extra)
    return out
