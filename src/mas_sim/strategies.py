"""Compatibility entry points for the recovered June API.

The six policy implementations live in mechanisms/ and share one scheduler.
"""
from dataclasses import dataclass

import numpy as np

from .domain import execute
from .ledger import MessageLedger
from .mechanisms import MECHANISMS
from .mechanisms.base import Context
from .metrics import measure


@dataclass(frozen=True)
class RobustnessConfig:
    bid_noise_std: float = 0.0
    message_delay_mean: float = 0.0
    agent_failure_rate: float = 0.0
    stale_position: bool = False


def clone_agents(agents):
    from .models import Agent
    return [Agent(a.id, a.position.copy(), a.skills.copy(), a.speed,
                  last_known_position=a.last_known_position.copy()) for a in agents]


def clone_tasks(tasks):
    from .models import Task
    return [Task(t.id, t.skill, t.workload, t.position.copy(), t.arrival) for t in tasks]


def _assign(agent, task, now):
    execute(agent, task, now)


def _run(name, agents, tasks, *, k_invite=5, rng=None, robustness=None,
         rebid=False, rebid_threshold_factor=1.35, max_rebid_rounds=1):
    if not agents:
        raise ValueError("At least one agent is required")
    robustness = robustness or RobustnessConfig()
    context = Context(agents, MessageLedger(), np.random.default_rng(0) if rng is None else rng,
                      k=k_invite, noise=robustness.bid_noise_std,
                      nonresponse=robustness.agent_failure_rate, rebid=rebid,
                      message_delay_mean=robustness.message_delay_mean,
                      stale_position=robustness.stale_position,
                      rebid_threshold_factor=rebid_threshold_factor, max_rebid_rounds=max_rebid_rounds)
    policy = MECHANISMS[name]()
    for task in tasks:
        policy.allocate(context, [task], task.arrival)
    result = measure(agents, context.assignments, context.ledger, context.latency_units if name == "cnp" else None)
    result["loads"] = [a.total_busy_time for a in agents]
    result["failed_bid_rounds"] = context.failed_bid_rounds
    result["rebids"] = context.rebid_rounds
    result["message_types"] = context.ledger.counts_by_type()
    result["messages_per_task"] = context.ledger.counts_by_task()
    return result


def run_round_robin(agents, tasks):
    return _run("round_robin", agents, tasks)


def run_nearest_agent(agents, tasks):
    return _run("nearest", agents, tasks)


def run_greedy_centralized(agents, tasks):
    return _run("greedy", agents, tasks)


def run_contract_net(agents, tasks, **kwargs):
    return _run("cnp", agents, tasks, **kwargs)


def run_contract_net_rebid(agents, tasks, **kwargs):
    return _run("cnp", agents, tasks, rebid=True, **kwargs)
