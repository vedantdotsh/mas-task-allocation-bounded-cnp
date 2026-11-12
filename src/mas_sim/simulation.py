from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby
from math import floor

import numpy as np

from .domain import Agent, Assignment
from .ledger import MessageLedger
from .ledger import PAPER_TOKEN_COSTS
from .mechanisms import MECHANISMS
from .mechanisms.base import Context, Mechanism
from .mechanisms.cnp import ContractNet
from .metrics import measure
from .scenario import Scenario


@dataclass
class SimulationResult:
    agents: list[Agent]
    assignments: list[Assignment]
    ledger: MessageLedger
    metrics: dict[str, float]
    rebid_rounds: int


def simulate(scenario: Scenario, mechanism: str | Mechanism, *, k: int = 5,
             batch_window: float = 25, noise: float = 0.0,
             nonresponse: float = 0.0, rebid: bool = False,
             message_delay_mean: float = 0.0, stale_position: bool = False,
             rebid_threshold_factor: float = 1.25, shortlist: str = 'capability',
             stale_every: int | None = 1, theta: float = 0.0, private_load: float = 0.0) -> SimulationResult:
    if k < 1 or batch_window <= 0 or noise < 0 or not 0 <= nonresponse <= 1:
        raise ValueError("Invalid mechanism settings")
    if scenario.config.version == 'v2':
        if rebid or stale_position or message_delay_mean:
            raise ValueError('Legacy rebidding/delay/stale switch is unavailable in v2')
        from .v2 import simulate_v2
        return simulate_v2(scenario, mechanism, k=k, batch_window=batch_window, noise=noise,
                           nonresponse=nonresponse, shortlist=shortlist, stale_every=stale_every,
                           theta=theta, private_load=private_load)
    policy = MECHANISMS[mechanism]() if isinstance(mechanism, str) else mechanism
    rng = np.random.default_rng(scenario.seed + (200_000 if rebid else 100_000))
    context = Context(deepcopy(scenario.agents), MessageLedger(), rng,
                      k=k, noise=noise, nonresponse=nonresponse, rebid=rebid,
                      message_delay_mean=message_delay_mean, stale_position=stale_position,
                      rebid_threshold_factor=rebid_threshold_factor)
    tasks = sorted(deepcopy(scenario.tasks), key=lambda task: (task.arrival, task.id))
    if policy.batched:
        for window, batch in groupby(tasks, key=lambda task: floor(task.arrival / batch_window)):
            time = min((window + 1) * batch_window, scenario.config.horizon)
            policy.allocate(context, list(batch), time)
    else:
        for task in tasks:
            policy.allocate(context, [task], task.arrival)
    is_cnp = isinstance(policy, ContractNet)
    metrics = measure(context.agents, context.assignments, context.ledger,
                      context.latency_units if is_cnp else None)
    metrics["rebid_rounds"] = context.rebid_rounds
    metrics["failed_bid_rounds"] = context.failed_bid_rounds
    metrics["paper_tokens"] = context.ledger.token_cost(PAPER_TOKEN_COSTS)
    return SimulationResult(context.agents, context.assignments, context.ledger, metrics, context.rebid_rounds)
