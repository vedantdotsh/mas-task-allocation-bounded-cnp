from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..domain import Agent, Assignment, Task, execute
from ..ledger import MessageLedger


@dataclass
class Context:
    agents: list[Agent]
    ledger: MessageLedger
    rng: np.random.Generator
    k: int = 5
    noise: float = 0.0
    nonresponse: float = 0.0
    rebid: bool = False
    assignments: list[Assignment] = field(default_factory=list)
    rebid_rounds: int = 0
    failed_bid_rounds: int = 0
    message_delay_mean: float = 0.0
    stale_position: bool = False
    rebid_threshold_factor: float = 1.25
    max_rebid_rounds: int = 1
    latency_units: float = 0.0

    def assign(self, agent: Agent, task: Task, time: float):
        self.assignments.append(execute(agent, task, time))


class Mechanism(Protocol):
    batched: bool

    def allocate(self, context: Context, tasks: list[Task], time: float) -> None: ...
