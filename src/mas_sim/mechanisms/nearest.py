from math import dist

from .base import Context
from ..domain import Task


class Nearest:
    batched = False

    def allocate(self, context: Context, tasks: list[Task], time: float):
        for task in tasks:
            for agent in context.agents:
                context.ledger.record("position_query", "manager", agent.id, task.id)
                context.ledger.record("position_response", agent.id, "manager", task.id)
            winner = min(context.agents, key=lambda a: (dist(a.position, task.position), a.id))
            context.assign(winner, task, time)
