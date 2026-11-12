from .base import Context
from ..domain import Task, bid


class Greedy:
    batched = False

    def allocate(self, context: Context, tasks: list[Task], time: float):
        for task in tasks:
            for agent in context.agents:
                context.ledger.record("centralized_full_query", "manager", agent.id, task.id)
                context.ledger.record("centralized_full_response", agent.id, "manager", task.id)
            winner = min(context.agents, key=lambda a: (bid(a, task, time), a.id))
            context.assign(winner, task, time)
