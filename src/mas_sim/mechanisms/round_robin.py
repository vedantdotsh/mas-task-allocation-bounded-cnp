from .base import Context
from ..domain import Task


class RoundRobin:
    batched = False

    def allocate(self, context: Context, tasks: list[Task], time: float):
        for task in tasks:
            context.assign(context.agents[len(context.assignments) % len(context.agents)], task, time)
