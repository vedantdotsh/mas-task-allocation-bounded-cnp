from .base import Context
from ..domain import Task, bid


class SequentialSingleItem:
    batched = True

    def allocate(self, context: Context, tasks: list[Task], time: float):
        # One announcement per task-agent, then one vector bid per agent per round.
        for task in tasks:
            for agent in context.agents:
                context.ledger.record("cfp", "manager", agent.id, task.id)
        remaining = list(tasks)
        round_id = 0
        while remaining:
            task_ids = tuple(t.id for t in remaining)
            offers = []
            for agent in context.agents:
                context.ledger.record("bid", agent.id, "manager", task_ids, round_id,
                                      payload={'bids': [{'task_id': t.id, 'cost': bid(agent, t, time)} for t in remaining]})
                offers.extend((bid(agent, task, time), task.id, agent.id, task, agent) for task in remaining)
            _, _, _, task, agent = min(offers, key=lambda offer: offer[:3])
            context.ledger.record("award", "manager", agent.id, task.id, round_id)
            context.assign(agent, task, time)
            remaining.remove(task)
            round_id += 1
