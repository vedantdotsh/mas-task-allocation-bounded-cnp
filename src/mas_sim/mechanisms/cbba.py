from copy import deepcopy
from math import inf

from .base import Context
from ..domain import Task, bid, execute


class CBBA:
    """Synchronous fully connected, append-only bundles with suffix release."""

    batched = True

    def allocate(self, context: Context, tasks: list[Task], time: float):
        agents = context.agents
        by_id = {t.id: t for t in tasks}
        task_ids = tuple(by_id)
        bundles = [[] for _ in agents]
        claims = [[] for _ in agents]
        winners = {task.id: (inf, len(agents)) for task in tasks}
        previous = None
        # Convergence failure is an error, never a silent allocation fallback.
        for round_id in range(4 * len(tasks) * len(agents) + 2):
            for agent in agents:
                bundle = bundles[agent.id]
                old_claims = claims[agent.id]
                cut = next((j for j, task_id in enumerate(bundle)
                            if winners[task_id][1] != agent.id), len(bundle))
                local = dict(winners)
                for task_id in bundle[cut:]:
                    if local[task_id][1] == agent.id:
                        local[task_id] = (inf, len(agents))
                del bundle[cut:]
                del old_claims[cut:]
                projected = deepcopy(agent)
                for task_id in bundle:
                    execute(projected, by_id[task_id], time)
                while True:
                    offers = [(bid(projected, task, time), task.id) for task in tasks
                              if task.id not in bundle
                              and (bid(projected, task, time), agent.id) < local[task.id]]
                    if not offers:
                        break
                    cost, task_id = min(offers)
                    bundle.append(task_id)
                    old_claims.append(cost)
                    local[task_id] = (cost, agent.id)
                    execute(projected, by_id[task_id], time)
            updated = {task.id: (inf, len(agents)) for task in tasks}
            for agent in agents:
                for task_id, cost in zip(bundles[agent.id], claims[agent.id]):
                    updated[task_id] = min(updated[task_id], (cost, agent.id))
                for receiver in agents:
                    if receiver.id != agent.id:
                        context.ledger.record("consensus", agent.id, receiver.id, task_ids, round_id,
                                              payload={'agent_id': agent.id, 'bundle': bundles[agent.id],
                                                       'claims': [{'task_id': t, 'cost': c} for t, c in zip(bundles[agent.id], claims[agent.id])],
                                                       'winners': [{'task_id': t, 'agent_id': a if a < len(agents) else None,
                                                                    'cost': c if c < inf else None} for t, (c, a) in winners.items()]})
            snapshot = tuple(tuple(zip(bundle, costs)) for bundle, costs in zip(bundles, claims))
            conflict_free = all(updated[t][1] == a.id for a in agents for t in bundles[a.id])
            covered = all(owner < len(agents) for _, owner in updated.values())
            winners = updated
            if snapshot == previous and conflict_free and covered:
                for agent in agents:
                    for task_id in bundles[agent.id]:
                        context.assign(agent, by_id[task_id], time)
                return
            previous = snapshot
        raise RuntimeError("CBBA failed to converge; inspect bundle release and score assumptions")
