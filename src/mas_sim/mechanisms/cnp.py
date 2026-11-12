"""June CNP behavior with an event ledger; paper experiment uses threshold 1.25."""
from math import dist

from .base import Context
from ..domain import Task, bid


class ContractNet:
    batched = False

    def _round(self, context, task, now, excluded, round_id):
        candidates = sorted((a for a in context.agents if a.id not in excluded), key=lambda a: (
            dist(a.last_known_position if context.stale_position else a.position, task.position), a.id))[:context.k]
        for agent in candidates:
            context.ledger.record("cfp", "manager", agent.id, task.id, round_id)
        offers = []
        latency = 0.0
        for agent in candidates:
            if context.message_delay_mean > 0:
                latency += float(context.rng.exponential(context.message_delay_mean))
            if context.nonresponse > 0 and context.rng.random() < context.nonresponse:
                continue
            context.ledger.record("bid", agent.id, "manager", task.id, round_id)
            agent.messages_sent += 1
            quote = agent.estimate_cost(task, now + latency, bid_noise_std=context.noise,
                                        rng=context.rng, use_last_known_position=context.stale_position)
            offers.append((quote, agent.id, agent))
        context.latency_units += latency
        if not offers:
            context.failed_bid_rounds += 1
            return None, 0.0, latency
        quote, _, winner = min(offers, key=lambda offer: offer[:2])
        context.ledger.record("award", "manager", winner.id, task.id, round_id)
        return winner, quote, latency

    def allocate(self, context: Context, tasks: list[Task], time: float):
        for task in tasks:
            excluded = set()
            winner = None
            total_latency = 0.0
            attempts = context.max_rebid_rounds if context.rebid else 0
            for round_id in range(attempts + 1):
                candidate, quote, latency = self._round(context, task, time + total_latency, excluded, round_id)
                total_latency += latency
                if candidate is None:
                    if round_id < attempts:
                        context.rebid_rounds += 1
                        continue
                    break
                should_rebid = (round_id < attempts
                                and bid(candidate, task, time + total_latency) > quote * context.rebid_threshold_factor
                                and len(excluded) + context.k < len(context.agents))
                if should_rebid:
                    excluded.add(candidate.id)
                    context.rebid_rounds += 1
                    task.rebid_count += 1
                    continue
                winner = candidate
                break
            if winner is None:
                # Recovered behavior: omniscient free fallback, explicitly criticized in notes.
                remaining = [a for a in context.agents if a.id not in excluded]
                winner = min(remaining or context.agents, key=lambda a: (bid(a, task, time), a.id))
            context.assign(winner, task, time + total_latency)
