"""Coordination layer: allocation mechanisms under comparison.

Every mechanism consumes an identical cloned team and task stream and
returns the same metrics dictionary, so outcome differences are
attributable to the coordination mechanism alone.
"""

from __future__ import annotations

import numpy as np

from .core import Agent, Task, schedule_task, tasks_with_private_work
from .llm_proxy import (
    AWARD_TOKENS,
    BID_TOKENS,
    CENTRAL_QUERY_TOKENS,
    CENTRAL_RESPONSE_TOKENS,
    CFP_TOKENS,
)
from .metrics import collect_metrics

CBBA_MSG_TOKENS = 60  # winning-bid list broadcast (structured, per round)


def run_round_robin(agents: list[Agent], tasks: list[Task], **_) -> dict:
    """No-coordination baseline: strict rotation, zero messages."""
    for i, task in enumerate(tasks):
        schedule_task(agents[i % len(agents)], task)
    return collect_metrics(agents, tasks, 0, 0.0)


def run_nearest_agent(agents: list[Agent], tasks: list[Task], **_) -> dict:
    """Spatial-only baseline: closest agent by queue-end position.

    Charged like a centralized position query: the dispatcher polls every
    agent's position (query + response) for each task.
    """
    total_messages = 0
    token_units = 0.0
    for task in tasks:
        total_messages += 2 * len(agents)
        token_units += len(agents) * (CENTRAL_QUERY_TOKENS + CENTRAL_RESPONSE_TOKENS)
        winner = min(
            agents,
            key=lambda a: np.linalg.norm(a.committed_position - task.position),
        )
        schedule_task(winner, task)
    return collect_metrics(agents, tasks, total_messages, token_units)


def run_greedy_centralized(agents: list[Agent], tasks: list[Task],
                         private_jobs=(), count_awards: bool = False, **_) -> dict:
    """Omniscient myopic oracle: queries all n agents per task (Eq. 3)."""
    total_messages = 0
    token_units = 0.0
    for task in tasks_with_private_work(agents, tasks, private_jobs):
        total_messages += 2 * len(agents) + int(count_awards)
        token_units += len(agents) * (CENTRAL_QUERY_TOKENS + CENTRAL_RESPONSE_TOKENS)
        token_units += AWARD_TOKENS if count_awards else 0
        winner = min(agents, key=lambda a: a.estimate_cost(task, task.arrival_time))
        schedule_task(winner, task)
    return collect_metrics(agents, tasks, total_messages, token_units)


def run_state_tracking_manager(agents: list[Agent], tasks: list[Task],
                               private_jobs=(), refresh_every: int | None = None,
                               **_) -> dict:
    """Predict queue ends from awards; optionally receive n status reports.

    Reports arrive before every s-th subsequent public task (indices s, 2s,
    ...); initial state is known. No completion reports or queries are used.
    """
    if refresh_every is not None and refresh_every < 1:
        raise ValueError("refresh_every must be positive or None")
    records = [Agent(a.agent_id, a.position.copy(), a.skills.copy(), a.speed,
                     busy_until=a.busy_until,
                     committed_position=a.committed_position.copy()) for a in agents]
    actual = {a.agent_id: a for a in agents}
    reports = 0
    for index, task in enumerate(tasks_with_private_work(agents, tasks, private_jobs)):
        if refresh_every is not None and index > 0 and index % refresh_every == 0:
            for record in records:
                agent = actual[record.agent_id]
                record.busy_until = agent.busy_until
                record.committed_position = agent.committed_position.copy()
            reports += len(agents)
        record = min(records, key=lambda a: a.estimate_cost(task, task.arrival_time))
        predicted = Task(task.task_id, task.arrival_time, task.skill_type,
                         task.workload, task.position)
        schedule_task(record, predicted)
        schedule_task(actual[record.agent_id], task)
    return collect_metrics(agents, tasks, len(tasks) + reports,
                           len(tasks) * AWARD_TOKENS + reports * CENTRAL_RESPONSE_TOKENS,
                           extra={"status_reports": reports})


def k_nearest_agents(agents: list[Agent], task: Task, k: int) -> list[Agent]:
    order = sorted(
        agents,
        key=lambda a: np.linalg.norm(a.committed_position - task.position),
    )
    return order[: min(k, len(agents))]


def run_contract_net(
    agents: list[Agent],
    tasks: list[Task],
    k_invite: int = 5,
    rng: np.random.Generator | None = None,
    bid_noise: float = 0.0,
    nonresponse_prob: float = 0.0,
    rebid: bool = False,
    rebid_threshold: float = 0.25,
    max_rounds: int = 2,
    private_jobs=(),
    **_,
) -> dict:
    """Manager-mediated bounded Contract Net Protocol (Smith, 1980).

    Each task's CFP goes only to the k nearest candidates. Optional
    robustness modes: multiplicative Gaussian bid noise, candidate
    nonresponse, and multi-round re-bidding when the awarded bid proves
    unreliable (realized cost exceeds the quoted bid by *rebid_threshold*).
    """
    rng = rng or np.random.default_rng()
    total_messages = 0
    token_units = 0.0
    total_rebids = 0
    no_responder_fallbacks = 0

    def solicit(task: Task) -> list[tuple[float, float, Agent]]:
        """One CFP round: returns (quoted_bid, true_cost, agent) tuples."""
        nonlocal total_messages, token_units
        candidates = k_nearest_agents(agents, task, k_invite)
        total_messages += len(candidates)  # CFP announcements
        token_units += len(candidates) * CFP_TOKENS
        bids = []
        for agent in candidates:
            if nonresponse_prob > 0 and rng.random() < nonresponse_prob:
                continue  # candidate silently drops the CFP
            true_cost = agent.estimate_cost(task, task.arrival_time)
            quoted = true_cost * max(0.05, 1.0 + rng.normal(0.0, bid_noise)) if bid_noise > 0 else true_cost
            bids.append((quoted, true_cost, agent))
            total_messages += 1  # bid response
            token_units += BID_TOKENS
        return bids

    for task in tasks_with_private_work(agents, tasks, private_jobs):
        rounds = 0
        winner = None
        while rounds < max_rounds:
            rounds += 1
            bids = solicit(task)
            if not bids:
                # Published rule: uncharged true-cost lookup among invitees.
                no_responder_fallbacks += 1
                bids = [
                    (a.estimate_cost(task, task.arrival_time),) * 2 + (a,)
                    for a in k_nearest_agents(agents, task, k_invite)
                ]
            quoted, true_cost, candidate = min(bids, key=lambda b: b[0])
            total_messages += 1  # award message
            token_units += AWARD_TOKENS
            winner = candidate
            # Re-bid only if enabled and the quote proved unreliable.
            if not (rebid and true_cost > quoted * (1.0 + rebid_threshold)):
                break
            total_rebids += 1
        schedule_task(winner, task)

    return collect_metrics(
        agents, tasks, total_messages, token_units,
        extra={"rebids": total_rebids, "no_responder_fallbacks": no_responder_fallbacks}
    )


def run_ssi_auction(
    agents: list[Agent],
    tasks: list[Task],
    window: float = 25.0,
    **_,
) -> dict:
    """Sequential single-item auction over arrival-time batches.

    Tasks arriving within each time window are auctioned as a batch: in
    every round all agents bid on every unassigned task, and the single
    globally cheapest (task, agent) pair is awarded. This is the classic
    SSI auction (Koenig et al.) adapted to an online stream.

    Decision time is the last arrival in each offline-grouped window.
    Each agent sends its cheapest remaining (task, cost) bid per round.
    Message accounting per batch of size B with n agents: B*n batch
    announcements, then one bid per agent per round (n*B) plus one award
    per round (B).
    """
    n = len(agents)
    total_messages = 0
    token_units = 0.0

    batches: list[list[Task]] = []
    current: list[Task] = []
    boundary = window
    for task in tasks:
        while task.arrival_time > boundary:
            if current:
                batches.append(current)
                current = []
            boundary += window
        current.append(task)
    if current:
        batches.append(current)

    for batch in batches:
        total_messages += len(batch) * n  # batch announcement broadcast
        token_units += len(batch) * n * CFP_TOKENS
        unassigned = list(batch)
        now = max(t.arrival_time for t in batch)
        while unassigned:
            best = None
            for task in unassigned:
                for agent in agents:
                    cost = agent.estimate_cost(task, now)
                    if best is None or cost < best[0]:
                        best = (cost, task, agent)
            total_messages += n + 1  # one bid per agent + award
            token_units += n * BID_TOKENS + AWARD_TOKENS
            _, task, agent = best
            schedule_task(agent, task, decision_time=now)
            unassigned.remove(task)

    return collect_metrics(agents, tasks, total_messages, token_units)


def run_cbba(
    agents: list[Agent],
    tasks: list[Task],
    window: float = 25.0,
    max_consensus_rounds: int = 50,
    **_,
) -> dict:
    """Consensus-Based Bundle Algorithm (Choi, Brunet & How, 2009), batch mode.

    Tasks arriving within each window are allocated with a sequential shared-table CBBA adaptation:
    agents greedily build bundles of tasks whose current winning bid they
    can beat, then broadcast winning-bid lists until consensus. We assume a
    fully connected network. One ordered sweep is counted as a round.
    Message accounting: n*(n-1) broadcasts per consensus round.
    """
    n = len(agents)
    total_messages = 0
    token_units = 0.0

    batches: list[list[Task]] = []
    current: list[Task] = []
    boundary = window
    for task in tasks:
        while task.arrival_time > boundary:
            if current:
                batches.append(current)
                current = []
            boundary += window
        current.append(task)
    if current:
        batches.append(current)

    for batch in batches:
        now = max(t.arrival_time for t in batch)
        winners: dict[int, tuple[float, int]] = {}
        bundles = {a.agent_id: [] for a in agents}
        rounds = 0
        changed = True
        while changed and rounds < max_consensus_rounds:
            changed = False
            rounds += 1
            for agent in agents:
                bundle = bundles[agent.agent_id]
                # Losing a prefix invalidates all dependent claims after it.
                for index, task in enumerate(bundle):
                    if winners.get(task.task_id, (0, -1))[1] != agent.agent_id:
                        for dropped in bundle[index:]:
                            if winners.get(dropped.task_id, (0, -1))[1] == agent.agent_id:
                                del winners[dropped.task_id]
                        del bundle[index:]
                        changed = True
                        break
                sim_pos = agent.committed_position.copy()
                sim_busy = max(now, agent.busy_until)
                # Retained tasks consume queue time before any new bid.
                for task in bundle:
                    sim_busy += (float(np.linalg.norm(sim_pos - task.position) / agent.speed)
                                 + agent.execution_time(task))
                    sim_pos = task.position
                while True:
                    best = None
                    for task in batch:
                        if task in bundle:
                            continue
                        duration = (float(np.linalg.norm(sim_pos - task.position) / agent.speed)
                                    + agent.execution_time(task))
                        cost = sim_busy - now + duration
                        held = winners.get(task.task_id)
                        if held is not None and cost >= held[0] - 1e-9:
                            continue
                        if best is None or cost < best[0]:
                            best = (cost, task, duration)
                    if best is None:
                        break
                    cost, task, duration = best
                    winners[task.task_id] = (cost, agent.agent_id)
                    bundle.append(task)
                    sim_busy += duration
                    sim_pos = task.position
                    changed = True
            total_messages += n * (n - 1)
            token_units += n * (n - 1) * CBBA_MSG_TOKENS

        if changed:
            raise RuntimeError("CBBA did not converge within max_consensus_rounds")
        agent_by_id = {a.agent_id: a for a in agents}
        for agent_id, bundle in bundles.items():
            for task in bundle:
                if winners.get(task.task_id, (0, -1))[1] == agent_id:
                    schedule_task(agent_by_id[agent_id], task, decision_time=now)
        if any(task.assigned_agent < 0 for task in batch):
            raise RuntimeError("CBBA converged with an unassigned task")

    return collect_metrics(agents, tasks, total_messages, token_units)
