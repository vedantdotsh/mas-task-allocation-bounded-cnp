"""v2 information policies on the recovered shared FIFO execution model."""
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from math import dist, floor

import numpy as np

from .domain import bid, execute
from .generators import generate_tasks
from .mechanisms import MECHANISMS
from .mechanisms.base import Context
from .metrics import jain_index, measure
from .wire import WireLedger, agent_card

NAMES = {'round_robin': 'Round-robin', 'nearest': 'Nearest-agent',
         'greedy': 'Full-information greedy', 'cnp': 'Manager-mediated bounded CNP',
         'ssi': 'SSI auction', 'cbba': 'CBBA', 'cached_central': 'Cached central'}


class V2Context(Context):
    def __init__(self, agents, tasks, seed, *, mechanism, k, noise, nonresponse,
                 shortlist, stale_every, theta):
        ledger = WireLedger()
        super().__init__(agents, ledger, np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[2]),
                         k=k, noise=noise, nonresponse=nonresponse)
        self.mechanism, self.shortlist, self.stale_every, self.theta = mechanism, shortlist, stale_every, theta
        self.task_by_id = {t.id: t for t in tasks}
        self.history = defaultdict(list)
        self.awards = defaultdict(int)
        self.registry = {a.id: {'position': a.position.copy(), 'finish': a.busy_until,
                                'skills': a.skills.copy(), 'speed': a.speed} for a in agents}
        self.unserved = []
        self.registry_used = mechanism in ('nearest', 'cached_central') or (mechanism == 'cnp' and shortlist != 'random')
        ledger.context = self
        for agent in agents:
            ledger.record('registration', agent.id, 'registry', payload={'agent_card': agent_card(agent)})

    def estimate(self, agent_id, task, now, backlog=False):
        entry = self.registry[agent_id]
        cost = dist(entry['position'], task.position) / entry['speed'] + task.workload / max(entry['skills'][task.skill], .05)
        return cost + (max(0, entry['finish'] - now) if backlog else 0)

    def candidates(self, task, now):
        agents = list(self.agents)
        if self.shortlist == 'random':
            return [agents[i] for i in self.rng.permutation(len(agents))]
        if self.shortlist == 'capability':
            agents = [a for a in agents if self.registry[a.id]['skills'][task.skill] >= self.theta]
            return sorted(agents, key=lambda a: (self.estimate(a.id, task, now), a.id))
        return sorted(agents, key=lambda a: (dist(self.registry[a.id]['position'], task.position), a.id))

    def assign(self, agent, task, time):
        # SSI already logs its single award. All other v2 policies dispatch here.
        if self.mechanism != 'ssi':
            self.ledger.record('award', 'manager', agent.id, task.id, getattr(self, 'award_round', 0))
        assignment = execute(agent, task, time)
        self.assignments.append(assignment)
        self.history[agent.id].append(assignment)
        self.awards[agent.id] += 1
        if self.registry_used and self.stale_every is not None and self.awards[agent.id] % self.stale_every == 0:
            self.registry[agent.id].update(position=agent.position.copy(), finish=agent.busy_until)
            self.ledger.record('status_update', agent.id, 'registry', task.id,
                               payload={'agent_id': agent.id, 'queue_end_position': agent.position.tolist(),
                                        'committed_finish': agent.busy_until})


def cnp_allocate(context, task, now):
    candidates = context.candidates(task, now)
    for start in range(0, len(candidates), context.k):
        round_id = start // context.k
        context.rebid_rounds += int(round_id > 0)
        group = candidates[start:start + context.k]
        offers = []
        for agent in group:
            context.ledger.record('cfp', 'manager', agent.id, task.id, round_id)
            if agent.skills[task.skill] < context.theta:
                context.ledger.record('decline', agent.id, 'manager', task.id, round_id,
                                      payload={'task_id': task.id, 'reason': 'ineligible'})
                continue
            if context.nonresponse and context.rng.random() < context.nonresponse:
                continue
            quote = agent.estimate_cost(task, now, bid_noise_std=context.noise, rng=context.rng)
            context.ledger.record('bid', agent.id, 'manager', task.id, round_id,
                                  payload={'task_id': task.id, 'cost': quote})
            offers.append((quote, agent.id))
        if offers:
            context.award_round = round_id
            context.assign(context.agents[min(offers)[1]], task, now)
            return
        context.failed_bid_rounds += 1
    context.unserved.append(task.id)


def allocate(context, mechanism, tasks, now):
    if mechanism in ('ssi', 'cbba'):
        if context.theta:
            raise ValueError('Hard eligibility experiments use CNP; SSI/CBBA support soft proficiency only')
        MECHANISMS[mechanism]().allocate(context, tasks, now)
        return
    for task in tasks:
        if mechanism == 'cnp':
            cnp_allocate(context, task, now)
            continue
        eligible = [a for a in context.agents if a.skills[task.skill] >= context.theta]
        if mechanism == 'greedy':
            for agent in context.agents:
                context.ledger.record('centralized_full_query', 'manager', agent.id, task.id)
                context.ledger.record('centralized_full_response', agent.id, 'manager', task.id)
        if not eligible:
            context.unserved.append(task.id)
            continue
        if mechanism == 'greedy':
            winner = min(eligible, key=lambda a: (bid(a, task, now), a.id))
        elif mechanism == 'cached_central':
            winner = min(eligible, key=lambda a: (context.estimate(a.id, task, now, backlog=True), a.id))
        elif mechanism == 'nearest':
            winner = min(eligible, key=lambda a: (dist(context.registry[a.id]['position'], task.position), a.id))
        elif mechanism == 'round_robin':
            winner = eligible[len(context.assignments) % len(eligible)]
        else:
            raise ValueError(f'Unknown v2 mechanism {mechanism}')
        context.assign(winner, task, now)


def private_jobs(scenario, rho):
    rng = np.random.default_rng(np.random.SeedSequence(scenario.seed).spawn(4)[3])
    c = scenario.config
    tasks = generate_tasks(c.expected_tasks * rho, c.n_skills, c.horizon, c.side, rng, force_nonempty=False)
    for i, task in enumerate(tasks):
        task.task_id = -i - 1
    return [(task, int(rng.integers(c.n_agents))) for task in tasks]


def simulate_v2(scenario, mechanism, *, k=5, batch_window=25, noise=0, nonresponse=0,
                shortlist='capability', stale_every=1, theta=0, private_load=0, context_hook=None):
    from .simulation import SimulationResult
    if not isinstance(mechanism, str):
        mechanism = next(key for key, cls in MECHANISMS.items() if isinstance(mechanism, cls))
    if shortlist not in ('nearest', 'capability', 'random') or theta < 0 or private_load < 0:
        raise ValueError('Invalid v2 information policy')
    if stale_every is not None and (not isinstance(stale_every, int) or stale_every < 1):
        raise ValueError('stale_every must be a positive integer or None (never)')
    tasks = sorted(deepcopy(scenario.tasks), key=lambda t: (t.arrival, t.id))
    private = private_jobs(scenario, private_load)
    ctx = V2Context(deepcopy(scenario.agents), tasks + [t for t, _ in private], scenario.seed,
                    mechanism=mechanism, k=k, noise=noise, nonresponse=nonresponse,
                    shortlist=shortlist, stale_every=stale_every, theta=theta)
    events = []
    if mechanism in ('ssi', 'cbba'):
        for window, batch in groupby(tasks, key=lambda t: floor(t.arrival / batch_window)):
            events.append((min((window+1)*batch_window, scenario.config.horizon), 1, list(batch), None))
    else:
        events.extend((t.arrival, 1, [t], None) for t in tasks)
    events.extend((t.arrival, 0, [t], owner) for t, owner in private)
    for now, kind, batch, owner in sorted(events, key=lambda e: (e[0], e[1])):
        ctx.ledger.time = now
        if kind == 0:
            assignment = execute(ctx.agents[owner], batch[0], now)
            ctx.history[owner].append(assignment)
        elif context_hook:
            context_hook(ctx, batch[0], now)
        else:
            allocate(ctx, mechanism, batch, now)
    metrics = measure(ctx.agents, ctx.assignments, ctx.ledger)
    # Fairness, quality and denominators concern managed work only, including in E6.
    loads = [sum(a.travel + a.execution for a in ctx.assignments if a.agent_id == agent.id) for agent in ctx.agents]
    total = len(tasks)
    metrics.update(fairness=jain_index(loads), tasks=total, served=len(ctx.assignments),
                   unserved=len(ctx.unserved), unserved_rate=len(ctx.unserved)/total if total else 0,
                   private_jobs=len(private), tokens=ctx.ledger.token_cost(),
                   messages_per_task=len(ctx.ledger.messages)/total if total else 0,
                   tokens_per_task=ctx.ledger.token_cost()/total if total else 0,
                   declines_per_task=ctx.ledger.counts_by_type().get('decline', 0)/total if total else 0,
                   reannouncements_per_task=ctx.rebid_rounds/total if total else 0,
                   reannouncements=ctx.rebid_rounds, failed_bid_rounds=ctx.failed_bid_rounds)
    return SimulationResult(ctx.agents, ctx.assignments, ctx.ledger, metrics, ctx.rebid_rounds)
