import json

import numpy as np
import pytest

from mas_sim.scenario import ScenarioConfig, generate_scenario
from mas_sim.simulation import simulate


def test_nested_teams_common_tasks_and_cached_equivalence():
    small = generate_scenario(ScenarioConfig(n_agents=4, expected_tasks=12), 9)
    large = generate_scenario(ScenarioConfig(n_agents=8, expected_tasks=12), 9)
    assert [t.arrival for t in small.tasks] == [t.arrival for t in large.tasks]
    assert all(a.skill == b.skill and a.workload == b.workload and np.array_equal(a.position,b.position)
               for a,b in zip(small.tasks,large.tasks))
    assert all(np.array_equal(a.skills, b.skills) for a, b in zip(small.agents, large.agents))
    assert all(a.speed == b.speed and np.array_equal(a.position,b.position) for a,b in zip(small.agents,large.agents))
    assert simulate(small, 'cached_central').assignments == simulate(small, 'greedy').assignments


@pytest.mark.parametrize('name,per_task', [('cnp', 8), ('greedy', 9), ('nearest', 2),
                                         ('round_robin', 1), ('ssi', 9), ('cached_central', 2)])
def test_v2_message_identity_and_serialized_tokens(name, per_task):
    scenario = generate_scenario(ScenarioConfig(n_agents=4, expected_tasks=10), 2)
    result = simulate(scenario, name, k=3)
    assert result.metrics['messages'] == 4 + per_task * len(scenario.tasks)
    assert result.ledger.counts_by_type()['registration'] == 4
    assert all(json.loads(m.serialized)['jsonrpc'] == '2.0' for m in result.ledger.messages)
    assert result.metrics['tokens'] == sum(m.tokens for m in result.ledger.messages)


def test_no_free_fallback_and_random_no_updates():
    scenario = generate_scenario(ScenarioConfig(n_agents=4, expected_tasks=10), 2)
    failed = simulate(scenario, 'cnp', k=3, nonresponse=1)
    assert failed.metrics['unserved'] == len(scenario.tasks)
    assert failed.ledger.counts_by_type() == {'registration': 4, 'cfp': 4 * len(scenario.tasks)}
    random = simulate(scenario, 'cnp', k=3, shortlist='random')
    assert 'status_update' not in random.ledger.counts_by_type()
    assert random.metrics['messages'] == 4 + 7 * len(scenario.tasks)


def test_eligibility_and_staleness():
    scenario = generate_scenario(ScenarioConfig(n_agents=4, expected_tasks=20), 3)
    for rule in ('nearest', 'random', 'capability'):
        result = simulate(scenario, 'cnp', k=2, shortlist=rule, theta=2)
        assert not result.assignments
        assert result.metrics['unserved_rate'] == 1
        if rule == 'capability':
            assert result.ledger.counts_by_type().get('decline', 0) == 0
    result = simulate(scenario, 'cnp', stale_every=5)
    expected = sum(len(a.completed_tasks) // 5 for a in result.agents)
    assert result.ledger.counts_by_type().get('status_update', 0) == expected
    assert 'status_update' not in simulate(scenario, 'cnp', stale_every=None).ledger.counts_by_type()


def test_running_example_real_code():
    from mas_sim.example import running_scenario
    from mas_sim.domain import bid
    scenario = running_scenario()
    task = scenario.tasks[0]
    assert [bid(a, task, 4) for a in scenario.agents] == [23, 10, 24]
    expected = {'round_robin': (0, 27, 4), 'nearest': (2, 28, 5), 'greedy': (1, 14, 10),
                'cnp': (1, 14, 9), 'cached_central': (1, 14, 5), 'ssi': (1, 35, 10), 'cbba': (1, 35, 22)}
    for name, (owner, finish, messages) in expected.items():
        result = simulate(scenario, name, k=2)
        assert (result.assignments[0].agent_id, result.assignments[0].finish, result.metrics['messages']) == (owner, finish, messages)


def test_private_jobs_shared_and_invisible_until_update():
    from mas_sim.v2 import private_jobs, V2Context
    from mas_sim.domain import execute
    from copy import deepcopy
    scenario = generate_scenario(ScenarioConfig(expected_tasks=20), 4)
    private = private_jobs(scenario, .5)
    assert [(t.arrival,a) for t,a in private] == [(t.arrival,a) for t,a in private_jobs(scenario,.5)]
    ctx = V2Context(deepcopy(scenario.agents), scenario.tasks, 4, mechanism='cached_central', k=5,
                    noise=0, nonresponse=0, shortlist='capability', stale_every=1, theta=0)
    task, owner = private[0]
    before = deepcopy(ctx.registry[owner])
    execute(ctx.agents[owner], task, task.arrival)
    assert ctx.registry[owner]['finish'] == before['finish']
    assert np.array_equal(ctx.registry[owner]['position'], before['position'])
    assert ctx.ledger.counts_by_type() == {'registration': 12}
    for name in ('cnp','greedy','cached_central'):
        result = simulate(scenario,name,private_load=.5)
        assert result.metrics['private_jobs'] == len(private)
        assert result.metrics['tasks'] == len(scenario.tasks)
        assert result.metrics['served'] == len(scenario.tasks)


def test_reannouncement_finds_eligible_and_charges_every_exchange():
    from mas_sim.models import Agent, Task
    from mas_sim.scenario import Scenario
    agents = [Agent(i, np.array([i, 0.]), np.array([.2 if i < 2 else 1.]), 1) for i in range(3)]
    scenario = Scenario(ScenarioConfig(n_agents=3, expected_tasks=1), agents,
                        [Task(0, 0, 1., np.array([0.,0.]), 0)], 0)
    result = simulate(scenario, 'cnp', k=2, shortlist='nearest', theta=.6)
    assert result.assignments[0].agent_id == 2
    assert result.ledger.counts_by_type() == {'registration':3, 'cfp':3, 'decline':2, 'bid':1, 'award':1, 'status_update':1}
    assert result.metrics['reannouncements'] == 1
    assert next(m for m in result.ledger.messages if m.type == 'award').round_id == 1


def test_shortlist_never_reads_backlog_and_cached_does():
    from mas_sim.example import running_scenario
    from mas_sim.v2 import V2Context
    scenario = running_scenario()
    ctx = V2Context(scenario.agents, scenario.tasks, 0, mechanism='cnp', k=2, noise=0,
                    nonresponse=0, shortlist='capability', stale_every=1, theta=0)
    task = scenario.tasks[0]
    ctx.registry[1]['finish'] = 10000
    ctx.agents[1].busy_until = 10000
    assert [a.id for a in ctx.candidates(task,4)] == [1,0,2]
    assert ctx.estimate(1,task,4) == 10
    assert ctx.estimate(1,task,4,backlog=True) == 10006


def test_queue_payload_is_active_and_vector_sizes_grow():
    from mas_sim.wire import count_tokens, serialize
    scenario = generate_scenario(ScenarioConfig(n_agents=4, expected_tasks=20, horizon=20), 5)
    result = simulate(scenario,'greedy')
    responses = [json.loads(m.serialized)['result'] for m in result.ledger.messages if m.type == 'centralized_full_response']
    assert all(not r['queue'] for r in responses[:4])
    assert any(r['queue'] for r in responses[4:])
    for message in result.ledger.messages:
        assert message.tokens == count_tokens(message.serialized)
    ssi = simulate(scenario,'ssi')
    bids = [m for m in ssi.ledger.messages if m.type == 'bid']
    assert max(m.tokens for m in bids if len(m.task_ids)>1) > max(m.tokens for m in bids if len(m.task_ids)==1)


def test_sample_sd_ci_and_prespecified_equivalence():
    from mas_sim.experiments_v2 import summarize, paired
    summary = summarize([1,2,3])
    assert summary['mean'] == 2
    assert summary['std'] == 1
    assert summary['ci95'][1] > 2
    rows = [dict(seed=s, mechanism=n, makespan=v, mean_response=v) for s in range(5)
            for n,v in [('cnp',104),('greedy',100)]]
    result = paired(rows,'greedy')
    assert result['makespan']['ci95'] == [4,4]
    assert result['mean_response']['within_margin'] is True
    for row in rows:
        if row['mechanism']=='cnp':
            row['makespan']=106
    assert paired(rows,'greedy')['makespan']['within_margin'] is False
