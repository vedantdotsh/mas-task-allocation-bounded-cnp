from itertools import permutations, product

import numpy as np
import pytest

from mas_sim.domain import Agent, Task, bid, execute, execution_time, travel_time
from mas_sim.ledger import MessageLedger
from mas_sim.mechanisms import MECHANISMS
from mas_sim.metrics import jain_index
from mas_sim.optimal import exact_optimum
from mas_sim.scenario import ScenarioConfig, generate_scenario
from mas_sim.simulation import simulate


def test_equations_and_queue_end_position():
    agent = Agent(0, np.array([0, 0]), np.array([0, 0.5]), 2, busy_until=10)
    task = Task(0, 1, 3, np.array([6, 8]), 4)
    assert travel_time(agent, task) == 5
    assert execution_time(agent, task) == 6
    assert bid(agent, task, 4) == 17
    assert bid(agent, task, 12) == 11
    assert execution_time(agent, Task(1, 0, 2, np.array([0, 0]), 0)) == 40
    assigned = execute(agent, task, 4)
    assert (assigned.dispatch, assigned.service_start, assigned.finish) == (10, 15, 21)
    assert np.array_equal(agent.position, task.position)
    assert agent.busy_time == 11


@pytest.mark.parametrize("name,per_task", [("greedy", 8), ("nearest", 8), ("cnp", 7), ("round_robin", 0)])
def test_message_identities(name, per_task):
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=15, horizon=40), 1)
    result = simulate(scenario, name, k=3)
    assert len(result.ledger.messages) == len(scenario.tasks) * per_task
    if per_task:
        assert set(result.ledger.counts_by_task().values()) == {per_task}
    if name == "cnp":
        assert result.ledger.counts_by_type() == {"cfp": 3 * len(scenario.tasks), "bid": 3 * len(scenario.tasks), "award": len(scenario.tasks)}


@pytest.mark.parametrize("name", list(MECHANISMS))
def test_determinism_and_complete_assignment(name):
    config = ScenarioConfig(version='legacy', n_agents=4, expected_tasks=12, horizon=35)
    scenario = generate_scenario(config, 2)
    again = generate_scenario(config, 2)
    assert [t.arrival for t in scenario.tasks] == [t.arrival for t in again.tasks]
    assert all(np.array_equal(a.position, b.position) for a, b in zip(scenario.agents, again.agents))
    first = simulate(scenario, name, k=3, noise=0.2, nonresponse=0.3, rebid=True)
    second = simulate(scenario, name, k=3, noise=0.2, nonresponse=0.3, rebid=True)
    assert first.metrics == second.metrics
    assert first.assignments == second.assignments
    assert first.ledger == second.ledger
    assert sorted(a.task_id for a in first.assignments) == [t.id for t in scenario.tasks]
    for agent in first.agents:
        queue = [a for a in first.assignments if a.agent_id == agent.id]
        assert all(a.finish <= b.dispatch for a, b in zip(queue, queue[1:]))


def test_full_invitation_is_greedy_and_single_invitation_is_nearest():
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=30, horizon=80), 3)
    assert simulate(scenario, "cnp", k=5).assignments == simulate(scenario, "greedy").assignments
    assert simulate(scenario, "cnp", k=1).assignments == simulate(scenario, "nearest").assignments


def test_nonresponse_fallback_and_rebid_cap():
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=25, horizon=40), 4)
    result = simulate(scenario, "cnp", k=3, nonresponse=1, noise=0.5, rebid=True)
    assert len(result.assignments) == len(scenario.tasks)
    assert result.metrics["failed_bid_rounds"] == 2 * len(scenario.tasks)
    assert result.ledger.counts_by_type() == {"cfp": 6 * len(scenario.tasks)}
    noisy = simulate(scenario, "cnp", k=3, noise=0.5, rebid=True)
    assert noisy.rebid_rounds <= len(scenario.tasks)
    assert len(noisy.ledger.messages) == 7 * len(scenario.tasks) + 7 * noisy.rebid_rounds


def test_message_payload_accounting():
    ledger = MessageLedger()
    ledger.record("consensus", 0, 1, (2, 3), round_id=1)
    assert len(ledger.messages) == 1
    assert ledger.counts_by_task() == {2: 1, 3: 1}
    assert ledger.token_cost() == 60
    ledger.record("centralized_full_query", "manager", 0, 2)
    ledger.record("centralized_full_response", 0, "manager", 2)
    assert ledger.token_cost() == 260


@pytest.mark.parametrize("loads", [[0, 0, 0], [1, 1, 1], [1, 0, 0], [2, 5, 8, 10]])
def test_jain_bounds(loads):
    assert 1 / len(loads) <= jain_index(loads) <= 1
    assert jain_index([1, 1, 1]) == 1
    assert jain_index([1, 0, 0]) == pytest.approx(1 / 3)


def test_exact_solver_matches_independent_enumeration_and_bounds_mechanisms():
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=2, expected_tasks=8, horizon=20), 5)
    scenario.tasks = scenario.tasks[:4]
    optimum = exact_optimum(scenario).makespan
    brute = float("inf")
    import copy
    for order in permutations(scenario.tasks):
        for owners in product(range(2), repeat=len(order)):
            agents = copy.deepcopy(scenario.agents)
            for task, owner in zip(order, owners):
                execute(agents[owner], task, task.arrival)
            brute = min(brute, max(a.busy_until for a in agents))
    assert optimum == pytest.approx(brute)
    for name in MECHANISMS:
        assert optimum <= simulate(scenario, name).metrics["makespan"] + 1e-10


def test_empty_stream():
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=0), 1)
    for name in MECHANISMS:
        result = simulate(scenario, name)
        assert result.metrics["makespan"] == 0
        assert result.metrics["fairness"] == 1
    assert exact_optimum(scenario).makespan == 0


def test_optimal_routes_are_feasible_and_cover_every_task():
    import copy
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=20, horizon=40), 8)
    scenario.tasks = scenario.tasks[:11]
    optimum = exact_optimum(scenario)
    assert sorted(t for route in optimum.routes.values() for t in route) == [t.id for t in scenario.tasks]
    agents = copy.deepcopy(scenario.agents)
    for agent_id, route in optimum.routes.items():
        for task_id in route:
            task = scenario.tasks[task_id]
            execute(agents[agent_id], task, task.arrival)
    assert max(a.busy_until for a in agents) == pytest.approx(optimum.makespan)


def test_policy_object_and_registered_name_have_identical_metrics():
    scenario = generate_scenario(ScenarioConfig(version='legacy', n_agents=4, expected_tasks=10, horizon=40), 6)
    assert simulate(scenario, "cnp").metrics == simulate(scenario, MECHANISMS["cnp"]()).metrics
