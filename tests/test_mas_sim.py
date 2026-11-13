import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_sim import (
    clone_agents,
    make_task_stream,
    make_team,
    run_cbba,
    run_contract_net,
    run_exhaustive_optimal,
    run_greedy_centralized,
    run_nearest_agent,
    run_round_robin,
    run_ssi_auction,
    run_state_tracking_manager,
)
from mas_sim.core import clone_tasks
from mas_sim.metrics import jains_fairness
from mas_sim.core import Agent, Task, make_private_jobs

MECHANISMS = [
    run_round_robin,
    run_nearest_agent,
    run_greedy_centralized,
    run_contract_net,
    run_ssi_auction,
    run_cbba,
    run_state_tracking_manager,
]


def make_scenario(seed=0, n_agents=6, tasks=40):
    rng = np.random.default_rng(seed)
    team = make_team(n_agents, 5, 100.0, rng)
    stream = make_task_stream(tasks, 200.0, 5, 100.0, rng)
    return team, stream


@pytest.mark.parametrize("mech", MECHANISMS)
def test_all_tasks_completed(mech):
    team, stream = make_scenario()
    result = mech(clone_agents(team), clone_tasks(stream))
    assert result["n_tasks"] == len(stream)
    assert result["makespan"] > 0
    assert 0 < result["fairness"] <= 1.0 + 1e-9


def test_tasks_never_start_before_arrival():
    team, stream = make_scenario()
    tasks = clone_tasks(stream)
    run_contract_net(clone_agents(team), tasks)
    for t in tasks:
        assert t.start_time >= t.arrival_time - 1e-9


def test_cnp_close_to_oracle_with_fewer_messages():
    gaps, msg_ratios = [], []
    for seed in range(5):
        team, stream = make_scenario(seed=seed, n_agents=12, tasks=150)
        gc = run_greedy_centralized(clone_agents(team), clone_tasks(stream))
        cnp = run_contract_net(clone_agents(team), clone_tasks(stream), k_invite=5)
        gaps.append(cnp["makespan"] / gc["makespan"])
        msg_ratios.append(cnp["messages"] / gc["messages"])
    assert np.mean(gaps) < 1.15
    assert np.mean(msg_ratios) < 0.6


def test_optimal_is_lower_bound():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        team = make_team(4, 3, 100.0, rng)
        stream = make_task_stream(9, 60.0, 3, 100.0, rng)
        if len(stream) > 11:
            stream = stream[:11]
        opt = run_exhaustive_optimal(clone_agents(team), clone_tasks(stream))
        for mech in [run_greedy_centralized, run_contract_net]:
            res = mech(clone_agents(team), clone_tasks(stream))
            assert res["makespan"] >= opt["makespan"] - 1e-6


def test_jains_fairness_bounds():
    assert jains_fairness(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert jains_fairness(np.array([1.0, 0.0, 0.0])) == pytest.approx(1 / 3)


def test_rebid_counts_extra_messages():
    team, stream = make_scenario(seed=1, n_agents=8, tasks=80)
    rng = np.random.default_rng(7)
    base = run_contract_net(
        clone_agents(team), clone_tasks(stream), bid_noise=0.3,
        rng=np.random.default_rng(7),
    )
    rebid = run_contract_net(
        clone_agents(team), clone_tasks(stream), bid_noise=0.3, rebid=True,
        rng=np.random.default_rng(7),
    )
    assert rebid["rebids"] >= 0
    if rebid["rebids"] > 0:
        assert rebid["messages"] > base["messages"]


@pytest.mark.parametrize("mech", [run_ssi_auction, run_cbba])
def test_batch_decision_precedes_every_start(mech):
    team = [Agent(0, np.zeros(2), np.ones(1), 1.0)]
    tasks = [Task(i, arrival, 0, 1.0, np.zeros(2))
             for i, arrival in enumerate([1.0, 24.0, 26.0, 49.0])]
    mech(team, tasks)
    assert all(t.start_time >= (24.0 if t.arrival_time <= 25.0 else 49.0)
               for t in tasks)


def test_cbba_keeps_held_bundle_load_and_order():
    team = [Agent(i, np.zeros(2), np.ones(1), 1.0) for i in range(2)]
    tasks = [Task(i, 1.0, 0, work, np.zeros(2))
             for i, work in enumerate([8.0, 7.0, 6.0, 5.0, 4.0, 3.0])]
    run_cbba(team, tasks)
    # Each successive shortest item uses the queue including retained claims.
    assert [t.assigned_agent for t in tasks] == [1, 0, 1, 0, 1, 0]
    assert [t.finish_time for t in tasks] == [19.0, 16.0, 11.0, 9.0, 5.0, 4.0]


def test_cbba_retained_prefix_prevents_overclaiming():
    team = [Agent(i, np.array(position, dtype=float), np.ones(1), 1.0)
            for i, position in enumerate([(8, 6), (5, 2), (3, 0)])]
    tasks = [Task(i, 1.0, 0, work, np.array(position, dtype=float))
             for i, (work, position) in enumerate([
                 (1, (0, 1)), (8, (6, 9)), (5, (6, 9)),
                 (7, (6, 5)), (6, (9, 2)), (8, (6, 0))])]
    run_cbba(team, tasks)
    # Omitting the retained-prefix load incorrectly gives task 3 to agent 0.
    assert [t.assigned_agent for t in tasks] == [2, 0, 0, 1, 1, 2]
    assert max(t.finish_time for t in tasks) == pytest.approx(22.242640687119284)


@pytest.mark.parametrize("seed", range(5))
def test_manager_equals_greedy_with_fresh_state(seed):
    team, stream = make_scenario(seed=seed)
    managed, polled = clone_tasks(stream), clone_tasks(stream)
    manager = run_state_tracking_manager(clone_agents(team), managed)
    greedy = run_greedy_centralized(clone_agents(team), polled)
    assert [(t.assigned_agent, t.start_time, t.finish_time) for t in managed] == [
        (t.assigned_agent, t.start_time, t.finish_time) for t in polled]
    assert manager["messages"] == len(stream)
    assert manager["makespan"] == greedy["makespan"]


@pytest.mark.parametrize("mech", [run_state_tracking_manager, run_contract_net,
                                  run_greedy_centralized])
def test_private_work_is_shared_and_seeded(mech):
    team, stream = make_scenario()
    jobs = make_private_jobs(6, 200.0, 0.3, np.random.default_rng(42))
    assert jobs == make_private_jobs(6, 200.0, 0.3, np.random.default_rng(42))
    heavier = make_private_jobs(6, 200.0, 0.5, np.random.default_rng(42))
    assert [(a, i) for a, i, d in jobs] == [(a, i) for a, i, d in heavier]
    assert [d / 0.3 for a, i, d in jobs] == pytest.approx([d / 0.5 for a, i, d in heavier])
    first = mech(clone_agents(team), clone_tasks(stream), private_jobs=jobs)
    second = mech(clone_agents(team), clone_tasks(stream), private_jobs=jobs)
    assert first == second


def test_private_load_and_report_accounting():
    team = [Agent(i, np.zeros(2), np.ones(1), 1.0) for i in range(2)]
    stream = [Task(i, float(i + 1), 0, 1.0, np.zeros(2)) for i in range(5)]
    jobs = [(0.5, 0, 100.0)]
    stale, live = clone_tasks(stream), clone_tasks(stream)
    manager = run_state_tracking_manager(clone_agents(team), stale, private_jobs=jobs)
    greedy = run_greedy_centralized(clone_agents(team), live, private_jobs=jobs,
                                  count_awards=True)
    assert stale[0].start_time == 100.5
    assert live[0].start_time == 1.0
    assert manager["messages"] == 5
    assert greedy["messages"] == 25
    refreshed = run_state_tracking_manager(clone_agents(team), clone_tasks(stream),
                                          private_jobs=jobs, refresh_every=2)
    assert refreshed["status_reports"] == 4
    assert refreshed["messages"] == 9


def test_published_no_responder_rule_is_explicit():
    team, stream = make_scenario()
    result = run_contract_net(team, stream, k_invite=5, nonresponse_prob=1.0,
                              rng=np.random.default_rng(7))
    assert result["no_responder_fallbacks"] == len(stream)
    assert result["messages"] == 6 * len(stream)  # 5 failed CFPs + award


def test_cbba_counts_actual_rounds_and_rejects_nonconvergence():
    team = [Agent(i, np.zeros(2), np.ones(1), 1.0) for i in range(2)]
    tasks = [Task(0, 1.0, 0, 1.0, np.zeros(2))]
    result = run_cbba(clone_agents(team), clone_tasks(tasks))
    assert result["messages"] == 4  # one changing sweep + one stable sweep
    with pytest.raises(RuntimeError, match="did not converge"):
        run_cbba(clone_agents(team), clone_tasks(tasks), max_consensus_rounds=1)
