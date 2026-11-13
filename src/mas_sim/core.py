"""Domain layer: agents, tasks, and scenario generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Proficiency floor so unskilled agents never divide by zero (Eq. 2).
SKILL_FLOOR = 0.05


@dataclass
class Task:
    task_id: int
    arrival_time: float
    skill_type: int
    workload: float
    position: np.ndarray
    # Filled in by the coordination mechanism during a run.
    assigned_agent: int = -1
    start_time: float = -1.0
    finish_time: float = -1.0

    @property
    def wait_time(self) -> float:
        return self.start_time - self.arrival_time


@dataclass
class Agent:
    agent_id: int
    position: np.ndarray
    skills: np.ndarray
    speed: float
    busy_until: float = 0.0
    total_travel_time: float = 0.0
    total_execution_time: float = 0.0
    tasks_done: int = 0
    # Position the agent will occupy once its current queue drains.
    committed_position: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.committed_position is None:
            self.committed_position = self.position.copy()

    @property
    def total_busy_time(self) -> float:
        return self.total_travel_time + self.total_execution_time

    def travel_time(self, target: np.ndarray) -> float:
        """Eq. 1: Euclidean distance from queue-end position / speed."""
        return float(np.linalg.norm(self.committed_position - target) / self.speed)

    def execution_time(self, task: Task) -> float:
        """Eq. 2: workload / proficiency (floored)."""
        return task.workload / max(float(self.skills[task.skill_type]), SKILL_FLOOR)

    def estimate_cost(self, task: Task, now: float) -> float:
        """Eq. 3: queue drain + travel + execution. Lower is better."""
        wait = max(0.0, self.busy_until - now)
        return wait + self.travel_time(task.position) + self.execution_time(task)


def schedule_task(agent: Agent, task: Task, decision_time: float | None = None) -> None:
    """Commit *task* to *agent*'s queue and update bookkeeping."""
    start = max(task.arrival_time, agent.busy_until,
                task.arrival_time if decision_time is None else decision_time)
    travel = agent.travel_time(task.position)
    execution = agent.execution_time(task)
    task.assigned_agent = agent.agent_id
    task.start_time = start
    task.finish_time = start + travel + execution
    agent.busy_until = task.finish_time
    agent.total_travel_time += travel
    agent.total_execution_time += execution
    agent.tasks_done += 1
    agent.committed_position = task.position.copy()


def make_private_jobs(n_agents: int, horizon: float, load: float,
                      rng: np.random.Generator) -> list[tuple[float, int, float]]:
    """Stationary private jobs: Poisson rate 1/25, mean service 25*load."""
    if not 0 <= load < 1:
        raise ValueError("Private offered load must be in [0, 1).")
    if load == 0:
        return []
    jobs = []
    for agent_id in range(n_agents):
        arrival = 0.0
        while True:
            arrival += rng.exponential(25.0)
            if arrival > horizon:
                break
            jobs.append((float(arrival), agent_id, float(rng.exponential(25.0) * load)))
    return sorted(jobs)


def tasks_with_private_work(agents: list[Agent], tasks: list[Task], private_jobs=()):
    """Append private service to its local FIFO queue before each public arrival.

    Private work does not move an agent's queue-end position. Jobs and public
    tasks are immutable inputs except for the ordinary task scheduling fields.
    """
    pending = iter(private_jobs)
    job = next(pending, None)
    by_id = {a.agent_id: a for a in agents}
    for task in tasks:
        while job is not None and job[0] <= task.arrival_time:
            arrival, agent_id, duration = job
            agent = by_id[agent_id]
            agent.busy_until = max(arrival, agent.busy_until) + duration
            agent.total_execution_time += duration
            job = next(pending, None)
        yield task


def make_team(n_agents: int, n_skills: int, side: float, rng: np.random.Generator) -> list[Agent]:
    """Heterogeneous team: 1-2 primary skills U(0.8,1.2), rest U(0.2,0.5)."""
    agents = []
    for i in range(n_agents):
        skills = rng.uniform(0.2, 0.5, size=n_skills)
        primary = rng.choice(n_skills, size=rng.integers(1, 3), replace=False)
        skills[primary] = rng.uniform(0.8, 1.2, size=primary.size)
        agents.append(
            Agent(
                agent_id=i,
                position=rng.uniform(0, side, size=2),
                skills=skills,
                speed=float(rng.uniform(1.0, 3.0)),
            )
        )
    return agents


def make_task_stream(
    expected_tasks: int,
    horizon: float,
    n_skills: int,
    side: float,
    rng: np.random.Generator,
) -> list[Task]:
    """True Poisson arrival process over [0, horizon]."""
    rate = expected_tasks / horizon
    tasks: list[Task] = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t > horizon:
            break
        tasks.append(
            Task(
                task_id=len(tasks),
                arrival_time=t,
                skill_type=int(rng.integers(0, n_skills)),
                workload=float(rng.uniform(2.0, 10.0)),
                position=rng.uniform(0, side, size=2),
            )
        )
    return tasks


def clone_agents(agents: list[Agent]) -> list[Agent]:
    return [
        Agent(
            agent_id=a.agent_id,
            position=a.position.copy(),
            skills=a.skills.copy(),
            speed=a.speed,
        )
        for a in agents
    ]


def clone_tasks(tasks: list[Task]) -> list[Task]:
    return [
        Task(
            task_id=t.task_id,
            arrival_time=t.arrival_time,
            skill_type=t.skill_type,
            workload=t.workload,
            position=t.position.copy(),
        )
        for t in tasks
    ]
