"""Shared scheduling operations; model classes come from the recovered source."""
from dataclasses import dataclass

from .models import Agent, Task


@dataclass(frozen=True)
class Assignment:
    task_id: int
    agent_id: int
    arrival: float
    decision: float
    dispatch: float
    service_start: float
    finish: float
    travel: float
    execution: float


def travel_time(agent: Agent, task: Task) -> float:
    return agent.travel_time(task.position)


def execution_time(agent: Agent, task: Task) -> float:
    return agent.execution_time(task)


def bid(agent: Agent, task: Task, time: float) -> float:
    return agent.estimate_cost(task, time)


def execute(agent: Agent, task: Task, decision: float) -> Assignment:
    """Recovered _assign semantics: append FIFO, expose committed queue endpoint."""
    travel = travel_time(agent, task)
    execution = execution_time(agent, task)
    dispatch = max(decision, task.arrival, agent.busy_until)
    service = dispatch + travel
    finish = service + execution
    agent.position = task.position.copy()
    agent.last_known_position = agent.position.copy()
    agent.busy_until = finish
    agent.total_travel_time += travel
    agent.total_execution_time += execution
    agent.completed_tasks.append(task.id)
    task.assigned_to, task.start_time, task.finish_time = agent.id, service, finish
    return Assignment(task.id, agent.id, task.arrival, decision, dispatch, service, finish, travel, execution)
