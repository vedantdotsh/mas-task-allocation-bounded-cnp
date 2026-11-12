"""Recovered June model, with concise aliases for the common protocol interface."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    task_id: int
    skill_required: int
    workload: float
    position: np.ndarray
    arrival_time: float
    assigned_to: int | None = None
    start_time: float | None = None
    finish_time: float | None = None
    rebid_count: int = 0

    @property
    def id(self):
        return self.task_id

    @property
    def arrival(self):
        return self.arrival_time

    @property
    def skill(self):
        return self.skill_required


@dataclass
class Agent:
    agent_id: int
    position: np.ndarray
    skills: np.ndarray
    speed: float
    busy_until: float = 0.0
    completed_tasks: list[int] = field(default_factory=list)
    total_travel_time: float = 0.0
    total_execution_time: float = 0.0
    messages_sent: int = 0
    last_known_position: np.ndarray | None = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.skills = np.asarray(self.skills, dtype=float)
        if self.speed <= 0:
            raise ValueError("Speed must be positive")
        if self.last_known_position is None:
            self.last_known_position = self.position.copy()

    @property
    def id(self):
        return self.agent_id

    @property
    def total_busy_time(self):
        return self.total_travel_time + self.total_execution_time

    @property
    def busy_time(self):
        return self.total_busy_time

    def travel_time(self, target, *, from_position=None):
        origin = self.position if from_position is None else from_position
        return float(np.linalg.norm(origin - target) / self.speed)

    def execution_time(self, task):
        return float(task.workload / max(float(self.skills[task.skill_required]), 0.05))

    def estimate_cost(self, task, now, *, bid_noise_std=0.0, rng=None, use_last_known_position=False):
        position = self.last_known_position if use_last_known_position else self.position
        cost = max(0.0, self.busy_until - now) + self.travel_time(task.position, from_position=position) + self.execution_time(task)
        if bid_noise_std > 0.0 and rng is not None:
            cost *= max(0.0, float(rng.normal(1.0, bid_noise_std)))
        return float(cost)
