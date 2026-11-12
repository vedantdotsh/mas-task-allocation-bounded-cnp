from dataclasses import dataclass

import numpy as np

from .generators import generate_agents, generate_tasks
from .models import Agent, Task


@dataclass(frozen=True)
class ScenarioConfig:
    n_agents: int = 12
    expected_tasks: int = 200
    horizon: float = 500
    side: float = 100
    n_skills: int = 5
    arrival_mode: str = "poisson"
    version: str = "v2"

    def __post_init__(self):
        if self.version not in ('v2', 'legacy'):
            raise ValueError('version must be v2 or legacy')
        if self.n_agents < 1 or self.n_skills < 1:
            raise ValueError("Need at least one agent and skill")
        if self.expected_tasks < 0 or self.horizon <= 0 or self.side <= 0:
            raise ValueError("Invalid arrival process or workspace")


@dataclass
class Scenario:
    config: ScenarioConfig
    agents: list[Agent]
    tasks: list[Task]
    seed: int


def generate_scenario(config: ScenarioConfig, seed: int) -> Scenario:
    """Independent nested v2 streams; preserve the June shared-RNG order in legacy."""
    streams = np.random.SeedSequence(seed).spawn(4)
    rng = np.random.default_rng(seed if config.version == 'legacy' else streams[0])
    agents = generate_agents(config.n_agents, config.n_skills, config.side, rng)
    if config.version == 'v2':
        rng = np.random.default_rng(streams[1])
    tasks = generate_tasks(config.expected_tasks, config.n_skills, config.horizon,
                           config.side, rng, arrival_mode=config.arrival_mode,
                           force_nonempty=config.version == 'legacy')
    return Scenario(config, agents, tasks, seed)
