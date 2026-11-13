"""Multi-agent task-allocation testbed for the MEDES 2026 paper.

Implements and compares coordination mechanisms for decentralized task
allocation in heterogeneous multi-agent teams:

- Round-Robin (RR): no-coordination baseline
- Nearest-Agent (NA): spatial-only baseline
- Greedy-Centralized (GC): omniscient myopic oracle
- Contract Net Protocol (CNP): proposed bounded decentralized mechanism
- Sequential Single-Item auction (SSI): stronger market-based baseline
- Consensus-Based Bundle Algorithm (CBBA): standard decentralized baseline
- Exhaustive-search optimum: ground-truth reference on small instances
- CNP with multi-round re-bidding: robustness extension
"""

from .core import Agent, Task, make_team, make_task_stream, clone_agents
from .mechanisms import (
    run_round_robin,
    run_nearest_agent,
    run_greedy_centralized,
    run_state_tracking_manager,
    run_contract_net,
    run_ssi_auction,
    run_cbba,
)
from .optimal import run_exhaustive_optimal
from .metrics import collect_metrics, jains_fairness

__all__ = [
    "Agent",
    "Task",
    "make_team",
    "make_task_stream",
    "clone_agents",
    "run_round_robin",
    "run_nearest_agent",
    "run_greedy_centralized",
    "run_state_tracking_manager",
    "run_contract_net",
    "run_ssi_auction",
    "run_cbba",
    "run_exhaustive_optimal",
    "collect_metrics",
    "jains_fairness",
]
