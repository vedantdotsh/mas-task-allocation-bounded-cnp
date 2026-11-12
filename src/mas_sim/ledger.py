from collections import Counter
from dataclasses import dataclass, field


# Token assumptions, not measured provider usage. Vector payloads are flat-priced.
TOKEN_COSTS = {
    "centralized_full_query": 120, "centralized_full_response": 80,
    "position_query": 32, "position_response": 24,
    "cfp": 32, "bid": 48, "award": 24, "consensus": 60,
}

# Explicit alternative for comparison only; the original proxy remains primary.
PAPER_TOKEN_COSTS = {
    "centralized_full_query": 400, "centralized_full_response": 50,
    "position_query": 400, "position_response": 50,
    "cfp": 150, "bid": 50, "award": 20, "consensus": 60,
}


@dataclass(frozen=True)
class Message:
    type: str
    sender: int | str
    receiver: int | str
    task_ids: tuple[int, ...]
    round_id: int = 0


@dataclass
class MessageLedger:
    messages: list[Message] = field(default_factory=list)

    def record(self, kind: str, sender: int | str, receiver: int | str,
               tasks: int | tuple[int, ...], round_id: int = 0, payload=None):
        if kind not in TOKEN_COSTS:
            raise ValueError(f"Unknown message type: {kind}")
        task_ids = (tasks,) if isinstance(tasks, int) else tuple(tasks)
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("Message must reference distinct tasks")
        self.messages.append(Message(kind, sender, receiver, task_ids, round_id))

    def counts_by_type(self) -> dict[str, int]:
        return dict(Counter(m.type for m in self.messages))

    def counts_by_task(self) -> dict[int, int]:
        """Message incidences; multi-task messages occur in several task counts."""
        return dict(Counter(t for m in self.messages for t in m.task_ids))

    def token_cost(self, costs: dict[str, int] | None = None) -> int:
        prices = TOKEN_COSTS if costs is None else costs
        return sum(prices[m.type] for m in self.messages)
