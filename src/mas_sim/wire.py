"""Concrete JSON-RPC-style control messages; these are not A2A conformance claims."""
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .ledger import MessageLedger

try:
    import tiktoken
    ENCODING = tiktoken.get_encoding('o200k_base')
    TOKENIZER = 'tiktoken/o200k_base'
except (ImportError, OSError):
    ENCODING = None
    TOKENIZER = 'ceil(UTF-8 bytes / 4) estimate'


@lru_cache(maxsize=16384)
def count_tokens(text):
    return len(ENCODING.encode(text, disallowed_special=())) if ENCODING else math.ceil(len(text.encode('utf-8')) / 4)


def compact(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite wire value')
        return round(value, 6)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(v) for v in value]
    return value


def serialize(value):
    return json.dumps(compact(value), separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def task_data(task):
    return {'task_id': task.id, 'required_skill': task.skill, 'workload': task.workload,
            'location': task.position.tolist(), 'arrival': task.arrival, 'deadline': None}


def agent_card(agent):
    return {'name': f'agent-{agent.id}', 'agent_id': agent.id, 'version': '1.0',
            'url': f'https://agents.example/{agent.id}',
            'capabilities': {'skills': agent.skills.tolist(), 'speed': agent.speed},
            'position': agent.position.tolist(), 'committed_finish': agent.busy_until}


def local_state(context, agent, now):
    return {'agent_id': agent.id, 'skills': agent.skills.tolist(), 'speed': agent.speed,
            'queue_end_position': agent.position.tolist(), 'committed_finish': agent.busy_until,
            'queue': [{**task_data(context.task_by_id[a.task_id]), 'service_start': a.service_start,
                       'finish': a.finish} for a in context.history[agent.id] if a.finish > now]}


@dataclass(frozen=True)
class WireMessage:
    type: str
    sender: int | str
    receiver: int | str
    task_ids: tuple[int, ...]
    round_id: int
    serialized: str
    tokens: int


class WireLedger(MessageLedger):
    def __init__(self):
        super().__init__()
        self.context = None
        self.time = 0.0

    def record(self, kind, sender, receiver, tasks=(), round_id=0, payload=None):
        ids = (tasks,) if isinstance(tasks, int) else tuple(tasks)
        ctx = self.context
        if payload is None:
            if kind in ('cfp', 'award'):
                payload = {'task': task_data(ctx.task_by_id[ids[0]])}
            elif kind == 'centralized_full_query':
                payload = {'agent_id': receiver, 'include': ['skills', 'position', 'speed', 'queue'], 'at': self.time}
            elif kind == 'centralized_full_response':
                payload = local_state(ctx, ctx.agents[sender], self.time)
            else:
                raise ValueError(f'Explicit payload required for {kind}')
        # Request IDs are scoped to sender/receiver transport connections. Routing metadata
        # lives in the ledger, not in an invented billable transport wrapper.
        request_id = f'{ids[0] if ids else "init"}:{round_id}:{sender if isinstance(sender, int) else receiver}'
        methods = {'registration': 'directory/register', 'cfp': 'tasks/propose', 'award': 'tasks/assign',
                   'status_update': 'directory/update', 'centralized_full_query': 'agents/state',
                   'consensus': 'bundles/consensus'}
        if kind in ('bid', 'decline', 'centralized_full_response'):
            envelope = {'jsonrpc': '2.0', 'id': request_id, 'result': payload}
        else:
            envelope = {'jsonrpc': '2.0', 'id': request_id, 'method': methods[kind], 'params': payload}
        encoded = serialize(envelope)
        self.messages.append(WireMessage(kind, sender, receiver, ids, round_id, encoded, count_tokens(encoded)))

    def token_cost(self, costs=None):
        if costs is not None:
            raise ValueError('Flat token constants are legacy-only')
        return sum(m.tokens for m in self.messages)

    def token_summary(self):
        result = defaultdict(lambda: {'messages': 0, 'tokens': 0, 'bytes': 0})
        for m in self.messages:
            row = result[m.type]
            row['messages'] += 1
            row['tokens'] += m.tokens
            row['bytes'] += len(m.serialized.encode('utf-8'))
        return {k: dict(v, mean_tokens=v['tokens']/v['messages']) for k, v in result.items()}
