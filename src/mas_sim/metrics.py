"""Recovered metric definitions and token proxy, plus audit diagnostics."""
import numpy as np

from .ledger import TOKEN_COSTS

CONTEXT_WINDOW_TOKENS = 8192
LATENCY_PER_MESSAGE = 0.02
TOKENS = TOKEN_COSTS


def jains_fairness(loads):
    values = np.asarray(loads, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Loads must be finite and nonnegative")
    denominator = float(np.dot(values, values))
    return float(values.sum() ** 2 / (values.size * denominator)) if denominator else 1.0


jain_index = jains_fairness


def communication_proxy(message_count, token_units, latency_units=None):
    return {"message_count": int(message_count), "estimated_token_cost": int(token_units),
            "estimated_latency": float(message_count * LATENCY_PER_MESSAGE if latency_units is None else latency_units),
            "context_pressure": float(token_units / CONTEXT_WINDOW_TOKENS)}


def collect_metrics(agents, tasks, *, total_messages, token_units, latency_units=None, extra=None):
    makespan = max((t.finish_time or 0.0 for t in tasks), default=0.0)
    wait = [(t.start_time if t.start_time is not None else t.arrival_time) - t.arrival_time for t in tasks]
    loads = [a.total_busy_time for a in agents]
    result = {"makespan": makespan, "avg_wait": float(np.mean(wait)) if wait else 0.0,
              "p95_wait": float(np.percentile(wait, 95)) if wait else 0.0,
              "fairness": jains_fairness(loads), "messages": total_messages, "loads": loads,
              "throughput": len(tasks) / makespan if makespan else 0.0}
    result.update(communication_proxy(total_messages, token_units, latency_units))
    result.update(extra or {})
    return result


def measure(agents, assignments, ledger, latency_units=None):
    makespan = max((a.finish for a in assignments), default=0.0)
    queue_wait = [a.dispatch - a.arrival for a in assignments]
    service_wait = [a.service_start - a.arrival for a in assignments]
    response = [a.finish - a.arrival for a in assignments]
    last_arrival = max((a.arrival for a in assignments), default=0.0)
    result = {
        "tasks": len(assignments), "makespan": makespan,
        "avg_wait": float(np.mean(service_wait)) if service_wait else 0.0,
        "mean_wait": float(np.mean(service_wait)) if service_wait else 0.0,
        "p95_wait": float(np.percentile(service_wait, 95)) if service_wait else 0.0,
        "mean_queue_wait": float(np.mean(queue_wait)) if queue_wait else 0.0,
        "p95_queue_wait": float(np.percentile(queue_wait, 95)) if queue_wait else 0.0,
        "mean_response": float(np.mean(response)) if response else 0.0,
        "fairness": jains_fairness([a.busy_time for a in agents]),
        "messages": len(ledger.messages), "proxy_tokens": ledger.token_cost(),
        "throughput": len(assignments) / makespan if makespan else 0.0,
        "last_arrival": last_arrival, "drain_time": makespan - last_arrival,
    }
    result.update(communication_proxy(len(ledger.messages), ledger.token_cost(), latency_units))
    return result
