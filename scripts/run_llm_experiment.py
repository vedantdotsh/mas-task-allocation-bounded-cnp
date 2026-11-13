"""Live LLM-agent replication of the coordination experiment via OpenRouter.

Compares two coordination conditions with REAL LLM calls, measuring actual
prompt/completion tokens, wall-clock latency, and allocation quality:

1. CNP (decentralized, bounded): for each task, the k nearest candidate
   agents each receive a compact CFP describing the task and ONLY their own
   local state, and return a numeric bid. The manager awards the task to the
   lowest bidder.
2. Centralized orchestrator: for each task, a single orchestrator receives
   the FULL state of every agent in the team and picks the agent directly.

Chosen assignments are replayed in the simulator so both conditions are
scored with the same objective metrics (makespan, wait, fairness) used in
the paper. Token usage comes from the API's own usage accounting, not an
estimate.

Usage:
    Set OPENROUTER_API_KEY in your environment before running.
    python3 scripts/run_llm_experiment.py [--model MODEL] [--trials N]
                                          [--agents N] [--tasks N] [--k K]

Cost control: defaults (3 trials, 8 agents, ~30 tasks, k=3) make roughly
3 * (30*3 + 30) = ~360 short calls per run. With a cheap model this is a few
tens of cents. The script prints running token totals and stops if
--max-calls is exceeded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mas_sim import clone_agents, make_task_stream, make_team
from mas_sim.core import clone_tasks, schedule_task
from mas_sim.mechanisms import k_nearest_agents
from mas_sim.metrics import collect_metrics

API_URL = "https://openrouter.ai/api/v1/chat/completions"

RESULTS_PATH = ROOT / "outputs" / "results" / "llm_results.json"

SIDE = 100.0
N_SKILLS = 5
HORIZON = 150.0

SKILL_NAMES = ["coding", "research", "planning", "retrieval", "evaluation"]


class Budget:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0.0
        self.failures = 0

    def spent(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_latency_ms": self.latency_ms,
            "failures": self.failures,
        }


def call_llm(api_key: str, model: str, system: str, user: str, budget: Budget,
             retries: int = 3) -> tuple[str, dict]:
    if budget.calls >= budget.max_calls:
        raise RuntimeError(f"Call budget exhausted ({budget.max_calls}); aborting for cost safety")
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 60,
    }).encode()
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            latency = (time.time() - t0) * 1000
            usage = data.get("usage", {})
            budget.calls += 1
            budget.prompt_tokens += usage.get("prompt_tokens", 0)
            budget.completion_tokens += usage.get("completion_tokens", 0)
            budget.latency_ms += latency
            content = data["choices"][0]["message"]["content"]
            return content, {"latency_ms": latency, **usage}
        except Exception as e:  # noqa: BLE001 - retry any transport/API error
            last_err = e
            time.sleep(2**attempt)
    budget.failures += 1
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def parse_number(text: str, key: str) -> float | None:
    """Extract a numeric field from a (possibly messy) JSON-ish response."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return float(obj[key])
    except Exception:  # noqa: BLE001
        import re

        match = re.search(rf'"{key}"\s*:\s*([-\d.]+)', text)
        if match:
            return float(match.group(1))
        match = re.search(r"[-\d]+\.?\d*", text)
        return float(match.group(0)) if match else None


def agent_state_line(agent, now: float) -> str:
    skills = ", ".join(
        f"{name}={agent.skills[i]:.2f}" for i, name in enumerate(SKILL_NAMES)
    )
    free_in = max(0.0, agent.busy_until - now)
    return (
        f"agent {agent.agent_id}: position=({agent.committed_position[0]:.0f},"
        f"{agent.committed_position[1]:.0f}), speed={agent.speed:.1f}, "
        f"free_in={free_in:.1f}, skills[{skills}]"
    )


def task_line(task) -> str:
    return (
        f"task: requires skill '{SKILL_NAMES[task.skill_type]}', "
        f"workload={task.workload:.1f}, location=({task.position[0]:.0f},"
        f"{task.position[1]:.0f})"
    )


BID_SYSTEM = (
    "You are a worker agent in a multi-agent team. Estimate your total cost "
    "to complete a task as: time_until_free + distance_to_task/your_speed + "
    "workload/your_proficiency_at_required_skill. Respond ONLY with JSON: "
    '{"bid": <number>}.'
)

ORCH_SYSTEM = (
    "You are a centralized orchestrator assigning a task to one agent from a "
    "team. Each agent's cost is: time_until_free + distance_to_task/speed + "
    "workload/proficiency_at_required_skill. Pick the agent with the lowest "
    'cost. Respond ONLY with JSON: {"agent_id": <integer>}.'
)


def run_llm_cnp(agents, tasks, api_key, model, budget, k):
    b0 = dict(budget.spent())
    for task in tasks:
        candidates = k_nearest_agents(agents, task, k)
        prompts = [
            (
                BID_SYSTEM,
                f"Your state: {agent_state_line(a, task.arrival_time)}\n"
                f"CFP {task_line(task)}\nReturn your bid.",
                a,
            )
            for a in candidates
        ]
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [
                (a, pool.submit(call_llm, api_key, model, sysm, usr, budget))
                for sysm, usr, a in prompts
            ]
            bids = []
            for a, fut in futures:
                try:
                    text, _ = fut.result()
                    bid = parse_number(text, "bid")
                except RuntimeError:
                    bid = None
                if bid is not None:
                    bids.append((bid, a))
        if not bids:  # every candidate failed: fall back to formula
            bids = [(a.estimate_cost(task, task.arrival_time), a) for a in candidates]
        winner = min(bids, key=lambda x: x[0])[1]
        schedule_task(winner, task)
    spent = {key: budget.spent()[key] - b0[key] for key in b0}
    return collect_metrics(agents, tasks, spent["calls"], spent["total_tokens"],
                           extra={"llm": spent})


def run_llm_centralized(agents, tasks, api_key, model, budget):
    b0 = dict(budget.spent())
    for task in tasks:
        team_state = "\n".join(agent_state_line(a, task.arrival_time) for a in agents)
        user = f"Team state:\n{team_state}\n\n{task_line(task)}\nPick one agent."
        try:
            text, _ = call_llm(api_key, model, ORCH_SYSTEM, user, budget)
            choice = parse_number(text, "agent_id")
        except RuntimeError:
            choice = None
        valid = choice is not None and 0 <= int(choice) < len(agents)
        winner = (
            agents[int(choice)]
            if valid
            else min(agents, key=lambda a: a.estimate_cost(task, task.arrival_time))
        )
        schedule_task(winner, task)
    spent = {key: budget.spent()[key] - b0[key] for key in b0}
    return collect_metrics(agents, tasks, spent["calls"], spent["total_tokens"],
                           extra={"llm": spent})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=1500)
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in the environment first.")

    budget = Budget(args.max_calls)
    from mas_sim.mechanisms import run_contract_net, run_greedy_centralized

    trials = []
    for seed in range(args.trials):
        rng = np.random.default_rng(7000 + seed)
        team = make_team(args.agents, N_SKILLS, SIDE, rng)
        stream = make_task_stream(args.tasks, HORIZON, N_SKILLS, SIDE, rng)
        print(f"Trial {seed}: {len(stream)} tasks")

        res = {
            "llm_cnp": run_llm_cnp(clone_agents(team), clone_tasks(stream),
                                   api_key, args.model, budget, args.k),
            "llm_centralized": run_llm_centralized(clone_agents(team), clone_tasks(stream),
                                                   api_key, args.model, budget),
            "formula_cnp": run_contract_net(clone_agents(team), clone_tasks(stream),
                                            k_invite=args.k),
            "formula_centralized": run_greedy_centralized(clone_agents(team),
                                                          clone_tasks(stream)),
        }
        for name, r in res.items():
            llm = r.get("llm", {})
            print(f"  {name:20s} makespan={r['makespan']:7.1f} "
                  f"tokens={llm.get('total_tokens', 0):6d} "
                  f"latency={llm.get('total_latency_ms', 0)/1000:6.1f}s "
                  f"failures={llm.get('failures', 0)}")
        trials.append(res)

    out = {
        "params": vars(args) | {"model": args.model},
        "budget": budget.spent(),
        "trials": trials,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nTotal: {budget.calls} calls, "
          f"{budget.prompt_tokens + budget.completion_tokens} tokens, "
          f"{budget.failures} failures")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
