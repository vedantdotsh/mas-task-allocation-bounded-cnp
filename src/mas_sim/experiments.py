"""Reproducible experiments and the recovered June public entry points."""
import csv
import gzip
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .ledger import PAPER_TOKEN_COSTS, TOKEN_COSTS
from .mechanisms import MECHANISMS
from .optimal import exact_optimum
from .scenario import ScenarioConfig, generate_scenario
from .simulation import simulate
from .stats import _bootstrap_ci
from .strategies import RobustnessConfig

PAPER_NAMES = {"round_robin": "Round-Robin", "nearest": "Nearest-Agent", "greedy": "Greedy-Centralized",
               "cnp": "Contract-Net", "ssi": "SSI-Auction", "cbba": "CBBA"}
JUNE_NAMES = {"round_robin": "round_robin", "nearest": "nearest_agent", "greedy": "greedy_centralized", "cnp": "contract_net"}
SUMMARY_METRICS = ("makespan", "mean_wait", "avg_wait", "p95_wait", "fairness", "messages",
                   "throughput", "proxy_tokens", "paper_tokens", "estimated_token_cost", "estimated_latency",
                   "context_pressure", "tasks", "mean_queue_wait", "p95_queue_wait", "mean_response",
                   "last_arrival", "drain_time", "failed_bid_rounds", "rebid_rounds", "optimal_ratio")


@dataclass(frozen=True)
class ExperimentConfig:
    n_trials: int = 30
    n_agents: int = 12
    expected_tasks: int = 200
    n_skills: int = 5
    horizon: float = 500.0
    workspace: float = 100.0
    k_invite: int = 5
    arrival_mode: str = "poisson"
    include_rebid: bool = False
    robustness: RobustnessConfig = RobustnessConfig()


def run_single_trial(config: ExperimentConfig, seed: int):
    """June API retained; original four mechanisms and optional 1.35-threshold re-bid."""
    scenario = generate_scenario(ScenarioConfig(config.n_agents, config.expected_tasks, config.horizon,
                                               config.workspace, config.n_skills, config.arrival_mode, 'legacy'), seed)
    results = {}
    mechanisms = list(JUNE_NAMES.items())
    if config.include_rebid:
        mechanisms.append(("cnp", "contract_net_rebid"))
    for mechanism, name in mechanisms:
        robust = config.robustness if mechanism == "cnp" else RobustnessConfig()
        result = simulate(scenario, mechanism, k=config.k_invite, noise=robust.bid_noise_std,
                          nonresponse=robust.agent_failure_rate, rebid=name.endswith("rebid"),
                          message_delay_mean=robust.message_delay_mean, stale_position=robust.stale_position,
                          rebid_threshold_factor=1.35)
        results[name] = dict(result.metrics, loads=[a.busy_time for a in result.agents], rebids=result.rebid_rounds)
    return results


def aggregate(raw_results, metrics=SUMMARY_METRICS):
    result = {}
    for name, rows in raw_results.items():
        result[name] = {}
        for metric in metrics:
            values = [row[metric] for row in rows if metric in row]
            if values:
                # Preserve the original's population standard deviation (ddof=0).
                result[name][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return result


def run_experiment(config: ExperimentConfig):
    raw = {}
    for seed in range(config.n_trials):
        for name, row in run_single_trial(config, seed).items():
            raw.setdefault(name, []).append(row)
    return aggregate(raw), raw


def _points(number):
    if number == 1:
        return [("base", {}, list(MECHANISMS))]
    if number == 2:
        return [(str(n), {"n_agents": n}, list(MECHANISMS)) for n in (4, 8, 12, 16, 24, 32)]
    if number == 3:
        return [(str(k), {"k": k}, ["cnp"]) for k in (1, 2, 3, 5, 8, 12)]
    if number == 4:
        return [("small", {"n_agents": 4}, list(MECHANISMS))]
    if number == 5:
        noise = [(f"{mode}/{sigma}", {"noise": sigma, "rebid": mode == "rebid"}, ["cnp"])
                 for mode in ("single_round", "rebid") for sigma in (0.0, 0.1, 0.2, 0.3, 0.5)]
        return noise + [(f"nonresponse/{p}", {"nonresponse": p}, ["cnp"]) for p in (0.0, 0.1, 0.3)]
    raise ValueError("Experiment number must be 1 through 5")


def _save_audit(output_dir, name, result):
    audit = output_dir / "audit"
    audit.mkdir(exist_ok=True)
    with gzip.open(audit / f"{name}_messages.jsonl.gz", "wt", encoding="utf-8") as handle:
        for message in result.ledger.messages:
            handle.write(json.dumps(asdict(message)) + "\n")
    (audit / f"{name}_schedule.json").write_text(json.dumps([asdict(a) for a in result.assignments], indent=2), encoding="utf-8")


def _bootstrap(rows):
    left = {r["seed"]: r for r in rows if r["mechanism"] == "cnp"}
    right = {r["seed"]: r for r in rows if r["mechanism"] == "greedy"}
    if left.keys() != right.keys():
        raise ValueError("Bootstrap requires matched trial seeds")
    return {m: _bootstrap_ci([left[s][m] - right[s][m] for s in sorted(left)])
            for m in ("makespan", "mean_wait", "p95_wait", "fairness", "messages", "throughput", "proxy_tokens")}


def run_paper_experiment(number: int, output_dir="outputs", trials=None):
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    n_trials = (30 if number == 1 else 20) if trials is None else trials
    if n_trials < 1:
        raise ValueError("Need at least one trial")
    rows, groups, optimal_records = [], {}, []
    for point, overrides, mechanisms in _points(number):
        config = ScenarioConfig(n_agents=overrides.get("n_agents", 12), version='legacy')
        settings = {"k": overrides.get("k", 5), "batch_window": 25,
                    "noise": overrides.get("noise", 0.0), "nonresponse": overrides.get("nonresponse", 0.0),
                    "rebid": overrides.get("rebid", False), "rebid_threshold_factor": 1.25}
        raw = {}
        for seed in range(n_trials):
            scenario = generate_scenario(config, seed)
            optimum = None
            if number == 4:
                scenario.tasks = scenario.tasks[:11]
                optimum = exact_optimum(scenario)
                optimal_records.append({"seed": seed, **asdict(optimum)})
            for name in mechanisms:
                result = simulate(scenario, name, **settings)
                row = {"experiment": number, "point": point, "seed": seed, "mechanism": name,
                       **asdict(config), **settings, **result.metrics,
                       "loads": [a.busy_time for a in result.agents],
                       "message_types": result.ledger.counts_by_type(),
                       "messages_per_task": result.ledger.counts_by_task()}
                if optimum is not None:
                    row["optimal_makespan"] = optimum.makespan
                    row["optimal_ratio"] = result.metrics["makespan"] / optimum.makespan if optimum.makespan else 1.0
                    if row["optimal_ratio"] < 1 - 1e-10:
                        raise AssertionError("Mechanism beat the claimed exact optimum")
                rows.append(row)
                raw.setdefault(PAPER_NAMES[name], []).append(row)
                if number == 1 and seed == 0:
                    _save_audit(output_dir, f"experiment1_seed0_{name}", result)
            if number == 4:
                print(f"Experiment 4 seed {seed}: exact optimum {optimum.makespan:.3f}", flush=True)
        groups[point] = aggregate(raw)
        print(f"Experiment {number}, {point}: {n_trials} trials completed", flush=True)
    summary = {"metadata": {"experiment": number, "trials_per_point": n_trials,
                            "seed_scheme": "default_rng(seed), seed=0..trials-1; agents then tasks; CNP +100000, rebid +200000",
                            "std_ddof": 0, "numpy": np.__version__, "python": platform.python_version(),
                            "token_costs": TOKEN_COSTS, "paper_token_costs": PAPER_TOKEN_COSTS,
                            "scenario": asdict(ScenarioConfig(version='legacy')), "batch_window": 25,
                            "source": "June recovered base; reconstructed SSI/CBBA/exact solver"}}
    if number == 1:
        summary.update(aggregate=groups["base"], bootstrap_cnp_vs_gc=_bootstrap(rows))
    elif number == 2:
        summary["by_team_size"] = groups
        summary["bootstrap_cnp_vs_gc"] = {p: _bootstrap([r for r in rows if r["point"] == p]) for p in groups}
    elif number == 3:
        summary["by_k"] = {p: group["Contract-Net"] for p, group in groups.items()}
    elif number == 4:
        summary["aggregate"] = groups["small"]
        summary["ratio_to_optimal"] = {name: metrics["optimal_ratio"] for name, metrics in groups["small"].items()}
        summary["optima"] = optimal_records
    else:
        summary["results"] = {mode: {p.split("/")[1]: group["Contract-Net"] for p, group in groups.items() if p.startswith(mode + "/")}
                              for mode in ("single_round", "rebid", "nonresponse")}
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (results_dir / f"experiment_{number}_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    (results_dir / f"experiment_{number}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Update only summaries which exist; plotting explicitly requires all five.
    combined = {f"experiment_{i}": json.loads(p.read_text(encoding="utf-8")) for i in range(1, 6)
                if (p := results_dir / f"experiment_{i}_summary.json").exists()}
    (results_dir / "results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return summary
