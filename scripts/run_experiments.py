"""Run all experiments for the MEDES 2026 paper and generate figures.

Outputs:
    outputs/results/results.json  -- aggregated metrics for every experiment
    outputs/figures/*.{png,pdf,svg} -- publication figures
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mas_sim import (
    clone_agents,
    make_task_stream,
    make_team,
    run_cbba,
    run_contract_net,
    run_exhaustive_optimal,
    run_greedy_centralized,
    run_nearest_agent,
    run_round_robin,
    run_ssi_auction,
    run_state_tracking_manager,
)
from mas_sim.core import clone_tasks, make_private_jobs

RESULTS_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIDE = 100.0
N_SKILLS = 5
HORIZON = 500.0

MECHS = {
    "Round-Robin": run_round_robin,
    "Nearest-Agent": run_nearest_agent,
    "Greedy-Centralized": run_greedy_centralized,
    "Contract-Net": lambda a, t, **kw: run_contract_net(a, t, k_invite=5, **kw),
    "SSI-Auction": run_ssi_auction,
    "CBBA": run_cbba,
}

COLORS = {
    "Round-Robin": "#777777",
    "Nearest-Agent": "#E69F00",
    "Greedy-Centralized": "#D55E00",
    "Contract-Net": "#0072B2",
    "SSI-Auction": "#009E73",
    "CBBA": "#CC79A7",
}

METRIC_KEYS = [
    "makespan", "mean_wait", "p95_wait", "fairness", "messages",
    "throughput", "proxy_tokens", "proxy_latency_ms", "context_pressure",
]


def scenario(seed: int, n_agents: int, expected_tasks: int, horizon: float = HORIZON):
    rng = np.random.default_rng(seed)
    team = make_team(n_agents, N_SKILLS, SIDE, rng)
    stream = make_task_stream(expected_tasks, horizon, N_SKILLS, SIDE, rng)
    return team, stream


def aggregate(trials: list[dict]) -> dict:
    out = {}
    for key in trials[0]:
        vals = np.array([t[key] for t in trials], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 123):
    """95% bootstrap CI for mean(a - b)."""
    rng = np.random.default_rng(seed)
    diffs = a - b
    boots = rng.choice(diffs, size=(n_boot, diffs.size), replace=True).mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


# ---------------------------------------------------------------- Experiment 1
def experiment_1(n_trials: int = 30, n_agents: int = 12, expected_tasks: int = 200):
    print("Experiment 1: strategy comparison")
    per_mech = {name: [] for name in MECHS}
    for seed in range(n_trials):
        team, stream = scenario(seed, n_agents, expected_tasks)
        for name, fn in MECHS.items():
            res = fn(clone_agents(team), clone_tasks(stream))
            per_mech[name].append({k: res[k] for k in METRIC_KEYS})
    agg = {name: aggregate(trials) for name, trials in per_mech.items()}
    boot = {
        "makespan": paired_bootstrap(
            np.array([t["makespan"] for t in per_mech["Contract-Net"]]),
            np.array([t["makespan"] for t in per_mech["Greedy-Centralized"]]),
        ),
        "mean_wait": paired_bootstrap(
            np.array([t["mean_wait"] for t in per_mech["Contract-Net"]]),
            np.array([t["mean_wait"] for t in per_mech["Greedy-Centralized"]]),
        ),
    }
    # Per-agent load distribution for a representative trial (fig 3).
    team, stream = scenario(0, n_agents, expected_tasks)
    loads = {}
    for name, fn in MECHS.items():
        agents = clone_agents(team)
        fn(agents, clone_tasks(stream))
        loads[name] = sorted(a.total_busy_time for a in agents)
    return {"aggregate": agg, "bootstrap_cnp_vs_gc": boot, "loads_trial0": loads,
            "params": {"n_trials": n_trials, "n_agents": n_agents, "expected_tasks": expected_tasks}}


# ---------------------------------------------------------------- Experiment 2
def experiment_2(team_sizes=(4, 8, 12, 16, 24, 32), n_trials: int = 20, expected_tasks: int = 200):
    print("Experiment 2: scalability sweep")
    out = {}
    for n in team_sizes:
        per_mech = {name: [] for name in MECHS}
        for seed in range(n_trials):
            team, stream = scenario(1000 + seed, n, expected_tasks)
            for name, fn in MECHS.items():
                res = fn(clone_agents(team), clone_tasks(stream))
                per_mech[name].append({k: res[k] for k in METRIC_KEYS})
        out[str(n)] = {name: aggregate(trials) for name, trials in per_mech.items()}
        print(f"  n={n} done")
    return {"by_team_size": out,
            "params": {"team_sizes": list(team_sizes), "n_trials": n_trials, "expected_tasks": expected_tasks}}


# ---------------------------------------------------------------- Experiment 3
def experiment_3(k_values=(1, 2, 3, 5, 8, 12), n_trials: int = 20, n_agents: int = 12, expected_tasks: int = 200):
    print("Experiment 3: invitation-list ablation")
    out = {}
    for k in k_values:
        trials = []
        for seed in range(n_trials):
            team, stream = scenario(2000 + seed, n_agents, expected_tasks)
            res = run_contract_net(clone_agents(team), clone_tasks(stream), k_invite=k)
            trials.append({m: res[m] for m in METRIC_KEYS})
        out[str(k)] = aggregate(trials)
    return {"by_k": out, "params": {"k_values": list(k_values), "n_trials": n_trials,
                                    "n_agents": n_agents, "expected_tasks": expected_tasks}}


# ---------------------------------------------------------------- Experiment 4
def experiment_4(n_trials: int = 20, n_agents: int = 4, expected_tasks: int = 9,
                 horizon: float = 60.0, max_tasks: int = 11):
    print("Experiment 4: optimality gap on small instances")
    mechs = {
        "Optimal (exhaustive)": run_exhaustive_optimal,
        "Greedy-Centralized": run_greedy_centralized,
        "Contract-Net": lambda a, t, **kw: run_contract_net(a, t, k_invite=3, **kw),
        "SSI-Auction": run_ssi_auction,
        "CBBA": run_cbba,
        "Nearest-Agent": run_nearest_agent,
        "Round-Robin": run_round_robin,
    }
    ratios = {name: [] for name in mechs if name != "Optimal (exhaustive)"}
    makespans = {name: [] for name in mechs}
    per_mech = {name: [] for name in mechs}
    for seed in range(n_trials):
        team, stream = scenario(3000 + seed, n_agents, expected_tasks, horizon)
        stream = stream[:max_tasks]
        if not stream:
            continue
        opt = run_exhaustive_optimal(clone_agents(team), clone_tasks(stream))
        makespans["Optimal (exhaustive)"].append(opt["makespan"])
        per_mech["Optimal (exhaustive)"].append({k: opt[k] for k in METRIC_KEYS})
        for name, fn in mechs.items():
            if name == "Optimal (exhaustive)":
                continue
            res = fn(clone_agents(team), clone_tasks(stream))
            makespans[name].append(res["makespan"])
            per_mech[name].append({k: res[k] for k in METRIC_KEYS})
            ratios[name].append(res["makespan"] / opt["makespan"])
    return {
        "aggregate": {name: aggregate(rows) for name, rows in per_mech.items()},
        "ratio_to_optimal": {n: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                                 "values": [float(x) for x in v]} for n, v in ratios.items()},
        "makespans": {n: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                      for n, v in makespans.items()},
        "params": {"n_trials": n_trials, "n_agents": n_agents, "expected_tasks": expected_tasks,
                   "horizon": horizon, "max_tasks": max_tasks},
    }


# ---------------------------------------------------------------- Experiment 5
def experiment_5(noise_levels=(0.0, 0.1, 0.2, 0.3, 0.5), n_trials: int = 20,
                 n_agents: int = 12, expected_tasks: int = 200):
    print("Experiment 5: bid noise and multi-round re-bidding")
    out = {"single_round": {}, "rebid": {}}
    for sigma in noise_levels:
        for mode, rebid in (("single_round", False), ("rebid", True)):
            trials = []
            for seed in range(n_trials):
                team, stream = scenario(4000 + seed, n_agents, expected_tasks)
                res = run_contract_net(
                    clone_agents(team), clone_tasks(stream), k_invite=5,
                    bid_noise=sigma, rebid=rebid,
                    rng=np.random.default_rng(9000 + seed),
                )
                trials.append({m: res[m] for m in METRIC_KEYS + ["rebids"]})
            out[mode][str(sigma)] = aggregate(trials)
    # Nonresponse robustness (single round).
    out["nonresponse"] = {}
    for p in (0.0, 0.1, 0.3):
        trials = []
        for seed in range(n_trials):
            team, stream = scenario(4000 + seed, n_agents, expected_tasks)
            res = run_contract_net(
                clone_agents(team), clone_tasks(stream), k_invite=5,
                nonresponse_prob=p, rng=np.random.default_rng(9500 + seed),
            )
            trials.append({m: res[m] for m in METRIC_KEYS + ["rebids"]})
        out["nonresponse"][str(p)] = aggregate(trials)
    return {"results": out, "params": {"noise_levels": list(noise_levels), "n_trials": n_trials,
                                       "n_agents": n_agents, "expected_tasks": expected_tasks}}


# ---------------------------------------------------------------- Experiment 6
def aggregate_with_ci(trials: list[dict]) -> dict:
    out = aggregate(trials)
    for key in out:
        values = np.array([row[key] for row in trials])
        ci = paired_bootstrap(values, np.zeros_like(values))
        out[key].update(ci_low=ci["ci_low"], ci_high=ci["ci_high"])
    return out


def experiment_6(load_levels=(0.0, 0.1, 0.2, 0.3, 0.5),
                 refresh_intervals=(1, 10, 50, None), n_trials=30,
                 n_agents=12, expected_tasks=200):
    print("Experiment 6: when bidding pays")
    out = {}
    metrics = ["makespan", "mean_wait", "mean_response", "messages_per_task"]
    for load in load_levels:
        per_mech = {"Greedy-Centralized": [], "Contract-Net": []}
        managers = {f"Manager-{s if s is not None else 'never'}": s
                    for s in refresh_intervals}
        per_mech.update({name: [] for name in managers})
        workloads = []
        for trial in range(n_trials):
            team, stream = scenario(5000 + trial, n_agents, expected_tasks)
            jobs = make_private_jobs(n_agents, stream[-1].arrival_time if stream else 0.0,
                                     load, np.random.default_rng(15000 + trial))
            workloads.append({"seed": 5000 + trial, "private_seed": 15000 + trial,
                              "n_tasks": len(stream), "private_jobs": len(jobs),
                              "private_service": sum(job[2] for job in jobs)})
            for name in per_mech:
                tasks = clone_tasks(stream)
                agents = clone_agents(team)
                if name == "Greedy-Centralized":
                    result = run_greedy_centralized(agents, tasks, private_jobs=jobs,
                                                  count_awards=True)
                elif name == "Contract-Net":
                    result = run_contract_net(agents, tasks, private_jobs=jobs, k_invite=5,
                                              rng=np.random.default_rng(25000 + trial))
                else:
                    result = run_state_tracking_manager(agents, tasks, private_jobs=jobs,
                                                         refresh_every=managers[name])
                result["messages_per_task"] = result["messages"] / len(tasks) if tasks else 0.0
                result["mean_response"] = float(np.mean([
                    t.finish_time - t.arrival_time for t in tasks])) if tasks else 0.0
                per_mech[name].append({key: result[key] for key in metrics})
        paired = {}
        for name in managers:
            paired[name] = {key: paired_bootstrap(
                np.array([row[key] for row in per_mech[name]]),
                np.array([row[key] for row in per_mech["Contract-Net"]])) for key in metrics}
        out[str(load)] = {"aggregate": {name: aggregate_with_ci(rows)
                                       for name, rows in per_mech.items()},
                          "paired_manager_minus_cnp": paired,
                          "trials": per_mech, "workloads": workloads}
        print(f"  private load={load} done")
    return {"by_private_load": out,
            "params": {"load_levels": list(load_levels), "refresh_intervals": list(refresh_intervals),
                       "n_trials": n_trials, "n_agents": n_agents, "expected_tasks": expected_tasks,
                       "horizon": HORIZON, "scenario_seed_offset": 5000,
                       "private_seed_offset": 15000, "cnp_seed_offset": 25000,
                       "private_interarrival_mean": 25.0, "bootstrap_seed": 123,
                       "bootstrap_resamples": 10000,
                       "polling_includes_award": True}}


# -------------------------------------------------------------------- Figures
# Exact 122 mm canvas width: no downscaling required; all text >= 7.2 pt.
FIG_WIDTH = 122 / 25.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.2, "pdf.fonttype": 42, "svg.fonttype": "none"})
STYLES = dict(zip(MECHS, [("o", "-"), ("s", "--"), ("^", "-."),
                          ("D", ":"), ("v", (0, (5, 1))), ("P", (0, (3, 1, 1, 1)))]))
SHORT = {"Round-Robin": "RR", "Nearest-Agent": "Nearest", "Greedy-Centralized": "Full polling",
         "Contract-Net": "CNP k=5", "SSI-Auction": "SSI", "CBBA": "CBBA"}


def save_figure(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def line(ax, x, y, name, **kwargs):
    marker, style = STYLES[name]
    return ax.plot(x, y, marker=marker, linestyle=style, markersize=4,
                   linewidth=1.2, color=COLORS[name], label=SHORT[name], **kwargs)


def finish_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)


def fig_strategy_comparison(exp1):
    names = list(MECHS)
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, 6.4), layout="constrained")
    for ax, metric, title in zip(axes, ["makespan", "mean_wait", "messages"],
                                 ["Makespan (time units)", "Mean wait (time units)", "Messages per trial"]):
        bars = ax.barh(range(len(names)), [exp1["aggregate"][n][metric]["mean"] for n in names],
                       xerr=[exp1["aggregate"][n][metric]["std"] for n in names],
                       color=[COLORS[n] for n in names], capsize=2)
        for bar, hatch in zip(bars, ["/", "\\", "x", ".", "-", "+"]):
            bar.set_hatch(hatch)
        ax.set_yticks(range(len(names)), labels=[SHORT[n] for n in names])
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "fig2_strategy_comparison")


def fig_load_distribution(exp1):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3), layout="constrained")
    for name, loads in exp1["loads_trial0"].items():
        line(ax, range(1, len(loads) + 1), loads, name)
    ax.set_xlabel("Agent (sorted by load)")
    ax.set_ylabel("Busy time (travel + execution)")
    finish_axes([ax])
    save_figure(fig, "fig3_load_distribution")


def fig_scalability(exp2):
    data = exp2["by_team_size"]
    sizes = sorted(map(int, data))
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 5.4), layout="constrained")
    for name in MECHS:
        for ax, metric in zip(axes, ["makespan", "messages"]):
            line(ax, sizes, [data[str(n)][name][metric]["mean"] for n in sizes], name)
    for ax in axes:
        ax.set_xlabel("Team size n")
        ax.set_xticks(sizes)
    axes[0].set_ylabel("Makespan (time units)")
    axes[1].set_ylabel("Messages per trial")
    finish_axes(axes)
    save_figure(fig, "fig4_scalability")


def fig_ablation(exp3):
    data = exp3["by_k"]
    ks = sorted(map(int, data))
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.7), layout="constrained")
    axes[0].errorbar(ks, [data[str(k)]["makespan"]["mean"] for k in ks],
                     yerr=[data[str(k)]["makespan"]["std"] for k in ks],
                     marker="o", color=COLORS["Contract-Net"], capsize=3, label="Makespan +/- SD")
    axes[0].set_ylabel("Makespan (time units)")
    axes[1].plot(ks, [data[str(k)]["messages"]["mean"] for k in ks],
                 marker="s", color="#D55E00", label="Messages")
    axes[1].plot(ks, [data[str(k)]["proxy_tokens"]["mean"] / 40 for k in ks],
                 marker="^", linestyle="--", color="#009E73", label="Modelled tokens / 40")
    axes[1].set_ylabel("Count / scaled token count")
    for ax in axes:
        ax.set_xlabel("Invitation-list size k")
        ax.set_xticks(ks)
    finish_axes(axes)
    save_figure(fig, "fig5_ablation_k")


def fig_token_proxy(exp1):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3), layout="constrained")
    for name in MECHS:
        row = exp1["aggregate"][name]
        ax.scatter(row["messages"]["mean"], row["proxy_tokens"]["mean"],
                    color=COLORS[name], marker=STYLES[name][0], label=SHORT[name], s=35)
    ax.set_xlabel("Messages per trial")
    ax.set_ylabel("Modelled token count")
    finish_axes([ax])
    save_figure(fig, "fig6_token_proxy")


def fig_robustness(exp5):
    data = exp5["results"]
    sigmas = sorted(map(float, data["single_round"]))
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.8), layout="constrained")
    for mode, label, color, marker, style in [
        ("single_round", "Single round", "#0072B2", "o", "-"),
        ("rebid", "Re-bidding", "#D55E00", "s", "--")]:
        for ax, metric in zip(axes, ["makespan", "messages"]):
            ax.errorbar(sigmas, [data[mode][str(s)][metric]["mean"] for s in sigmas],
                         yerr=[data[mode][str(s)][metric]["std"] for s in sigmas],
                         color=color, marker=marker, linestyle=style, label=label, capsize=3)
    axes[0].set_ylabel("Makespan (time units)")
    axes[1].set_ylabel("Messages per trial")
    for ax in axes:
        ax.set_xlabel("Bid noise sigma")
    finish_axes(axes)
    save_figure(fig, "fig7_robustness")


def fig_optimality_gap(exp4):
    data = exp4["ratio_to_optimal"]
    names = list(data)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.4), layout="constrained")
    labels = ["CNP k=3" if n == "Contract-Net" else SHORT[n] for n in names]
    bp = ax.boxplot([data[n]["values"] for n in names], tick_labels=labels,
                     patch_artist=True, showmeans=True, vert=False)
    for patch, name, hatch in zip(bp["boxes"], names, ["/", "x", ".", "-", "+", "\\"]):
        patch.set_facecolor(COLORS[name])
        patch.set_hatch(hatch)
        patch.set_alpha(0.65)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Makespan / arrival-order optimum")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "fig8_optimality_gap")


def fig_when_bidding_pays(exp6):
    data = exp6["by_private_load"]
    loads = sorted(map(float, data))
    names = list(data["0.0"]["aggregate"])
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#000000"]
    markers = ["^", "D", "o", "s", "v", "P"]
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 5.5), layout="constrained")
    for name, color, marker, style in zip(names, colors, markers, styles):
        label = SHORT.get(name, name.replace("Manager-", "Manager s="))
        for ax, metric in zip(axes, ["mean_wait", "messages_per_task"]):
            values = [data[str(load)]["aggregate"][name][metric] for load in loads]
            means = np.array([v["mean"] for v in values])
            errors = np.array([[v["mean"] - v["ci_low"] for v in values],
                               [v["ci_high"] - v["mean"] for v in values]])
            ax.errorbar(loads, means, yerr=np.maximum(errors, 0), color=color, marker=marker,
                         linestyle=style, label=label, capsize=2, markersize=4, linewidth=1.2)
    axes[0].set_ylabel("Mean public-task wait (time units)")
    axes[0].set_title("When bidding pays: means and 95% bootstrap CIs")
    axes[1].set_ylabel("Messages per public task")
    for ax in axes:
        ax.set_xlabel("Private offered load (per agent)")
        ax.set_xticks(loads)
    finish_axes(axes)
    save_figure(fig, "fig10_when_bidding_pays")


def fig_live_llm():
    """Plot preserved paid-run results only; never invokes an LLM."""
    sizes = [8, 16, 24]
    data = {n: json.loads((RESULTS_DIR / f"llm_sweep_n{n}.json").read_text()) for n in sizes}
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.6), layout="constrained")
    for condition, label, color, marker, style in [
        ("llm_cnp", "Stored LLM CNP k=3", "#0072B2", "o", "-"),
        ("llm_centralized", "Stored LLM full-state", "#D55E00", "s", "--")]:
        tokens = [sum(t[condition]["llm"]["total_tokens"] for t in data[n]["trials"]) /
                  sum(t[condition]["n_tasks"] for t in data[n]["trials"]) for n in sizes]
        makespans = [np.mean([t[condition]["makespan"] for t in data[n]["trials"]]) for n in sizes]
        for ax, values in zip(axes, [tokens, makespans]):
            ax.plot(sizes, values, label=label, color=color, marker=marker, linestyle=style)
    axes[0].set_ylabel("Measured tokens per public task")
    axes[1].set_ylabel("Makespan (time units)")
    for ax in axes:
        ax.set_xlabel("Team size n")
        ax.set_xticks(sizes)
    finish_axes(axes)
    save_figure(fig, "fig9_llm_live")


def main():
    t0 = time.time()
    results = {
        "experiment_1": experiment_1(),
        "experiment_2": experiment_2(),
        "experiment_3": experiment_3(),
        "experiment_4": experiment_4(),
        "experiment_5": experiment_5(),
        "experiment_6": experiment_6(),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    fig_strategy_comparison(results["experiment_1"])
    fig_load_distribution(results["experiment_1"])
    fig_scalability(results["experiment_2"])
    fig_ablation(results["experiment_3"])
    fig_token_proxy(results["experiment_1"])
    fig_robustness(results["experiment_5"])
    fig_optimality_gap(results["experiment_4"])
    fig_when_bidding_pays(results["experiment_6"])
    fig_live_llm()
    from correction_report import write_report
    write_report()
    print(f"All experiments and figures complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
