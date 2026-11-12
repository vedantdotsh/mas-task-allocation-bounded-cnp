"""Seven print-sized figures from saved results, without rerunning simulations."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiments import PAPER_NAMES

NAMES = list(PAPER_NAMES.values())
SHORT = ["RR", "NA", "GC", "CNP", "SSI", "CBBA"]
COLORS = ["#666666", "#E69F00", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
HATCHES = ["///", "...", "xx", "\\\\", "++", "oo"]
WIDTH = 12 / 2.54
FIGURE_NAMES = {
    "1": "fig4_scalability", "2": "fig1_strategy_comparison", "3": "fig3_load_balance",
    "4": "fig5_ablation_k", "5": "fig8_optimality_gap", "6": "fig7_robustness",
    "7": "fig6_communication_proxy",
}


def _figure(rows=1, height=None):
    fig, axes = plt.subplots(rows, 1, figsize=(WIDTH, height or 2.55 * rows), squeeze=False,
                             layout="constrained")
    return fig, list(axes[:, 0])


def _bars(ax, group, metric, label, scale=1):
    means = [group[n][metric]["mean"] / scale for n in NAMES]
    std = [group[n][metric]["std"] / scale for n in NAMES]
    bars = ax.bar(SHORT, means, yerr=std, capsize=3, color=COLORS, edgecolor="black", linewidth=0.6)
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)


def _save(fig, directory, number):
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{FIGURE_NAMES[str(number)]}.{extension}"
        # No tight bounding box: preserve the exact 12 cm physical width and font scale.
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def generate_all_figures(output_dir="outputs"):
    output_dir = Path(output_dir)
    data_path = output_dir / "results" / "results.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if any(f"experiment_{i}" not in data for i in range(1, 6)):
        raise ValueError("Run all five experiments before generating figures")
    directory = output_dir / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2})
    paths = []
    strategy = data["experiment_1"]["aggregate"]

    fig, axes = _figure(2, 6.1)
    sweep = data["experiment_2"]["by_team_size"]
    sizes = sorted(map(int, sweep))
    for ax, metric, label, scale in zip(axes, ["makespan", "messages"], ["Makespan", "Messages (thousands)"], [1, 1000]):
        for name, short, color, marker in zip(NAMES, SHORT, COLORS, MARKERS):
            ax.errorbar(sizes, [sweep[str(n)][name][metric]["mean"] / scale for n in sizes],
                        yerr=[sweep[str(n)][name][metric]["std"] / scale for n in sizes],
                        marker=marker, color=color, label=short, linewidth=1.2, markersize=5, capsize=2)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(label)
        ax.set_xticks(sizes)
        ax.set_ylim(bottom=0)
    axes[0].legend(ncol=3, loc="upper right", columnspacing=0.6, handlelength=1)
    paths += _save(fig, directory, 1)

    fig, axes = _figure(3, 7.1)
    for ax, metric, label, scale in zip(axes, ["makespan", "mean_wait", "messages"],
                                       ["Makespan", "Mean service wait", "Messages (thousands)"], [1, 1, 1000]):
        _bars(ax, strategy, metric, label, scale)
    paths += _save(fig, directory, 2)

    with (output_dir / "results" / "experiment_1_raw.csv").open(encoding="utf-8", newline="") as handle:
        representative = {PAPER_NAMES[r["mechanism"]]: json.loads(r["loads"]) for r in csv.DictReader(handle) if r["seed"] == "0"}
    fig, axes = _figure(1, 3.8)
    ax = axes[0]
    for name, short, color, marker in zip(NAMES, SHORT, COLORS, MARKERS):
        loads = sorted(representative[name])
        ax.plot(range(1, len(loads) + 1), loads, color=color, marker=marker, label=short, markersize=4)
    ax.set_xlabel("Agent rank by load (seed 0)")
    ax.set_ylabel("Busy time: travel + execution")
    ax.legend(ncol=3, columnspacing=0.6, handlelength=1)
    paths += _save(fig, directory, 3)

    ablation = data["experiment_3"]["by_k"]
    ks = sorted(map(int, ablation))
    fig, axes = _figure(3, 7.1)
    for ax, metric, label, scale in zip(axes, ["makespan", "messages", "proxy_tokens"],
                                       ["Makespan", "Messages (thousands)", "Token proxy (thousands)"], [1, 1000, 1000]):
        ax.errorbar(ks, [ablation[str(k)][metric]["mean"] / scale for k in ks],
                    yerr=[ablation[str(k)][metric]["std"] / scale for k in ks], color=COLORS[3], marker="D", capsize=3)
        ax.set_ylabel(label)
        ax.set_xticks(ks)
    axes[-1].set_xlabel("Invitation-list size k")
    paths += _save(fig, directory, 4)

    fig, axes = _figure(1, 3.5)
    _bars(axes[0], data["experiment_4"]["aggregate"], "optimal_ratio", "Makespan / offline optimum")
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    paths += _save(fig, directory, 5)

    robust = data["experiment_5"]["results"]
    fig, axes = _figure(3, 7.6)
    for ax, metric, label in zip(axes[:2], ["makespan", "messages"], ["Makespan", "Messages"]):
        for mode, label_mode, color, marker in [("single_round", "Single round", COLORS[2], "o"), ("rebid", "One re-bid", COLORS[3], "D")]:
            sigmas = sorted(map(float, robust[mode]))
            ax.errorbar(sigmas, [robust[mode][str(s)][metric]["mean"] for s in sigmas],
                        yerr=[robust[mode][str(s)][metric]["std"] for s in sigmas],
                        color=color, marker=marker, label=label_mode, capsize=2)
        ax.set_ylabel(label)
        ax.set_xlabel("Bid-noise standard deviation")
    axes[0].legend()
    probs = sorted(map(float, robust["nonresponse"]))
    axes[2].errorbar(probs, [robust["nonresponse"][str(p)]["makespan"]["mean"] for p in probs],
                    yerr=[robust["nonresponse"][str(p)]["makespan"]["std"] for p in probs],
                    color=COLORS[3], marker="s", capsize=3)
    axes[2].set_xlabel("Candidate nonresponse probability")
    axes[2].set_ylabel("Makespan")
    paths += _save(fig, directory, 6)

    fig, axes = _figure(2, 5.3)
    _bars(axes[0], strategy, "messages", "Messages (thousands)", 1000)
    _bars(axes[1], strategy, "proxy_tokens", "Token proxy (thousands)", 1000)
    paths += _save(fig, directory, 7)
    (directory / "manifest.json").write_text(json.dumps({"paper_figure_to_filename": FIGURE_NAMES,
        "width_cm": 12, "font_pt": 11, "png_dpi": 300, "error_bars": "trial population standard deviation",
        "token_model": "recovered June constants", "abbreviations": dict(zip(SHORT, NAMES))}, indent=2), encoding="utf-8")
    return paths
