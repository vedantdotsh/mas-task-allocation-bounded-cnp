"""Always use paired bootstrap, independent of optional installed packages."""
import numpy as np


def _bootstrap_ci(diff, *, seed=12345, n_boot=10000):
    diff = np.asarray(diff, dtype=float)
    if not diff.size or not np.all(np.isfinite(diff)):
        raise ValueError("Need finite paired differences")
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"method": "paired_percentile_bootstrap", "n_pairs": len(diff),
            "n_bootstrap": n_boot, "seed": seed, "mean_difference": float(diff.mean()),
            "ci95": [float(lo), float(hi)], "ci_low": float(lo), "ci_high": float(hi)}


def paired_comparisons(raw_results, *, left="contract_net", right="greedy_centralized",
                       metrics=("makespan", "avg_wait", "messages")):
    output = {"comparison": f"{left} minus {right}", "metrics": {}}
    if len(raw_results[left]) != len(raw_results[right]):
        raise ValueError("Unpaired trial counts")
    for metric in metrics:
        diff = [a[metric] - b[metric] for a, b in zip(raw_results[left], raw_results[right])]
        output["metrics"][metric] = _bootstrap_ci(diff)
    return output
