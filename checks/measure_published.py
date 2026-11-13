"""One-time pre-correction audit; refuses to overwrite the preserved evidence."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_experiments as R
from mas_sim import mechanisms as M
from mas_sim.core import clone_agents, clone_tasks


def main():
    destination = R.RESULTS_DIR / "published_audit.json"
    if destination.exists():
        raise SystemExit("Audit already preserved; do not regenerate with corrected code.")
    source = inspect.getsource(M.run_contract_net)
    if "no_responder_fallbacks" in source:
        raise SystemExit("Original source required; this pre-edit audit is retired.")
    namespace = dict(vars(M), fallback_events=[])
    source = source.replace("if not bids:\n", "if not bids:\n                fallback_events.append(task.task_id)\n")
    exec(source, namespace)
    cnp = namespace["run_contract_net"]
    audit = {}

    def measure(label, trials, offset, n=12, k=5, **kwargs):
        rows = []
        for seed in range(trials):
            horizon, expected = (60.0, 9) if label == "experiment_4" else (500.0, 200)
            team, tasks = R.scenario(offset + seed, n, expected, horizon)
            if label == "experiment_4":
                tasks = tasks[:11]
            namespace["fallback_events"].clear()
            rng_seed = (9500 if "nonresponse" in label else 9000) + seed
            result = cnp(clone_agents(team), clone_tasks(tasks), k_invite=k,
                         rng=np.random.default_rng(rng_seed), **kwargs)
            rows.append({"seed": offset + seed, "n_tasks": len(tasks),
                         "fallbacks": len(namespace["fallback_events"]),
                         "fallback_task_ids": list(namespace["fallback_events"]),
                         "messages": result["messages"]})
        audit[label] = {"fallbacks": sum(r["fallbacks"] for r in rows),
                        "n_tasks": sum(r["n_tasks"] for r in rows), "trials": rows}

    measure("experiment_1", 30, 0)
    for n in (4, 8, 12, 16, 24, 32):
        measure(f"experiment_2.n={n}", 20, 1000, n=n)
    for k in (1, 2, 3, 5, 8, 12):
        measure(f"experiment_3.k={k}", 20, 2000, k=k)
    measure("experiment_4", 20, 3000, n=4, k=3)
    for sigma in (0.0, 0.1, 0.2, 0.3, 0.5):
        for rebid in (False, True):
            measure(f"experiment_5.{('rebid' if rebid else 'single_round')}.{sigma}",
                    20, 4000, bid_noise=sigma, rebid=rebid)
    for p in (0.0, 0.1, 0.3):
        measure(f"experiment_5.nonresponse.{p}", 20, 4000, nonresponse_prob=p)
    old_exp4 = {}
    for name, fn in (("SSI-Auction", M.run_ssi_auction), ("CBBA", M.run_cbba)):
        rows = []
        for seed in range(20):
            team, tasks = R.scenario(3000 + seed, 4, 9, 60.0)
            result = fn(clone_agents(team), clone_tasks(tasks[:11]))
            rows.append({key: result[key] for key in R.METRIC_KEYS})
        old_exp4[name] = {"aggregate": R.aggregate(rows), "trials": rows}
    files = [R.RESULTS_DIR / "results_published_original.json", *R.RESULTS_DIR.glob("llm*.json")]
    output = {"fallbacks": audit, "experiment_4_original": old_exp4,
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print({key: value["fallbacks"] for key, value in audit.items()})


if __name__ == "__main__":
    main()
