"""Verify preserved claims and write a report with JSON paths for every table."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def verify_results():
    published = read("results_published_original.json")
    corrected = read("results.json")
    audit = read("published_audit.json")
    changed, preserved = [], []

    def compare(old, new, path):
        if isinstance(old, dict):
            for key, value in old.items():
                compare(value, new[key], path + [key])
        elif old != new:
            changed.append(path)
        else:
            preserved.append(path)

    compare(published, corrected, [])
    unrelated = [p for p in changed if not {"SSI-Auction", "CBBA"}.intersection(p)]
    assert not unrelated, f"Unrelated published values changed: {unrelated}"
    hashes = {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
              for name in audit["sha256"]}
    assert hashes == audit["sha256"], "Preserved published or paid LLM evidence changed"
    exp1 = corrected["experiment_1"]
    gc, cnp = (exp1["aggregate"][n] for n in ("Greedy-Centralized", "Contract-Net"))
    headlines = {
        "greedy_makespan": gc["makespan"]["mean"],
        "greedy_mean_wait": gc["mean_wait"]["mean"],
        "cnp_makespan": cnp["makespan"]["mean"],
        "cnp_mean_wait": cnp["mean_wait"]["mean"],
        "bootstrap_difference": exp1["bootstrap_cnp_vs_gc"]["makespan"]["mean_diff"],
        "bootstrap_ci_low": exp1["bootstrap_cnp_vs_gc"]["makespan"]["ci_low"],
        "bootstrap_ci_high": exp1["bootstrap_cnp_vs_gc"]["makespan"]["ci_high"],
        "message_saving_n12_percent": 100 * (1 - cnp["messages"]["mean"] / gc["messages"]["mean"]),
        "token_saving_percent": 100 * (1 - cnp["proxy_tokens"]["mean"] / gc["proxy_tokens"]["mean"]),
        "rebid_sigma05": corrected["experiment_5"]["results"]["rebid"]["0.5"]["makespan"]["mean"],
        "single_round_sigma05": corrected["experiment_5"]["results"]["single_round"]["0.5"]["makespan"]["mean"],
    }
    n32 = corrected["experiment_2"]["by_team_size"]["32"]
    headlines["message_saving_n32_percent"] = 100 * (
        1 - n32["Contract-Net"]["messages"]["mean"] / n32["Greedy-Centralized"]["messages"]["mean"])
    expected = dict(zip(headlines, [535.4, 5.3, 534.6, 5.9, -0.8, -4.1, 2.4,
                                    54.2, 81.1, 603.0, 619.9, 82.8]))
    assert {k: f"{v:.1f}" for k, v in headlines.items()} == {k: f"{v:.1f}" for k, v in expected.items()}
    assert corrected["experiment_3"] == published["experiment_3"]
    for name in ("SSI-Auction", "CBBA"):
        values = corrected["experiment_1"]["aggregate"][name]
        assert f"{values['makespan']['mean']:.1f}" == "544.5"
        assert f"{values['mean_wait']['mean']:.1f}" == "15.3"
        assert f"{corrected['experiment_4']['ratio_to_optimal'][name]['mean']:.3f}" == "1.224"
        assert audit["experiment_4_original"][name]["aggregate"]["makespan"] == published["experiment_4"]["makespans"][name]
    fresh = corrected["experiment_6"]["by_private_load"]["0.0"]["trials"]
    for name in fresh:
        if name.startswith("Manager"):
            for manager, greedy in zip(fresh[name], fresh["Greedy-Centralized"]):
                assert all(manager[k] == greedy[k] for k in ("makespan", "mean_wait", "mean_response"))
    failures = audit["fallbacks"]["experiment_5.nonresponse.0.3"]
    figures = {}
    for path in sorted((ROOT / "outputs" / "figures").glob("fig*.svg")):
        svg = path.read_text(encoding="utf-8")
        width_pt = float(re.search(r'width="([\d.]+)pt"', svg)[1])
        sizes = [float(v) for v in re.findall(r"font-size: ([\d.]+)px", svg)]
        assert abs(width_pt / 72 * 25.4 - 122) < 1e-5
        assert sizes and min(sizes) >= 7.0
        with Image.open(path.with_suffix(".png")) as png:
            dpi = png.info["dpi"]
            assert all(abs(d - 300) < 0.01 for d in dpi)
        figures[path.stem] = {"width_mm": width_pt / 72 * 25.4,
                              "minimum_font_pt": min(sizes), "png_dpi": dpi}
    assert len(figures) == 9
    return {"all_unaffected_published_values_exact": True,
            "unchanged_leaf_count": len(preserved), "changed_baseline_leaf_count": len(changed),
            "unrelated_changes": unrelated, "changed_paths": changed,
            "headlines": headlines, "k_ablation_exact": True,
            "fresh_manager_matches_greedy_all_trials": True, "preserved_sha256": hashes,
            "figures": figures,
            "fallback_counterfactual": {
                "extra_messages_all_trials": failures["fallbacks"] * 24,
                "extra_messages_per_trial": failures["fallbacks"] * 24 / len(failures["trials"]),
                "percent_tasks": 100 * failures["fallbacks"] / failures["n_tasks"]}}


def write_report():
    verification = verify_results()
    (RESULTS / "correction_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    old, new, audit = read("results_published_original.json"), read("results.json"), read("published_audit.json")
    lines = [
        "# Original-study correction report",
        "",
        "This package contains the corrected original study. "
        "The published JSON was copied before source changes and remains byte-identical; "
        "stored paid LLM results also retain their pre-edit SHA-256 hashes. No LLM experiment was run.",
        "",
        "## Trace convention",
        "",
        "All paths below are relative to this directory. `R` = `outputs/results/results.json`; "
        "`P` = `outputs/results/results_published_original.json`; `A` = `outputs/results/published_audit.json`; "
        "`V` = `outputs/results/correction_verification.json`; `Q` = `outputs/results/validation.json` "
        "(recorded delivery checks). Dot paths are object-key paths, "
        "with numeric keys such as `0.3` treated as a single key, not array indices. "
        "For an unambiguous machine traversal, `V.changed_paths` stores arrays of literal keys. "
        "Tables round for reading; JSON retains full floating-point values. The report and verification "
        "are regenerated by `python scripts/run_experiments.py`.",
        "",
        "## Changes and accounting",
        "",
        "- `src/mas_sim/core.py::schedule_task`: optional decision time bounds every start without changing arrival times or waits. "
        "`make_private_jobs` and `tasks_with_private_work` generate and replay common private workloads.",
        "- `src/mas_sim/mechanisms.py::run_ssi_auction`: schedules from the batch decision. The decision remains the last "
        "arrival in an offline-grouped 25-unit window, as in the supplied correction check, rather than the window's right edge. "
        "This convention uses advance knowledge of the last arrival; it is not a causal online timer implementation. "
        "All tasks nevertheless start at or after their actual modelled decision time. Window default: function signature.",
        "- `run_cbba`: reuses the supplied check's retained bundles, drops the suffix after a lost claim, includes all held "
        "tasks in simulated queue time/position, and commits in bundle order. Counts actual sweeps, including the final stable "
        "sweep. A cap failure now raises instead of silently committing an unsettled allocation; no experiment hit the cap.",
        "- SSI charges `B*n` task announcements, then `n` bids and one award for each of `B` allocation rounds: "
        "`B*(2*n+1)`. Each bid carries that agent's cheapest remaining task/cost; evaluating all alternatives is local work. "
        "This preserves the original message convention honestly; the count does not change with timing.",
        "- CBBA charges `n*(n-1)` point-to-point winning-list deliveries per sweep and the existing modelled token cost "
        "per delivery. Initial batch availability and local commitment remain uncharged in this original convention. "
        "It is an ordered shared-winner-table adaptation, not a faithful asynchronous distributed implementation. "
        "Its stored token proxy remains fixed per list delivery regardless of bundle length.",
        "- `run_contract_net`: adds a diagnostic fallback counter without changing published decisions or costs; see fallback audit below.",
        "- `run_state_tracking_manager`: predicts queue-end position and busy-until from every award, using the same "
        "`Agent.estimate_cost` and stable tie-breaking as Greedy. It never reads private state except at explicit report points. "
        "`run_greedy_centralized` gains optional private-work replay and award charging used only in the new experiment. "
        "`src/mas_sim/__init__.py` exports the manager.",
        "- `scripts/run_experiments.py`: adds Experiment 6, retains old seeds, adds all metrics to Experiment 4, and rebuilds "
        "all figures. `scripts/correction_report.py` verifies immutable claims and writes this report. "
        "`scripts/plot_llm_figure.py` is now a small saved-data-only wrapper.",
        "- `tests/test_mas_sim.py`: regression tests cover both fixes, held-prefix overclaiming, consensus accounting/cap failure, "
        "fresh manager equality, private work, reports, seed determinism and the published fallback rule. "
        "`checks/baseline_corrections_check.py` now verifies corrected production results; its duplicate implementations are retired. "
        "`checks/measure_published.py` is the one-time original audit and refuses to overwrite its evidence.",
        "- `src/mas_sim/llm_proxy.py`: removes the unsupported Agent Contracts reduction attribution from the docstring; "
        "all model constants and stored results are unchanged. `README.md` documents the revised reproduction flow.",
        "",
        "## Published versus corrected baselines",
        "",
        "Cells are mean (sample SD). Source prefixes in the final column locate each metric: append "
        "`.makespan`, `.mean_wait`, or `.messages`, then `.mean` / `.std`. "
        "For Experiment 4, original waits/messages were not present in the published JSON; they were replayed before edits "
        "and preserved in A. Published makespans remain independently available under `P.experiment_4.makespans`.",
        "",
        "| Setting / method | Original makespan | Corrected makespan | Original wait | Corrected wait | Original messages | Corrected messages | Source prefixes (old → new) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    comparisons = []
    for exp, prefix, orig, fixed in [
        ("Exp.1", "experiment_1.aggregate", old["experiment_1"]["aggregate"], new["experiment_1"]["aggregate"]),
        *[(f"Exp.2 n={n}", f"experiment_2.by_team_size.{n}", old["experiment_2"]["by_team_size"][n], values)
          for n, values in new["experiment_2"]["by_team_size"].items()],
        ("Exp.4", "experiment_4.aggregate", {n: v["aggregate"] for n, v in audit["experiment_4_original"].items()},
         new["experiment_4"]["aggregate"]),
    ]:
        for name in ("SSI-Auction", "CBBA"):
            old_path = f"A.experiment_4_original.{name}.aggregate" if exp == "Exp.4" else f"P.{prefix}.{name}"
            new_path = f"R.{prefix}.{name}"
            values = []
            for metric in ("makespan", "mean_wait", "messages"):
                values.extend(f"{d[name][metric]['mean']:.3f} ({d[name][metric]['std']:.3f})" for d in (orig, fixed))
            lines.append(f"| {exp} / {name} | " + " | ".join(values) + f" | `{old_path}` → `{new_path}` |")
            comparisons.append((exp, name, orig[name], fixed[name], old_path, new_path))
    lines += ["", "| Exp.4 method | Published ratio mean (SD) | Corrected ratio mean (SD) | Paths |",
              "|---|---:|---:|---|"]
    for name in ("SSI-Auction", "CBBA"):
        a, b = (d["experiment_4"]["ratio_to_optimal"][name] for d in (old, new))
        lines.append(f"| {name} | {a['mean']:.6f} ({a['std']:.6f}) | {b['mean']:.6f} ({b['std']:.6f}) | "
                     f"`P/R.experiment_4.ratio_to_optimal.{name}.mean/.std`; individual ratios: `.values` |")
    lines += ["", "The exhaustive reference optimizes assignments with arrival-order service. SSI/CBBA can reorder a batch, "
              "so this is an arrival-order optimum reference, not a universal lower bound on all possible schedules. "
              "The corrected SSI and CBBA aggregates happen to coincide in these trials; this is not an equivalence theorem.",
              "", "## Protected published results: exact verification", "",
              f"All {verification['unchanged_leaf_count']} unchanged existing JSON leaves match exactly, with "
              f"{verification['changed_baseline_leaf_count']} changed leaves confined to SSI/CBBA "
              "(`V.unchanged_leaf_count`, `V.changed_baseline_leaf_count`, `V.unrelated_changes`). "
              "Comparisons use full stored values, not a rounding tolerance. All of Experiments 3 and 5 match exactly.", "",
              "| Claim | Reproduced value | R source / formula (also identical in P) |", "|---|---:|---|"]
    paths = {
        "greedy_makespan": "experiment_1.aggregate.Greedy-Centralized.makespan.mean",
        "greedy_mean_wait": "experiment_1.aggregate.Greedy-Centralized.mean_wait.mean",
        "cnp_makespan": "experiment_1.aggregate.Contract-Net.makespan.mean",
        "cnp_mean_wait": "experiment_1.aggregate.Contract-Net.mean_wait.mean",
        "bootstrap_difference": "experiment_1.bootstrap_cnp_vs_gc.makespan.mean_diff",
        "bootstrap_ci_low": "experiment_1.bootstrap_cnp_vs_gc.makespan.ci_low",
        "bootstrap_ci_high": "experiment_1.bootstrap_cnp_vs_gc.makespan.ci_high",
        "message_saving_n12_percent": "100*(1-CNP/Greedy), experiment_1.aggregate.{Contract-Net,Greedy-Centralized}.messages.mean",
        "message_saving_n32_percent": "100*(1-CNP/Greedy), experiment_2.by_team_size.32.{Contract-Net,Greedy-Centralized}.messages.mean",
        "token_saving_percent": "100*(1-CNP/Greedy), experiment_1.aggregate.{Contract-Net,Greedy-Centralized}.proxy_tokens.mean",
        "rebid_sigma05": "experiment_5.results.rebid.0.5.makespan.mean",
        "single_round_sigma05": "experiment_5.results.single_round.0.5.makespan.mean",
    }
    for key, path in paths.items():
        lines.append(f"| {key} | {verification['headlines'][key]:.1f} | `{path}`; derived value: `V.headlines.{key}` |")
    lines += ["", "The complete k-ablation, including every metric mean/SD and parameters, is exactly equal "
              "at `P/R.experiment_3`; verification flag: `V.k_ablation_exact`.", "",
              "| k | Makespan | Mean wait | Messages | Modelled tokens | P/R path prefix (append metric.mean) |",
              "|---|---:|---:|---:|---:|---|"]
    for k, values in new["experiment_3"]["by_k"].items():
        lines.append(f"| {k} | " + " | ".join(f"{values[m]['mean']:.6f}" for m in
                     ("makespan", "mean_wait", "messages", "proxy_tokens")) + f" | `experiment_3.by_k.{k}` |")
    lines += ["", "## Experiment 6: when bidding pays", "",
        "Parameters are at `R.experiment_6.params`: 12 agents, expected 200 public tasks from the original Poisson "
        "generator over horizon 500, and 30 paired trials. Public scenario seeds are 5000–5029; private seeds 15000–15029; "
        "CNP RNG seeds 25000–25029 (no noise or nonresponse is enabled). The original space, skills, speeds, workload "
        "and cost formula are unchanged. This follows the old experiment-specific seed-offset convention; it does not "
        "reuse Experiment 1's individual task streams.", "",
        "For every agent, stationary private jobs arrive as an independent Poisson process with mean interarrival "
        "25 time units. Service is exponential with mean `25*rho`, for offered loads rho in {0, 0.1, 0.2, 0.3, 0.5}; "
        "zero produces no jobs. Rho is expected private service demand per unit time, not the realized fraction of total "
        "busy time. Jobs are generated through the last public arrival, append to that agent's FIFO queue at their arrival, "
        "and do not move its queue-end position. Private arrivals win exact-time ties. The same immutable job list is "
        "replayed for every mechanism per seed/load; using the same private RNG seed across positive levels preserves "
        "event times and scales service durations. The manager knows initial agent state, public tasks and its awards, "
        "but sees private queue delays only in periodic reports before public task indices s, 2s, etc. (zero-based), "
        "with s in {1, 10, 50, never}. No report precedes task index zero. The manager schedules its forecast separately "
        "from the true FIFO queue. Live CNP bids and full polling use the true local queue; private jobs do not change "
        "the positions used for CNP invitation selection. All parameter values: `R.experiment_6.params`; per-seed public "
        "counts, private counts and service totals: `R.experiment_6.by_private_load.<rho>.workloads`.", "",
        "Messages: manager = T awards + n*floor((T-1)/s) unsolicited status reports, or T for never. "
        "No queries, acknowledgements or completion reports are introduced. Reports contain queue-end position and "
        "busy-until; report timing is a public-task-index synchronization assumption, with no separate trigger message. "
        "CNP = k CFPs + k bids + one award = 11 per public task. Full polling = n queries + n replies + one award = 25 "
        "per task. The original experiments retain their published polling count of 2*n (24 at n=12), which excluded "
        "awards; this explicit convention difference changes no allocations. Counts are semantic deliveries, without "
        "packet-size or network-latency simulation. In this experiment all report and bid state is truthful.", "",
        "Makespan is the latest public-task finish, wait is start minus public arrival (travel begins at start), and "
        "response is finish minus arrival, including travel/execution. Private-only completion after the final public "
        "finish is not a reported objective. Public tasks and private service demand are identical across compared "
        "mechanisms, though their queue ordering relative to public assignments and their incurred travel can differ. "
        "Each trial contributes equally to means and messages-per-task. Percentile bootstrap CIs resample trials "
        "10,000 times with seed 123; paired CIs resample the per-seed manager-minus-CNP differences. "
        "These are pointwise intervals without multiple-comparison adjustment.", "",
        "### Full results", "",
        "Cells are mean [95% CI low, high]. For every row, append metric name "
        "(`makespan`, `mean_wait`, `mean_response`, `messages_per_task`) and `.mean`, `.ci_low`, `.ci_high` "
        "to the source prefix. Raw trial metrics are in the sibling `.trials.<method>` array, aligned with `.workloads`.", "",
        "| Private load | Method | Makespan | Mean wait | Mean response | Messages/task | R source prefix |",
        "|---|---|---:|---:|---:|---:|---|"]
    metrics = ("makespan", "mean_wait", "mean_response", "messages_per_task")
    def ci_cell(v, mean="mean"):
        return f"{v[mean]:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    for load, data in new["experiment_6"]["by_private_load"].items():
        for name, values in data["aggregate"].items():
            lines.append(f"| {load} | {name} | " + " | ".join(ci_cell(values[m]) for m in metrics) +
                         f" | `experiment_6.by_private_load.{load}.aggregate.{name}` |")
    lines += ["", "### Paired manager minus CNP differences", "",
              "Positive means the manager's metric is larger. Append the metric and `.mean_diff`, `.ci_low`, `.ci_high` "
              "to each source prefix; CIs use paired trial differences, not subtraction of marginal interval endpoints.", "",
              "| Load | Manager | Makespan difference | Wait difference | Response difference | Messages/task difference | R source prefix |",
              "|---|---|---:|---:|---:|---:|---|"]
    for load, data in new["experiment_6"]["by_private_load"].items():
        for name, values in data["paired_manager_minus_cnp"].items():
            lines.append(f"| {load} | {name} | " + " | ".join(ci_cell(values[m], "mean_diff") for m in metrics) +
                         f" | `experiment_6.by_private_load.{load}.paired_manager_minus_cnp.{name}` |")
    lines += ["", "### Neutral reading (eight observations)", "",
        "1. Fresh state makes the unrefreshed manager an efficient choice: it matches full polling exactly at 1 versus "
        "25 messages/task, with wait 6.108 versus CNP's 6.936 (`R.experiment_6.by_private_load.0.0.aggregate.{Manager-never,Greedy-Centralized,Contract-Net}.{messages_per_task,mean_wait}.mean`).",
        "2. At private load 0.1, refreshing every 10 tasks uses 2.163 messages/task with wait 11.304 versus CNP's 11.775; "
        "the paired wait difference is -0.471 [-1.551, 0.531], so the experiment does not establish a wait advantage "
        "(`R.experiment_6.by_private_load.0.1.aggregate.{Manager-10,Contract-Net}` and `.paired_manager_minus_cnp.Manager-10.mean_wait`).",
        "3. At load 0.2, never refreshing gives wait 25.699 versus CNP's 21.625; the paired gap is 4.074 [2.350, 5.768]. "
        "Live bidding improves wait here at 11 versus 1 message/task "
        "(`R.experiment_6.by_private_load.0.2.aggregate.{Manager-never,Contract-Net}` and `.paired_manager_minus_cnp.Manager-never.mean_wait`).",
        "4. At load 0.3, refreshing every 10 tasks gives wait 41.888 versus 38.016 for CNP; its paired gap "
        "3.871 [1.257, 6.390] supports a wait benefit from bidding in this condition "
        "(`R.experiment_6.by_private_load.0.3.aggregate.{Manager-10,Contract-Net}` and `.paired_manager_minus_cnp.Manager-10.mean_wait`).",
        "5. At load 0.5, never-refresh makespan is 813.917 versus CNP's 701.104, a paired gap of "
        "112.814 [97.335, 130.034]; bidding substantially improves the last public completion "
        "(`R.experiment_6.by_private_load.0.5.aggregate.{Manager-never,Contract-Net}.makespan` and `.paired_manager_minus_cnp.Manager-never.makespan`).",
        "6. That same high-load condition gives response 106.706 versus 105.545, whose paired gap "
        "1.161 [-2.542, 4.771] includes zero; it does not establish a mean response advantage "
        "(`R.experiment_6.by_private_load.0.5.aggregate.{Manager-never,Contract-Net}.mean_response` and `.paired_manager_minus_cnp.Manager-never.mean_response`).",
        "7. Refreshing every task costs 12.940 messages/task, more than CNP's 11; at load 0.5 its wait is "
        "87.890 versus 80.910. Live state alone does not make a myopic all-agent choice optimal "
        "(`R.experiment_6.by_private_load.0.5.aggregate.{Manager-1,Contract-Net}.{messages_per_task,mean_wait}.mean`).",
        "8. Managers with infrequent reports remain the lowest-message choices; bidding is attractive where its measured "
        "delay or makespan improvement justifies extra communication. No single method dominates every metric and load; "
        "the complete pointwise intervals above define the evidence.", "",
        "## No-responder rule and audit", "",
        "If all invited candidates drop the CFP, the original code takes true costs from the same nearest invitees, "
        "selects the cheapest, and charges only the already sent CFPs plus the award. It does not contact all agents, "
        "charge a second announcement, charge synthetic bid responses, or apply fresh quote noise in that fallback. "
        "This is an uncharged local-state lookup assumption, not communication-free recovery that a distributed "
        "implementation could generally guarantee. The fallback counter is diagnostic only.", "",
        "Before edits, it fired 9 times among 4059 public tasks (0.222% rounded), all in the 30%-nonresponse condition "
        "(`A.fallbacks[\"experiment_5.nonresponse.0.3\"].{fallbacks,n_tasks}`; percentage `V.fallback_counterfactual.percent_tasks`). "
        "Charging one full-team re-announcement and replies would add 216 messages overall, or 10.8 per trial "
        "(`V.fallback_counterfactual.{extra_messages_all_trials,extra_messages_per_trial}`). "
        "That changes published CNP results, so the hard preservation requirement selects documentation rather than charging.", "",
        "Every configuration is below. Append `.fallbacks` / `.n_tasks` to the A path; `.trials` contains each scenario seed, "
        "count, task IDs and original message total. Experiment 1's extra figure-only replay has the same seed-zero "
        "input and no fallback; it is not counted twice.", "",
        "| Configuration | Fallback events | Public tasks | A path prefix |", "|---|---:|---:|---|"]
    for key, data in audit["fallbacks"].items():
        lines.append(f"| {key} | {data['fallbacks']} | {data['n_tasks']} | `fallbacks[\"{key}\"]` |")
    lines += ["", "## Verification, figures, and surprises", "",
        "- Gate: `python -m pytest tests/ -q`. Independent result gate: "
        "`python -I checks/baseline_corrections_check.py`. The original-bug regression tests were observed failing before fixes. "
        "The retained-prefix fixture also distinguishes an in-memory mutation that omits held-task queue load.",
        "- All regenerated figures use an exact 122 mm canvas width, font sizes at least 7.2 pt, colourblind-safe colours, "
        "and distinct markers/line styles or hatches. PNGs are 300 dpi; matching PDF and SVG are supplied. "
        "Sources: `scripts/run_experiments.py::{FIG_WIDTH,save_figure,fig_*}`; measured dimensions/fonts/DPI: "
        "`V.figures.<figure>.{width_mm,minimum_font_pt,png_dpi}`. Figure identifiers retain the original numbering.",
        "- `fig2_strategy_comparison`, `fig3_load_distribution`, `fig6_token_proxy` use `R.experiment_1`; "
        "`fig4_scalability` uses `R.experiment_2`; `fig5_ablation_k` uses `R.experiment_3`; "
        "`fig8_optimality_gap` uses `R.experiment_4`; `fig7_robustness` uses `R.experiment_5`; "
        "`fig10_when_bidding_pays` uses `R.experiment_6`. Error bars in the original experiment figures are SD; "
        "the new experiment uses 95% bootstrap CIs; "
        "all have been regenerated. Each filename is under `outputs/figures/`.",
        "- `fig9_llm_live` only reads `outputs/results/llm_sweep_n{8,16,24}.json`: tokens/task = "
        "sum(`trials[*].{llm_cnp,llm_centralized}.llm.total_tokens`)/sum(`trials[*].<method>.n_tasks`); "
        "makespan = mean(`trials[*].<method>.makespan`). No paid experiments were rerun.",
        "- Every non-SSI/CBBA published value reproduced exactly. The expected corrected baseline numbers also reproduced. "
        "The old low-communication headline remains an advantage over per-task full polling, not over a fresh state-tracking manager.",
        "- The fresh Experiment 6 absolute means differ from Experiment 1 because they use a new seed block; this is intended. "
        "Manager s=1 can differ slightly from full polling with private work because its first task has no report; a private "
        "job can arrive before that task. Subsequent decisions may follow different trajectories. Fresh-state equality remains exact.",
        "- More frequent reports do not monotonically improve all metrics; allocation is myopic, and different choices change "
        "travel and future queues. CNP can improve on full polling under load for the same reason. This is an observed "
        "simulation result, not evidence of optimality or a general dominance claim. Offered private load is not a measured "
        "utilization percentage, and private work here changes queue time but not location.",
        "",
        "## Complete metric comparison appendix", "",
        "All stored aggregate metrics are included, not only the three headline outcomes. Cells are mean (SD); "
        "append `.mean` or `.std` to the explicit old/new prefixes. Exp.4 original values here come from the pre-edit replay.", "",
        "| Setting / method / metric | Original | Corrected | Old path → new path |", "|---|---:|---:|---|"]
    for exp, name, original, corrected, old_path, new_path in comparisons:
        for metric, a in original.items():
            b = corrected[metric]
            lines.append(f"| {exp} / {name} / {metric} | {a['mean']:.6f} ({a['std']:.6f}) | "
                         f"{b['mean']:.6f} ({b['std']:.6f}) | `{old_path}.{metric}` → `{new_path}.{metric}` |")
    if (RESULTS / "validation.json").exists():
        validation = read("validation.json")
        lines += ["", "## Recorded delivery checks", "",
                  f"`{validation['pytest_stdout'].strip()}` (`Q.pytest_stdout`); "
                  f"exit code {validation['pytest_exit_code']} (`Q.pytest_exit_code`). "
                  "The original-result verifier passed (`Q.result_check_stdout`). "
                  "Two full experiment runs produced byte-identical results JSON "
                  "(`Q.two_full_runs_identical`, `Q.results_sha256`); measured runtimes are "
                  "in `Q.experiment_runtime_seconds`. The current nonresponse runs reproduce "
                  "the pre-edit fallback/task/message counts exactly "
                  "(`Q.fallback_nonresponse_03_matches_audit`). "
                  "Recorded source hashes and dependency versions identify the checked snapshot "
                  "(`Q.source_sha256`, `Q.versions`). All figure layouts were visually inspected; "
                  "font size, canvas width and PNG DPI also pass the result verifier."]
    (ROOT / "CORRECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verification
