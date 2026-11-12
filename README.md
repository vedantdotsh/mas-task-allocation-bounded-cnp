# Reproducible multi-agent task allocation

The default **v2** study compares seven information policies: round-robin, nearest-agent, full-information greedy polling, manager-mediated bounded CNP, SSI, CBBA, and cached central routing. It includes registry maintenance, measured JSON-RPC-style message sizes, stale state, private jobs, skill eligibility, and a small live LLM arithmetic-routing replication.

Read [generated v2 tables](docs/results_v2.md), [model notes and limitations](docs/model_notes.md), and the [tested running example](docs/running_example.md). All paper numbers should come from these saved results. The old paper's Table 1 did not match its recovered model; [legacy reproduction](docs/reproduction.md) documents that discrepancy. No parameters were tuned to fit the old paper.

## Reproduce

Python 3.13 on Windows was used. Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_all.py
```

`run_all.py` defaults to v2 and regenerates E1-E6 (30 trials at every point), CSV/JSON outputs, figures, token sensitivity, the running example and all result tables. It **only replays the live LLM cache**, never makes a paid call. An incomplete cache is an explicit error. Use `--workers 4` to set simulation parallelism. Nothing executes against the recovered OneDrive folder.

```powershell
python scripts/run_v2.py --experiments 1 2 3 4 5 6 --workers 4
python scripts/write_results_v2.py
python scripts/make_figures.py
```

The last two commands read saved simulation results. Individual `run_experiment1.py`, `run_scalability.py`, `run_k_ablation.py`, `run_optimality.py`, and `run_robustness.py` default to v2 and accept `--version legacy`.

For the untouched recovered behavior:

```powershell
python scripts/run_all.py --version legacy
```

`ScenarioConfig(version="legacy")` and the original `strategies.py`/`ExperimentConfig` APIs preserve original seeds, message/token conventions and metrics. All 30 archived trials are regression-tested. `outputs/recovered_original/` is immutable evidence, not regenerated output.

## Live replication: explicitly separate from reproduction

An already completed run replays without spending:

```powershell
python scripts/run_llm_experiment.py --cache-only
```

For an intentionally authorized new live run, the git-ignored root `.env` must contain `OPENROUTER_API_KEY`. Never put the key in a command, output or report. Follow this sequence:

```powershell
python scripts/run_llm_experiment.py --dry-run
python scripts/run_llm_experiment.py --smoke
python scripts/run_llm_experiment.py
```

The dry-run estimate must be below USD 2; inspect the 20-call smoke result before the full run. Persistent guards cap cumulative spend at USD 4 and attempts at 6,000. Ambiguous requests keep conservative reservations; missing `usage.cost` stops the run. Full-request caching makes repeated runs free. Live runs use an exclusive lock; after a crash, inspect the budget/unknown reservations and confirm no live process exists before removing a stale `outputs/v2/llm_cache/live.lock`. Do not reset the budget to bypass limits.

Worked-reasoning variant, zero-cost references and additional models (`gpt-4o-mini` worked, `google/gemini-3.1-flash-lite`, `meta-llama/llama-3.3-70b-instruct`) use `--model`; each has its own journal and a separate result file, and `run_all.py` replays all of them from cache. The sequence per model is the same:

```powershell
python scripts/run_llm_experiment.py --model llama-3.3-70b --dry-run
python scripts/run_llm_experiment.py --model llama-3.3-70b --smoke
python scripts/run_llm_experiment.py --model llama-3.3-70b            # add --with-main for the 8-agent design
```

The total across the project is capped at USD 8 (key usage is read from OpenRouter before and after each run). This is a small synthetic arithmetic-routing replication with `openai/gpt-4o-mini`, not a semantic LLM-agent benchmark. Formula answers are not supplied to the model. Invalid responses get one retry, then exclusion/unserved status, never an undisclosed formula fallback.

## Output map

- `outputs/v2/results/experiment_{1..6}_raw.csv`: every trial/configuration/condition, metrics, message-type counts, token/byte totals, managed loads and settings.
- `outputs/v2/results/experiment_N_summary.json`, `results.json`: sample SD, 95% mean CIs, paired bootstrap percentages and the pre-specified +/-5% margin decisions.
- `outputs/v2/audit/`: complete seed-0 E1 serialized messages and schedules at both loads.
- `outputs/v2/results/token_sensitivity.json`: queue-length x CFP-size sensitivity, also copied to `outputs/results/token_sensitivity.json` as requested.
- `outputs/v2/results/llm_*.json`, `llm_*_raw.csv`: plan, smoke, live trials, API tokens, latencies, validity and actual usage.cost.
- `outputs/v2/llm_cache/`: full-request response cache and cumulative spending journal; no credentials.
- `outputs/v2/figures/`: 12 cm wide, minimum 11 pt, colour-blind-safe PNGs (300 dpi) and vector PDFs. Error bars are 95% CIs of trial means. `manifest.json` lists all figures.
- `docs/results_v2.md`: generated tables, never hand-entered numbers.

The exact solver covers makespan only, with at most four agents and eleven tasks, arbitrary per-agent ordering, release times, no pre-travel, and no imposed batch wait. CNP uses k=2 for this comparison. See model notes for statistical definitions, message accounting and limits.

## API example

```python
from mas_sim.scenario import ScenarioConfig, generate_scenario
from mas_sim.simulation import simulate
scenario = generate_scenario(ScenarioConfig(), seed=0)
result = simulate(scenario, "cnp", k=5, shortlist="capability", stale_every=1)
print(result.metrics)
print(result.ledger.token_summary())
```

Use `shortlist="nearest"` or `"random"`, `stale_every=None` for never, `theta=0.6` for the CNP hard-eligibility variant, and `private_load=0.5` for private/managed arrival-rate ratio. Quality is on managed work; failures stay visible in unserved rates.
