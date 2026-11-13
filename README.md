# Original MEDES 2026 task-allocation study

Code and results for "AI Agent Collaboration in Multi-Agent Systems for Complex
Problem Solving", accepted at MEDES 2026 (paper #33). This package contains the
corrected original simulation, its six experiments, and stored LLM results.

## Install and reproduce

Use Python 3.10 or newer and run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_experiments.py
python -I checks/baseline_corrections_check.py
```

The experiments typically take about 35 seconds; allow a minute or more depending
on hardware. Tests typically take under a second excluding Python startup.
The main experiment command makes no API calls and needs no credentials.

## Experiments

1. Allocation strategy comparison, including load distribution and token proxies.
2. Scalability across team sizes.
3. Invitation-bound k ablation.
4. Optimality gap against an exact small-instance solution.
5. Robustness to bid noise and nonresponse, including re-bidding.
6. When bidding pays: a state-tracking manager with private background work and
   periodic status reports, compared with live bidding and full polling.

Regenerated results land in `outputs/results/results.json`; figures land in
`outputs/figures/` as PNG, PDF and SVG. The command also regenerates
`CORRECTION_REPORT.md` and `outputs/results/correction_verification.json`.
The correction report documents accounting assumptions, limitations, confidence
intervals and exact JSON paths. `results_published_original.json` and
`published_audit.json` retain the original comparison evidence. `validation.json`
records historical validation; its source hashes describe that earlier snapshot.
`checks/measure_published.py` is a retired audit that refuses to overwrite evidence.

Stored `outputs/results/llm*.json` files support the LLM figure without new calls.
Running `python scripts/run_llm_experiment.py` separately requires your own
`OPENROUTER_API_KEY` in the environment and costs money through OpenRouter.
See its `--help` for model, trial and call-limit options. It does not load a key file.

## Changes since the first release

Corrected SSI/CBBA batch timing and CBBA bundle accounting; added Experiment 6
(state-tracking manager). The earlier rebuilt study remains available in git history.
