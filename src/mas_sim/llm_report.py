"""Saved-output-only tables and figure data for the live LLM variants, reference conditions and extra models."""
import json
from pathlib import Path

import numpy as np

from .experiments_v2 import summarize
from .llm import MODEL_SPECS
from .stats import _bootstrap_ci

GROUPS = ('sweep/n8', 'sweep/n16', 'sweep/n24', 'main/n8')
VARIANTS = {'oneshot': ('llm_cnp', 'llm_centralized'), 'worked': ('llm_worked_cnp', 'llm_worked_centralized')}
ONESHOT_CAP = 128
METRICS = ('total_tokens_per_task', 'prompt_tokens_per_task', 'completion_tokens_per_task', 'calls_per_task', 'latency_per_task',
           'invalid_response_rate', 'truncated_response_rate', 'routing_agreement', 'makespan', 'mean_response')


def _trials(summary, calls=None):
    rows = [dict(r) for r in summary['trials']]
    truncated = {}
    for c in calls or []:
        key = (c['study'], c['n_agents'], c['seed'], c['condition'])
        truncated.setdefault(key, []).append((c['completion_tokens'] or 0) >= ONESHOT_CAP)
    for r in rows:
        r.setdefault('total_tokens_per_task', r['prompt_tokens_per_task']+r['completion_tokens_per_task'])
        if 'truncated_response_rate' not in r:
            # The one-shot run did not store finish_reason; a completion that used the whole 128-token cap was cut off.
            flags = truncated.get((r['study'], r['n_agents'], r['seed'], r['condition']))
            r['truncated_response_rate'] = sum(flags)/len(flags) if flags else 0.0
    return rows


def load_models(output):
    """{model key: {'id', 'rows', 'summary'}} in presentation order; missing result files are skipped."""
    output = Path(output)/'results'
    models = {}
    legacy = output/'llm_results.json'
    if legacy.exists():
        summary = json.loads(legacy.read_text())
        calls = json.loads((output/'llm_results_calls.json').read_text())
        rows = _trials(summary, calls)
        models['gpt-4o-mini'] = {'id': summary['model'], 'rows': rows, 'summary': summary, 'variants': ['oneshot']}
    for name, spec in MODEL_SPECS.items():
        path = output/f'{spec["prefix"]}_results.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        if summary['status'] != 'complete':
            continue
        entry = models.setdefault(name, {'id': spec['id'], 'rows': [], 'summary': {}, 'variants': []})
        entry['rows'] += [r for r in _trials(summary) if name != 'gpt-4o-mini' or r['condition'] not in ('formula_cnp', 'formula_centralized')]
        entry['summary_extra' if name == 'gpt-4o-mini' else 'summary'] = summary
        entry['variants'] += [v for v in spec['variants'] if v not in entry['variants']]
    return models


def groups_of(rows):
    present = {f'{r["study"]}/n{r["n_agents"]}' for r in rows}
    return [g for g in GROUPS if g in present]


def select(rows, group, condition):
    study, n = group.split('/n')
    return sorted((r for r in rows if r['study'] == study and r['n_agents'] == int(n) and r['condition'] == condition),
                  key=lambda r: r['seed'])


def mean_ci(rows, metric):
    return summarize([r[metric] for r in rows]) if rows else None


def bid_error(rows):
    """Pooled bid-error distribution over valid bids; relative to the formula bid on the same state."""
    signed = np.array([e for r in rows for e in r.get('bid_signed_relative_errors', [])], dtype=float)
    if not signed.size:
        return None
    absolute = np.abs(signed)
    return {'valid_bids': int(signed.size), 'median_absolute': float(np.median(absolute)),
            'within_10': float((absolute <= .1).mean()), 'within_25': float((absolute <= .25).mean()),
            'zero_share': float((signed == -1).mean()), 'median_signed': float(np.median(signed))}


def paired_tokens(rows, group, variant):
    bid, central = (select(rows, group, c) for c in VARIANTS[variant])
    if not bid or len(bid) != len(central) or [r['seed'] for r in bid] != [r['seed'] for r in central]:
        return None
    diff = [a['total_tokens_per_task']-b['total_tokens_per_task'] for a, b in zip(bid, central)]
    result = _bootstrap_ci(diff)
    result['mean_bidding'] = float(np.mean([r['total_tokens_per_task'] for r in bid]))
    result['mean_centralized'] = float(np.mean([r['total_tokens_per_task'] for r in central]))
    return result


def figure_series(models, name):
    """{condition: {metric: [(n, mean, ci_half)]}} over the sweep for one model, for the live-LLM figure."""
    rows = models[name]['rows']
    out = {}
    for condition in ('llm_cnp', 'llm_centralized', 'llm_worked_cnp', 'llm_worked_centralized', 'formula_cnp', 'formula_centralized'):
        out[condition] = {}
        for metric in ('total_tokens_per_task', 'mean_response'):
            points = []
            for n in (8, 16, 24):
                s = mean_ci(select(rows, f'sweep/n{n}', condition), metric)
                if s is not None:
                    points.append((n, s['mean'], s['ci95'][1]-s['mean']))
            out[condition][metric] = points
    return out


def _ci(v, digits=3):
    return f'{v["mean"]:.{digits}f} [{v["ci95"][0]:.{digits}f}, {v["ci95"][1]:.{digits}f}]'


def markdown(output):
    models = load_models(output)
    if not models:
        return []
    lines = ['', '## Live LLM: worked variant, reference conditions and additional models', '',
             'Generated from the saved per-model result files by `src/mas_sim/llm_report.py`. All models use the same scenarios, seeds, '
             'k=3 capability shortlist and scheduler as the one-shot gpt-4o-mini run, temperature 0, and (for the new runs) the same prompt facts. '
             '**One-shot** requires a bare JSON answer (128-token cap). **Worked** asks for brief step-by-step working and ends with the same JSON '
             'object on the final line; the last JSON object is parsed with the same strict validation, and a response cut off by `max_tokens` '
             '(`finish_reason` = length) counts as invalid. `shortlist_random` and `shortlist_top1` are zero-cost references: a seeded uniform '
             'pick from the same three-agent shortlist, and its first entry, with no bids. The formula conditions compute the bid exactly. '
             'Entries are mean [95% Student-t CI over trials]; there are five trials per group, so intervals are wide. Token counts are '
             'provider-billed `usage` tokens. Truncated rate for the original one-shot gpt-4o-mini run is inferred from completions that used the whole 128-token cap '
             '(finish_reason was not stored then). Latency is summed API seconds per task, original latency when replayed from cache. '
             'Trials run in parallel threads in the new runs, so latency is not inflated by queueing between trials.', '']
    info = []
    for name, m in models.items():
        s = m.get('summary') or m.get('summary_extra')
        extra = m.get('summary_extra')
        info.append(f'- `{m["id"]}` ({name}): variants {", ".join(m["variants"])}; '
                    f'prices USD {MODEL_SPECS[name]["prices"][0]:g}/{MODEL_SPECS[name]["prices"][1]:g} per million input/output tokens '
                    f'(OpenRouter catalogue, 2026-11-12). {MODEL_SPECS[name]["why"]}')
    lines += ['Models:', ''] + info + ['']
    spend = []
    for name, m in models.items():
        for s in (m.get('summary'), m.get('summary_extra')):
            if s and name in MODEL_SPECS and s.get('budget') and 'truncated_calls' in s:
                spend.append(f'`{name}`: journal USD {s["budget"]["actual_usd"]:.4f} over {s["budget"]["calls"]} attempts including the smoke test; '
                             f'{s["truncated_calls"]} truncated and {s["invalid_or_failed_calls"]} invalid/failed of {s["logical_call_attempts"]} replayed attempts; '
                             f'billed reasoning tokens {s["reasoning_tokens"]}.')
    log_path = Path(output)/'results/llm_key_usage.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    if spend:
        lines += ['Spend of this addition (per-model journals; the original one-shot run is separate, USD 0.1022):', ''] + [f'- {x}' for x in spend] + ['']
    if log:
        lines += ['Cumulative usage of the OpenRouter key (`/api/v1/key`) before and after each live invocation. Small gaps between consecutive rows are '
                  'catalogue/parameter probe calls (under USD 0.02 in all):', '',
                  '| Model | Invocation | Key usage before (USD) | After (USD) | New API attempts |', '|---|---|---:|---:|---:|']
        lines += [f'| {e["model"]} | {e["invocation"]} | {e["key_usage_before_usd"]:.4f} | {e["key_usage_after_usd"]:.4f} | {e["new_calls"]} |' for e in log]
        lines += ['', f'Key usage after the last invocation: USD {log[-1]["key_usage_after_usd"]:.4f}, against the USD 8.00 project limit.', '']
    for name, m in models.items():
        rows = m['rows']
        lines += [f'### {name}: {m["id"]}', '']
        conditions = [c for c in ('formula_cnp', 'formula_centralized', 'shortlist_random', 'shortlist_top1', 'llm_cnp', 'llm_centralized',
                                  'llm_worked_cnp', 'llm_worked_centralized') if any(r['condition'] == c for r in rows)]
        for group in groups_of(rows):
            lines += [f'#### {name} {group}', '']
            for metrics in (('total_tokens_per_task', 'prompt_tokens_per_task', 'completion_tokens_per_task', 'calls_per_task', 'latency_per_task'),
                            ('invalid_response_rate', 'truncated_response_rate', 'routing_agreement', 'makespan', 'mean_response')):
                lines += ['| Condition | '+' | '.join(metrics)+' |', '|---|'+'---:|'*len(metrics)]
                for c in conditions:
                    sel = select(rows, group, c)
                    if sel:
                        lines.append(f'| {c} | '+' | '.join(_ci(mean_ci(sel, k)) for k in metrics)+' |')
                lines.append('')
            lines += ['Bid error against the formula bid on the same state (valid bids pooled over trials):', '',
                      '| Condition | Valid bids | Median abs. rel. error | Within 10% | Within 25% | Exact zeros | Median signed error |', '|---|---:|---:|---:|---:|---:|---:|']
            for c in ('llm_cnp', 'llm_worked_cnp'):
                d = bid_error(select(rows, group, c))
                if d:
                    lines.append(f'| {c} | {d["valid_bids"]} | {d["median_absolute"]:.3f} | {d["within_10"]:.3f} | {d["within_25"]:.3f} | '
                                 f'{d["zero_share"]:.3f} | {d["median_signed"]:+.3f} |')
            lines.append('')
        lines += ['Paired comparison of total tokens per task, bidding minus centralized, per trial (10,000 paired bootstrap resamples of the trial '
                  'differences, seed 12345):', '',
                  '| Variant | Group | Bidding tokens/task | Centralized tokens/task | Difference [95% bootstrap CI] | Bidding / centralized |', '|---|---|---:|---:|---|---:|']
        for variant in VARIANTS:
            for group in groups_of(rows):
                p = paired_tokens(rows, group, variant)
                if p:
                    lines.append(f'| {variant} | {group} | {p["mean_bidding"]:.1f} | {p["mean_centralized"]:.1f} | '
                                 f'{p["mean_difference"]:+.1f} [{p["ci95"][0]:+.1f}, {p["ci95"][1]:+.1f}] | {p["mean_bidding"]/p["mean_centralized"]:.2f} |')
        lines.append('')
    lines += cross_model(models)
    return lines


def cross_model(models):
    lines = ['### Cross-model summary (team-size sweep, expected 20 tasks, 5 trials per size)', '',
             'Total tokens per task, bidding versus centralized (mean over trials):', '',
             '| Model | Variant | n | Bidding | Centralized | Bidding / centralized | Paired difference [95% bootstrap CI] |', '|---|---|---:|---:|---:|---:|---|']
    for name, m in models.items():
        for variant in m['variants']:
            for n in (8, 16, 24):
                p = paired_tokens(m['rows'], f'sweep/n{n}', variant)
                if p:
                    lines.append(f'| {name} | {variant} | {n} | {p["mean_bidding"]:.0f} | {p["mean_centralized"]:.0f} | '
                                 f'{p["mean_bidding"]/p["mean_centralized"]:.2f} | {p["mean_difference"]:+.0f} [{p["ci95"][0]:+.0f}, {p["ci95"][1]:+.0f}] |')
    lines += ['', 'Bid accuracy, routing agreement with the formula choice, and mean response time (sweep pooled over n=8, 16, 24 for accuracy; per n for the rest):', '',
              '| Model | Variant | Median abs. rel. error | Within 10% | Zero bids | Agreement bid / central (n=8, 16, 24) | Mean response bid / central / formula (n=8, 16, 24) |',
              '|---|---|---:|---:|---:|---|---|']
    for name, m in models.items():
        rows = m['rows']
        for variant in m['variants']:
            bid_c, cen_c = VARIANTS[variant]
            sweep = [r for r in rows if r['study'] == 'sweep' and r['condition'] == bid_c]
            d = bid_error(sweep)
            agree = ' ; '.join(f'{mean_ci(select(rows, f"sweep/n{n}", bid_c), "routing_agreement")["mean"]:.2f} / '
                               f'{mean_ci(select(rows, f"sweep/n{n}", cen_c), "routing_agreement")["mean"]:.2f}' for n in (8, 16, 24))
            resp = ' ; '.join(f'{mean_ci(select(rows, f"sweep/n{n}", bid_c), "mean_response")["mean"]:.1f} / '
                              f'{mean_ci(select(rows, f"sweep/n{n}", cen_c), "mean_response")["mean"]:.1f} / '
                              f'{mean_ci(select(rows, f"sweep/n{n}", "formula_cnp"), "mean_response")["mean"]:.1f}' for n in (8, 16, 24))
            lines.append(f'| {name} | {variant} | {d["median_absolute"]:.3f} | {d["within_10"]:.3f} | {d["zero_share"]:.3f} | {agree} | {resp} |')
    lines.append('')
    return lines
