"""Thirty paired trials at each pre-specified point; all tables derive from saved rows."""
import csv
import gzip
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .optimal import exact_optimum
from .scenario import ScenarioConfig, generate_scenario
from .simulation import simulate
from .stats import _bootstrap_ci
from .v2 import NAMES
from .wire import TOKENIZER

METRICS = ('makespan', 'mean_response', 'mean_wait', 'p95_wait', 'fairness', 'messages_per_task',
           'tokens_per_task', 'unserved_rate', 'declines_per_task', 'reannouncements_per_task',
           'drain_time', 'tasks', 'served', 'private_jobs', 'optimal_ratio')


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite observations')
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(student_t.ppf(.975, len(values)-1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {'mean': mean, 'std': sd, 'ci95': [mean-half, mean+half], 'n': len(values)}


def points(number):
    all_names = list(NAMES)
    if number == 1:
        return [(str(load), {'expected_tasks': load}, [(n, {}) for n in all_names]) for load in (200, 300)]
    if number == 2:
        return [(f'{mode}/n{n}', {'n_agents': n, 'expected_tasks': 200 if mode == 'fixed' else 200*n/12},
                 [(name, {}) for name in all_names]) for mode in ('fixed', 'proportional') for n in (4, 8, 12, 16, 24, 32)]
    if number == 3:
        result = [(f'{rule}/k{k}', {}, [('cnp', {'k': k, 'shortlist': rule})])
                  for rule in ('nearest', 'capability', 'random') for k in (1, 2, 3, 5, 8, 12)]
        result += [(f'stale/{s}', {}, [(name, {'stale_every': s}) for name in ('cnp', 'cached_central')])
                   for s in (1, 2, 5, 10, None)]
        result += [(f'eligible/{rule}/k{k}', {}, [('cnp', {'k': k, 'shortlist': rule, 'theta': .6})])
                   for rule in ('nearest', 'capability', 'random') for k in (2, 3, 5)]
        return result
    if number == 4:
        return [('small', {'n_agents': 4}, [(name, {'k': 2}) for name in all_names])]
    if number == 5:
        return [(f'sigma{sigma}/p{p}', {}, [('cnp', {'noise': sigma, 'nonresponse': p})])
                for sigma in (0, .1, .2, .3, .5) for p in (0, .1, .3)]
    if number == 6:
        return [(str(rho), {}, [('cached_central', {'private_load': rho}),
                               ('cnp', {'private_load': rho}),
                               ('cnp_random', {'private_load': rho, 'shortlist': 'random'}),
                               ('greedy', {'private_load': rho})]) for rho in (0, .25, .5, 1)]
    raise ValueError(number)


def paired(rows, reference):
    left = {r['seed']: r for r in rows if r['mechanism'] == 'cnp'}
    right = {r['seed']: r for r in rows if r['mechanism'] == reference}
    if left.keys() != right.keys() or not left:
        return None
    output = {}
    for metric in ('makespan', 'mean_response'):
        relative = [100*(left[s][metric]-right[s][metric])/right[s][metric] for s in sorted(left)]
        result = _bootstrap_ci(relative)
        result['margin_percent'] = 5.0
        result['within_margin'] = result['ci95'][0] >= -5 and result['ci95'][1] <= 5
        result['absolute_difference'] = _bootstrap_ci([left[s][metric]-right[s][metric] for s in sorted(left)])
        result['definition'] = 'mean of paired trial relative differences: 100*(CNP-reference)/reference'
        output[metric] = result
    return output


def _trial(job):
    number, point, config_kwargs, mechanisms, seed, output = job
    scenario = generate_scenario(ScenarioConfig(**config_kwargs), seed)
    optimum = None
    if number == 4:
        scenario.tasks = scenario.tasks[:11]
        optimum = exact_optimum(scenario)
    rows = []
    for name, settings in mechanisms:
        result = simulate(scenario, 'cnp' if name == 'cnp_random' else name, **settings)
        row = {'experiment': number, 'point': point, 'seed': seed, 'mechanism': name,
               **asdict(scenario.config), 'k': 5, 'shortlist': 'capability', 'stale_every': 1,
               'theta': 0, 'private_load': 0, 'noise': 0, 'nonresponse': 0, **settings,
               **result.metrics, 'message_types': result.ledger.counts_by_type(),
               'token_types': result.ledger.token_summary(),
               'loads': [sum(a.travel+a.execution for a in result.assignments if a.agent_id == agent.id) for agent in result.agents]}
        if optimum:
            row.update(optimal_makespan=optimum.makespan,
                       optimal_ratio=result.metrics['makespan']/optimum.makespan,
                       optimum=asdict(optimum))
            assert row['optimal_ratio'] >= 1-1e-10
        rows.append(row)
        if number == 1 and seed == 0:
            audit = Path(output) / 'audit'
            audit.mkdir(parents=True, exist_ok=True)
            with gzip.open(audit / f'load{point}_{name}.jsonl.gz', 'wt', encoding='utf-8') as handle:
                for message in result.ledger.messages:
                    handle.write(json.dumps(asdict(message))+'\n')
            (audit / f'load{point}_{name}_schedule.json').write_text(json.dumps([asdict(a) for a in result.assignments]), encoding='utf-8')
        # Break ledger/context cycle promptly; avoid retaining large consensus ledgers.
        result.ledger.context = None
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run_experiments(output='outputs/v2', trials=30, workers=4, numbers=range(1, 7)):
    output = Path(output)
    directory = output / 'results'
    directory.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number in numbers:
            rows, groups, comparisons = [], {}, {}
            for point, config, mechanisms in points(number):
                jobs = [(number, point, config, mechanisms, seed, str(output)) for seed in range(trials)]
                subset = [r for result in pool.map(_trial, jobs) for r in result]
                rows.extend(subset)
                groups[point] = {name: {m: summarize([r[m] for r in subset if r['mechanism'] == name])
                                        for m in METRICS if m in subset[0]}
                                 for name, _ in mechanisms}
                comparisons[point] = {ref: value for ref in ('greedy', 'cached_central') if (value := paired(subset, ref))}
                print(f'v2 E{number} {point}: {trials} trials complete', flush=True)
            types = {}
            for row in rows:
                target = types.setdefault(row['mechanism'], {})
                for kind, values in row['token_types'].items():
                    totals = target.setdefault(kind, {'messages': 0, 'tokens': 0, 'bytes': 0})
                    for key in totals:
                        totals[key] += values[key]
            for by_type in types.values():
                for v in by_type.values():
                    v['mean_tokens'] = v['tokens']/v['messages']
            summary = {'metadata': {'version': 'v2', 'trials': trials, 'std_ddof': 1,
                                     'ci': 'Student t 95% CI of trial mean; paired percentile bootstrap for differences',
                                     'seed_scheme': 'SeedSequence(seed).spawn(4): agents, tasks, protocol, private jobs; seeds 0..29',
                                     'tokenizer': TOKENIZER, 'python': platform.python_version(), 'numpy': np.__version__,
                                     'equivalence_margin_percent': 5, 'quality_population': 'served managed tasks; unserved reported separately'},
                       'groups': groups, 'paired_relative_percent': comparisons, 'tokens_by_type': types}
            if number == 4:
                summary['optima'] = [r['optimum'] | {'seed': r['seed']} for r in rows if r['mechanism'] == 'cnp']
                summary['optimal_objective'] = 'makespan only; response-time optimum is not solved'
            write_csv(directory / f'experiment_{number}_raw.csv', rows)
            (directory / f'experiment_{number}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    combined = {f'experiment_{n}': json.loads(p.read_text()) for n in range(1, 7)
                if (p := directory / f'experiment_{n}_summary.json').exists()}
    (directory / 'results.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    return combined
