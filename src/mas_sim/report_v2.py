"""Saved-output-only tables/figures and transparent payload sensitivity."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .experiments_v2 import METRICS
from .llm_report import figure_series, load_models, markdown as llm_markdown, select
from .plotting import COLORS, MARKERS, HATCHES, WIDTH
from .v2 import NAMES
from .wire import count_tokens, serialize

COLORS = COLORS + ['#56B4E9']
MARKERS = MARKERS + ['X']
HATCHES = HATCHES + ['--']


LLM_STYLES = (('llm_cnp', 'One-shot bidding', '#D55E00', 'o', '-', True, -.75),
              ('llm_worked_cnp', 'Worked bidding', '#D55E00', 's', '--', False, -.25),
              ('llm_centralized', 'One-shot centralized', '#0072B2', 'D', '-', True, .25),
              ('llm_worked_centralized', 'Worked centralized', '#0072B2', '^', '--', False, .75))


def llm_panel(axes, name, series, one_label=False):
    """Total tokens / task (log scale) and mean response against team size. Points are dodged horizontally so
    that error bars and lines never sit on top of each other; the formula reference stays at the true team size."""
    for condition, label, color, marker, style, filled, dodge in LLM_STYLES:
        for ax, metric in zip(axes, ('total_tokens_per_task', 'mean_response')):
            points = series[condition][metric]
            if points:
                ax.errorbar([n+dodge for n, _, _ in points], [m for _, m, _ in points], yerr=[h for _, _, h in points], label=label,
                            color=color, marker=marker, linestyle=style, markerfacecolor=color if filled else 'white',
                            markersize=5.5, capsize=2, linewidth=1.2, zorder=3)
    formula = series['formula_cnp']['mean_response']
    same = formula == series['formula_centralized']['mean_response']
    axes[1].errorbar([n for n, _, _ in formula], [m for _, m, _ in formula], yerr=[h for _, _, h in formula], color='#555555',
                     marker='x', linestyle=':', markersize=9, markeredgewidth=1.6, capsize=2, linewidth=1.2, label='Formula (CNP = greedy)' if same else 'Formula CNP', zorder=4)
    axes[0].set_yscale('log')
    axes[0].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.set_xticks([8, 16, 24])
    if not one_label:
        axes[0].set_ylabel('Total tokens / task')
        axes[1].set_ylabel('Mean response')
    # The formula line lives in the second axes; give the first axes a proxy so one legend serves the column.
    axes[0].plot([], [], color='#555555', marker='x', linestyle=':', label='Formula (CNP = greedy)' if same else 'Formula CNP')


def sensitivity(output='outputs/v2'):
    output = Path(output)
    def read(name):
        with gzip.open(output/'audit'/f'load200_{name}.jsonl.gz', 'rt', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    cnp, greedy, cached = [read(name) for name in ('cnp', 'greedy', 'cached_central')]
    cfp = sum(m['tokens'] for m in cnp if m['type'] == 'cfp')
    other = sum(m['tokens'] for m in cnp if m['type'] != 'cfp')
    gc_other = sum(m['tokens'] for m in greedy if m['type'] != 'centralized_full_response')
    responses = [json.loads(m['serialized']) for m in greedy if m['type'] == 'centralized_full_response']
    template = next(r['result']['queue'][0] for r in responses if r['result']['queue'])
    total_tasks = sum(m['type'] == 'award' for m in cnp)
    cached_tokens = sum(m['tokens'] for m in cached)
    rows = []
    for length in (0, 1, 2, 5, 10, 20, 50, 100):
        tokens = gc_other
        for response in responses:
            original = response['result']['queue']
            queue = [dict((original or [template])[i % max(1, len(original))], task_id=i) for i in range(length)]
            envelope = {**response, 'result': {**response['result'], 'queue': queue}}
            tokens += count_tokens(serialize(envelope))
        for multiplier in (.25, .5, 1, 2, 4, 8):
            bounded = other + multiplier*cfp
            rows.append({'queue_length': length, 'cfp_multiplier': multiplier,
                         'cnp_tokens_per_task': bounded/total_tasks, 'greedy_tokens_per_task': tokens/total_tasks,
                         'cached_tokens_per_task': cached_tokens/total_tasks,
                         'cnp_saving_percent': 100*(1-bounded/tokens),
                         'cached_saving_percent': 100*(1-cached_tokens/tokens)})
    result = {'scope': 'E1 load200 seed0 control-plane messages, including registrations. Counterfactual payload sizes, fixed allocation.',
              'queue_policy': 'Exactly q active queue items in each polling response; truncate/cycle observed items (or first observed item if empty), unique synthetic task ids. No full history.',
              'cfp_policy': 'Multiply measured serialized CFP token totals only, from 0.25x to 8x; this is a sensitivity, not a claim of a smaller valid envelope.',
              'actual_seed0': {'cnp_tokens_per_task': (other+cfp)/total_tasks,
                               'greedy_tokens_per_task': sum(m['tokens'] for m in greedy)/total_tasks,
                               'cached_tokens_per_task': cached_tokens/total_tasks}, 'rows': rows}
    for path in (output/'results/token_sensitivity.json', output.parent/'results/token_sensitivity.json'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def generate_figures(output='outputs/v2'):
    output = Path(output)
    data = json.loads((output/'results/results.json').read_text())
    directory = output/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
                         'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
    paths = []
    def figure(rows=1):
        fig, axes = plt.subplots(rows, 1, figsize=(WIDTH, 2.8*rows), layout='constrained', squeeze=False)
        return fig, list(axes[:, 0])
    def save(fig, name):
        for ext in ('png', 'pdf'):
            path = directory/f'{name}.{ext}'
            fig.savefig(path, dpi=300)
            paths.append(str(path))
        plt.close(fig)
    def bars(ax, group, metric, ylabel):
        keys = list(group)
        vals = [group[k][metric] for k in keys]
        bar = ax.bar(range(len(keys)), [v['mean'] for v in vals],
                     yerr=[v['ci95'][1]-v['mean'] for v in vals], capsize=2, color=COLORS[:len(keys)], edgecolor='black')
        for b, hatch in zip(bar, HATCHES):
            b.set_hatch(hatch)
        ax.set_xticks(range(len(keys)), [dict(round_robin='RR', nearest='NA', greedy='FG', cnp='CNP', ssi='SSI', cbba='CBBA', cached_central='CC')[k] for k in keys])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    def line(ax, xs, values, label, i):
        ax.errorbar(xs, [v['mean'] for v in values], yerr=[v['ci95'][1]-v['mean'] for v in values],
                    label=label, color=COLORS[i], marker=MARKERS[i], capsize=2, linewidth=1)
    for load, group in data['experiment_1']['groups'].items():
        fig, axes = figure(4)
        for ax, metric, label in zip(axes, ('makespan','mean_response','messages_per_task','tokens_per_task'),
                                     ('Makespan','Mean response','Messages / task','Tokens / task')):
            bars(ax, group, metric, label)
        axes[0].set_title(f'Expected {load} tasks; 95% mean CIs')
        save(fig, f'e1_load{load}')
    for mode in ('fixed', 'proportional'):
        fig, axes = figure(3)
        sizes = [4,8,12,16,24,32]
        for i, name in enumerate(NAMES):
            for ax, metric in zip(axes, ('mean_response','messages_per_task','tokens_per_task')):
                line(ax, sizes, [data['experiment_2']['groups'][f'{mode}/n{n}'][name][metric] for n in sizes],
                     {'round_robin':'RR','nearest':'NA','greedy':'FG','cnp':'CNP','ssi':'SSI','cbba':'CBBA','cached_central':'CC'}[name], i)
        for ax, label in zip(axes, ('Mean response', 'Messages / task', 'Tokens / task')):
            ax.set_ylabel(label)
            ax.set_xlabel('Agents')
            ax.set_xticks(sizes)
            ax.set_yscale('log')
        axes[0].legend(ncol=3, loc='best')
        axes[0].set_title(f'{mode.capitalize()} offered load')
        save(fig, 'e2_'+mode)
    fig, axes = figure(3)
    ks = [1,2,3,5,8,12]
    for i, rule in enumerate(('nearest','capability','random')):
        for ax, metric in zip(axes, ('mean_response','messages_per_task','tokens_per_task')):
            line(ax, ks, [data['experiment_3']['groups'][f'{rule}/k{k}']['cnp'][metric] for k in ks], rule,
                 {'nearest':1,'capability':3,'random':0}[rule])
    for ax, label in zip(axes, ('Mean response', 'Messages / task','Tokens / task')):
        ax.set_ylabel(label); ax.set_xlabel('Invitation bound k'); ax.set_xticks(ks)
    axes[0].legend()
    save(fig,'e3_shortlist')
    fig, axes = figure(2)
    for i, name in enumerate(('cnp','cached_central')):
        for ax, metric in zip(axes, ('mean_response','messages_per_task')):
            line(ax, range(5), [data['experiment_3']['groups'][f'stale/{s}'][name][metric] for s in (1,2,5,10,None)],
                 'CNP' if name == 'cnp' else 'Cached central', 3 if name == 'cnp' else 6)
    for ax, label in zip(axes, ('Mean response','Messages / task')):
        ax.set_xticks(range(5), ['1','2','5','10','never']); ax.set_xlabel('Refresh every s awards'); ax.set_ylabel(label)
    axes[0].legend(); save(fig,'e3_staleness')
    fig, axes = figure(3)
    for i, rule in enumerate(('nearest','capability','random')):
        for ax, metric in zip(axes, ('declines_per_task','reannouncements_per_task','unserved_rate')):
            line(ax, [2,3,5], [data['experiment_3']['groups'][f'eligible/{rule}/k{k}']['cnp'][metric] for k in (2,3,5)], rule,
                 {'nearest':1,'capability':3,'random':0}[rule])
    for ax, label in zip(axes, ('Declines / task','Reannouncements / task','Unserved fraction')):
        ax.set_ylabel(label); ax.set_xlabel('Invitation bound k')
    axes[0].legend(); save(fig,'e3_eligibility')
    fig, axes = figure()
    bars(axes[0], data['experiment_4']['groups']['small'], 'optimal_ratio', 'Makespan / exact optimum')
    axes[0].axhline(1, color='black', linestyle='--'); save(fig,'e4_optimum')
    fig, axes = figure(2)
    for i, p in enumerate((0,.1,.3)):
        for ax, metric in zip(axes, ('mean_response','messages_per_task')):
            line(ax, [0,.1,.2,.3,.5], [data['experiment_5']['groups'][f'sigma{s}/p{p}']['cnp'][metric] for s in (0,.1,.2,.3,.5)], f'p={p}', i)
    for ax, label in zip(axes, ('Mean response','Messages / task')):
        ax.set_ylabel(label); ax.set_xlabel('Bid noise sigma')
    axes[0].legend(); save(fig,'e5_robustness')
    fig, axes = figure(3)
    for i, name in enumerate(('cached_central','cnp','cnp_random','greedy')):
        for ax, metric in zip(axes, ('mean_response','messages_per_task','tokens_per_task')):
            line(ax, [0,.25,.5,1], [data['experiment_6']['groups'][str(rho)][name][metric] for rho in (0,.25,.5,1)],
                 {'cached_central':'Cached','cnp':'CNP cap.','cnp_random':'CNP random','greedy':'Full greedy'}[name],
                 {'cached_central':6,'cnp':3,'cnp_random':0,'greedy':2}[name])
    for ax, label in zip(axes, ('Mean response','Messages / task','Tokens / task')):
        ax.set_ylabel(label); ax.set_xlabel('Private / managed arrival rate')
    axes[0].legend(ncol=2); save(fig,'e6_private_load')
    sens = json.loads((output/'results/token_sensitivity.json').read_text())
    fig, axes = figure()
    for i, mult in enumerate((.25,.5,1,2,4,8)):
        rows = [r for r in sens['rows'] if r['cfp_multiplier'] == mult]
        axes[0].plot([r['queue_length']+1 for r in rows], [r['cnp_saving_percent'] for r in rows],
                     marker=MARKERS[i], color=COLORS[i], label=f'{mult:g}x CFP')
    axes[0].axhline(0,color='black',linestyle='--'); axes[0].set_xscale('log')
    axes[0].set_ylabel('CNP token saving (%)'); axes[0].set_xlabel('Central queue items + 1')
    axes[0].legend(ncol=2); save(fig,'token_sensitivity')
    models = load_models(output)
    if 'gpt-4o-mini' in models and all(models['gpt-4o-mini']['rows'] and select(models['gpt-4o-mini']['rows'], f'sweep/n{n}', c)
                                       for n in (8, 16, 24) for c in ('llm_cnp', 'llm_centralized')):
        # Live LLM: total tokens per task and mean response against team size. Four LLM conditions plus the formula reference.
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.2), layout='constrained', sharex=True)
        llm_panel(axes, 'gpt-4o-mini', figure_series(models, 'gpt-4o-mini'))
        axes[1].set_xlabel('Agents')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
        save(fig, 'llm_replication')
        extra = [name for name in models if name != 'gpt-4o-mini' and all(select(models[name]['rows'], f'sweep/n{n}', 'llm_cnp') for n in (8, 16, 24))]
        if extra:
            names = ['gpt-4o-mini'] + extra
            fig, axes = plt.subplots(len(names), 2, figsize=(WIDTH, 2.45*len(names)+1.7), layout='constrained', sharex=True)
            for row, name in zip(axes, names):
                llm_panel(row, name, figure_series(models, name), one_label=True)
                row[0].set_ylabel(name.replace('claude-', '').replace('-instruct', ''), fontsize=11, fontweight='bold')
            axes[0][0].set_title('Tokens / task'); axes[0][1].set_title('Mean response')
            for ax in axes[-1]:
                ax.set_xlabel('Agents')
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
            save(fig, 'llm_models')
    (directory/'manifest.json').write_text(json.dumps({'width_cm':12,'font_pt':11,'error_bars':'95% CI of trial mean','files':paths},indent=2))
    return paths


def write_report(output='outputs/v2', path='docs/results_v2.md'):
    output = Path(output)
    data = json.loads((output/'results/results.json').read_text())
    lines = ['# v2 regenerated results', '', 'Generated from saved CSV/JSON summaries by `scripts/write_results_v2.py`; do not edit numbers by hand.', '',
             'Every simulation point uses 30 trials, seeds 0–29. Entries are mean ± sample standard deviation [95% Student-t CI of the mean]. '
             'Managed task denominators include unserved tasks; duration metrics describe served managed tasks. '
             'Soft proficiency theta=0 except E3 eligibility theta=0.6. Initialization is included in message/token rates.', '',
             'Polling greedy is full-information but myopic; it is not the offline optimum. '
             'CNP is manager-mediated bounded CNP, with capability shortlist by default. '
             'Wire tokens use o200k_base on serialized JSON-RPC-style envelopes; these are control-plane sizes, not provider-billed LLM usage.', '']
    primary = data['experiment_1']['groups']['200']
    comparisons = data['experiment_1']['paired_relative_percent']['200']['greedy']
    cnp, greedy, cached = [primary[name] for name in ('cnp','greedy','cached_central')]
    saving_messages = 100*(1-cnp['messages_per_task']['mean']/greedy['messages_per_task']['mean'])
    saving_tokens = 100*(1-cnp['tokens_per_task']['mean']/greedy['tokens_per_task']['mean'])
    lines += ['## Headline findings', '',
              f'At primary load, CNP makespan={cnp["makespan"]["mean"]:.2f}, response={cnp["mean_response"]["mean"]:.2f}; '
              f'full-information greedy/cached central makespan={greedy["makespan"]["mean"]:.2f}, response={greedy["mean_response"]["mean"]:.2f}. '
              f'CNP saves {saving_messages:.2f}% of messages and {saving_tokens:.2f}% of measured wire tokens against polling greedy '
              '(ratios of trial-mean rates, not API-billing savings).', '',
              f'Cached central exactly matches polling assignments at fresh state/no private load, at '
              f'{cached["messages_per_task"]["mean"]:.2f} messages and {cached["tokens_per_task"]["mean"]:.2f} tokens/task, '
              f'versus CNP {cnp["messages_per_task"]["mean"]:.2f} messages and {cnp["tokens_per_task"]["mean"]:.2f} tokens/task. '
              'Cheap cached routing is therefore a necessary comparator; bounded bidding does not have a universal cost advantage over centralization.', '']
    for metric, v in comparisons.items():
        lines += [f'CNP versus greedy, {metric}: paired relative difference {v["mean_difference"]:.2f}% '
                  f'[95% bootstrap {v["ci95"][0]:.2f}%, {v["ci95"][1]:.2f}%]; '
                  f'within pre-specified ±5%: **{v["within_margin"]}**'
                  + ('' if v['within_margin'] else
                     f' (the interval extends {"below -5%, i.e. CNP may be more than 5% better" if v["ci95"][0] < -5 else ""}'
                     f'{" and " if v["ci95"][0] < -5 and v["ci95"][1] > 5 else ""}'
                     f'{"above +5%, i.e. CNP may be more than 5% worse" if v["ci95"][1] > 5 else ""}{"; the relative interval also includes 0" if v["ci95"][0] <= 0 <= v["ci95"][1] else ""})')
                  + '.', '']
    private = data['experiment_6']['groups']['0.5']
    loaded = [data['experiment_6']['groups'][rho] for rho in ('0.25', '0.5', '1')]
    response = lambda group, name: group[name]['mean_response']['mean']
    capability_beats_cached = all(response(g, 'cnp') < response(g, 'cached_central') for g in loaded)
    random_beats_cached = all(response(g, 'cnp_random') < response(g, 'cached_central') for g in loaded)
    lines += [f'With private-load ratio 0.5, mean response is cached central '
              f'{private["cached_central"]["mean_response"]["mean"]:.2f}, capability CNP '
              f'{private["cnp"]["mean_response"]["mean"]:.2f}, random CNP '
              f'{private["cnp_random"]["mean_response"]["mean"]:.2f}, and polling greedy '
              f'{private["greedy"]["mean_response"]["mean"]:.2f}. All degrade under added load. '
              f'At every private-load ratio >0, capability-shortlist CNP is {"below" if capability_beats_cached else "not consistently below"} cached central '
              f'and random-shortlist CNP is {"below" if random_beats_cached else "above"} it: live bids reduce the penalty from registry state that omits private jobs '
              'only when combined with an informed shortlist; bidding alone does not.', '']
    llm_path = output/'results/llm_results.json'
    if llm_path.exists():
        llm = json.loads(llm_path.read_text())
        dist = llm.get('bid_error_distribution')
        agreement = {c: [g[c]['routing_agreement']['mean'] for g in llm['aggregate'].values()] for c in ('llm_cnp', 'llm_centralized')}
        if dist:
            lines += [f'Live LLM replication ({llm["model"]}): individual LLM bids have median absolute relative error '
                      f'{dist["median_absolute"]:.2f} (mean {dist["mean_absolute"]:.2f}); {100*dist["share_within_10_percent"]:.1f}% are within 10% '
                      f'and {100*dist["share_within_25_percent"]:.1f}% within 25% of the formula bid; median signed error {dist["median_signed"]:+.2f}, '
                      f'{100*dist["share_overestimates"]:.1f}% overestimates, {100*dist["share_zero_bids"]:.1f}% zero bids. '
                      f'Routing agreement is {min(agreement["llm_cnp"]):.2f}-{max(agreement["llm_cnp"]):.2f} for LLM bidding (k=3) and '
                      f'{min(agreement["llm_centralized"]):.2f}-{max(agreement["llm_centralized"]):.2f} for LLM centralized routing. '
                      'The CNP condition chooses among a formula-computed capability shortlist, so its smaller quality loss partly reflects '
                      'that deterministic pre-filter, not bidding alone. This measures single-shot JSON arithmetic without a scratchpad or tools.', '']
    def fmt(v):
        return f'{v["mean"]:.3f} ± {v["std"]:.3f} [{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}]'
    for experiment, summary in data.items():
        lines += [f'## {experiment.replace("_", " ").title()}', '']
        for point, group in summary['groups'].items():
            lines += [f'### {point}', '']
            for metrics in (('makespan','mean_response','mean_wait','p95_wait','fairness'),
                            ('messages_per_task','tokens_per_task','unserved_rate','declines_per_task','reannouncements_per_task')):
                lines += ['| Mechanism | '+' | '.join(metrics)+' |', '|---|'+'---:|'*len(metrics)]
                for name, stats in group.items():
                    lines.append('| '+NAMES.get(name, 'CNP random')+' | '+' | '.join(fmt(stats[m]) for m in metrics)+' |')
                lines.append('')
            if 'optimal_ratio' in next(iter(group.values())):
                lines += ['Exact optimum: makespan only, four agents, at most eleven tasks, CNP k=2. '
                          'Offline routes may reorder tasks, but may not pre-travel before release. No batching delay is imposed on the optimum.', '',
                          '| Mechanism | Makespan / optimum |', '|---|---:|']
                lines.extend(f'| {NAMES[n]} | {fmt(v["optimal_ratio"])} |' for n,v in group.items())
                lines.append('')
            comparison = summary['paired_relative_percent'].get(point, {})
            if comparison:
                lines += ['Paired relative differences are mean trial percentages, 100×(CNP−reference)/reference; 10,000 paired bootstrap resamples. '
                          'The ±5% margin was specified before examining results. Both CI endpoints must lie inside it.', '',
                          '| Reference | Metric | Difference (%) | Bootstrap 95% CI (%) | Absolute difference [95% CI] | Within ±5%? |', '|---|---|---:|---|---|---|']
                for reference, metrics in comparison.items():
                    for m,v in metrics.items():
                        absolute = v['absolute_difference']
                        lines.append(f'| {NAMES[reference]} | {m} | {v["mean_difference"]:.3f} | [{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}] | '
                                     f'{absolute["mean_difference"]:.3f} [{absolute["ci95"][0]:.3f}, {absolute["ci95"][1]:.3f}] | {v["within_margin"]} |')
                lines.append('')
        lines += ['### Measured control-message sizes (pooled physical messages)', '',
                  '| Mechanism | Type | Messages | Mean tokens | Total bytes |', '|---|---|---:|---:|---:|']
        for name, types in summary['tokens_by_type'].items():
            for kind,v in types.items():
                lines.append(f'| {NAMES.get(name, name)} | {kind} | {v["messages"]} | {v["mean_tokens"]:.2f} | {v["bytes"]} |')
        lines.append('')
    sens = json.loads((output/'results/token_sensitivity.json').read_text())
    lines += ['## Token sensitivity', '', sens['scope'], '', sens['queue_policy'], '', sens['cfp_policy'], '',
              '| Queue items | CFP multiplier | CNP tokens/task | Greedy tokens/task | Cached tokens/task | CNP saving (%) | Cached saving (%) |', '|---:|---:|---:|---:|---:|---:|---:|']
    for r in sens['rows']:
        lines.append('| '+' | '.join(f'{r[k]:.3f}' for k in ('queue_length','cfp_multiplier','cnp_tokens_per_task','greedy_tokens_per_task','cached_tokens_per_task','cnp_saving_percent','cached_saving_percent'))+' |')
    llm_path = output/'results/llm_results.json'
    if llm_path.exists():
        llm = json.loads(llm_path.read_text())
        lines += ['', '## Small live LLM replication', '', llm['limitations'], '',
                  f'Status: {llm["status"]}. Actual API spend: USD {llm["budget"]["actual_usd"]:.8f}; '
                  f'unknown-cost reservations: USD {llm["budget"]["reserved_unknown_usd"]:.8f}. '
                  f'API attempts: {llm["budget"]["calls"]}; invalid/failed replayed attempts: {llm["invalid_or_failed_calls"]}.', '',
                  f'Calls without trustworthy usage: {llm.get("calls_without_usage", 0)}. '
                  'Token/latency tables include all recorded attempts; unavailable usage is recorded as unknown and contributes zero to token totals, so those totals are lower bounds if any call lacks usage.', '',
                  f'Unique cached attempts: {llm.get("unique_cached_attempts", "not recorded")}; '
                  f'unique invalid/failed attempts: {llm.get("unique_failed_attempts", "not recorded")}; '
                  f'unique attempts without usage: {llm.get("unique_calls_without_usage", "not recorded")}.', '',
                  f'Journal cost lacking a per-call usage record: USD {llm.get("journal_cost_without_call_usage_usd", 0):.8f}. '
                  'The cumulative journal remains the spend authority; missing per-call tokens are not reconstructed from a guessed price split.', '',
                  llm['routing_agreement_definition']+'. '+llm['bid_error_definition']+'.', '', 'Fallback: '+llm['fallback']+'.', '']
        if 'bid_error_distribution' in llm:
            d = llm['bid_error_distribution']
            lines += ['Pooled bid-error distribution ('+d['definition']+'):', '', '| Valid bids | Mean abs. | Median abs. | Within 10% | Within 25% | Median signed | Overestimates | Zero bids |',
                      '|---:|---:|---:|---:|---:|---:|---:|---:|',
                      f'| {d["valid_bids"]} | {d["mean_absolute"]:.3f} | {d["median_absolute"]:.3f} | {d["share_within_10_percent"]:.3f} | '
                      f'{d["share_within_25_percent"]:.3f} | {d["median_signed"]:+.3f} | {d["share_overestimates"]:.3f} | {d["share_zero_bids"]:.3f} |', '']
        for key, group in llm['aggregate'].items():
            lines += [f'### {key}', '']
            for metrics in (('makespan','mean_response','unserved_rate'), ('prompt_tokens_per_task','completion_tokens_per_task','calls_per_task','latency_per_task'),
                            ('invalid_response_rate','routing_agreement','bid_relative_error')):
                lines += ['| Condition | '+' | '.join(metrics)+' |', '|---|'+'---:|'*len(metrics)]
                for condition, stats in group.items():
                    lines.append('| '+condition+' | '+' | '.join(fmt(stats[m]) for m in metrics)+' |')
                lines.append('')
    else:
        lines += ['', '## Live replication', '', 'No completed live-result artifact yet. No live numbers inferred from simulation.', '']
    lines += llm_markdown(output)
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')
