"""Small paid arithmetic-routing replication; no semantic task execution is claimed."""
import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import urllib.error
import urllib.request
from pathlib import Path

from .domain import bid
from .experiments_v2 import summarize, write_csv
from .scenario import ScenarioConfig, generate_scenario
from .simulation import simulate
from .v2 import simulate_v2
from .wire import local_state, serialize, task_data, count_tokens

MODEL = 'openai/gpt-4o-mini'
INPUT_PRICE, OUTPUT_PRICE = .15/1e6, .60/1e6
MAX_OUTPUT = 128
BUDGET, CALL_CAP = 4.0, 6000
# The worked variant has its own journal and a hard USD 2.00 ceiling on additional spend; the
# dry-run estimate must stay under USD 1.50. It shares the full-request cache with the one-shot run.
WORKED_BUDGET, WORKED_ESTIMATE_LIMIT, WORKED_JOURNAL = 2.0, 1.5, 'budget_worked.json'
WORKED_MAX_BID, WORKED_MAX_CENTRAL = 2048, 8192
INSTRUCTIONS = (
    'You route spatial service tasks. An agent appends work to its FIFO queue. '
    'It cannot start travelling until both now and its committed_finish have passed. '
    'Travel is Euclidean distance from queue_end_position to task location divided by speed. '
    'Execution duration is workload divided by its proficiency for required_skill (floor 0.05). '
    'Estimate the duration from now until this task completes, including waiting, travel and execution. '
    'Use only the supplied state. Return only the requested JSON object, without explanation.'
)
WORKED_INSTRUCTIONS = INSTRUCTIONS.rsplit(' Use only the supplied state.', 1)[0] + ' Use only the supplied state. '
WORKED_BID = ('Show brief step-by-step working: compute the waiting time, the travel time and the execution time separately, '
              'then add them to get the total. End your response with the answer as a JSON object on its own final line: '
              '{"bid": nonnegative number}.')
WORKED_CENTRAL = ('Show brief step-by-step working: for each agent compute the waiting time, travel time and execution time and '
                  'their sum, which is its estimated completion duration, one short line per agent; then choose the agent with the '
                  'smallest duration, breaking ties by smallest id. End your response with the answer as a JSON object on its own '
                  'final line: {"agent_id": integer}.')


def make_request(context, task, now, agent=None, variant='oneshot', model=None):
    model = model or {'id': MODEL, 'prices': (.15, .60)}
    price = {'prompt': model['prices'][0], 'completion': model['prices'][1]}
    content = {'now': now, 'task': task_data(task)}
    if variant == 'worked':
        content['agent' if agent is not None else 'agents'] = (
            local_state(context, agent, now) if agent is not None else [local_state(context, a, now) for a in context.agents])
        return {'model': model['id'], 'temperature': 0, 'max_tokens': WORKED_MAX_BID if agent is not None else WORKED_MAX_CENTRAL,
                'provider': {'max_price': price, 'require_parameters': True},
                'messages': [{'role': 'system', 'content': WORKED_INSTRUCTIONS + (WORKED_BID if agent is not None else WORKED_CENTRAL)},
                             {'role': 'user', 'content': serialize(content)}],
                'usage': {'include': True}}
    if agent is None:
        content['agents'] = [local_state(context, a, now) for a in context.agents]
        instruction = ' Choose the agent with the smallest duration; break ties by smallest id. Return {"agent_id": integer}.'
    else:
        content['agent'] = local_state(context, agent, now)
        instruction = ' Return {"bid": nonnegative number} for this agent.'
    return {'model': model['id'], 'temperature': 0, 'max_tokens': MAX_OUTPUT,
            'response_format': {'type': 'json_object'},
            'provider': {'max_price': price, 'require_parameters': True},
            'messages': [{'role': 'system', 'content': INSTRUCTIONS+instruction},
                         {'role': 'user', 'content': serialize(content)}],
            'usage': {'include': True}}


def validate(text, agent_ids=None):
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        value = data.get('bid' if agent_ids is None else 'agent_id')
        if isinstance(value, bool):
            return None
        if agent_ids is not None:
            return value if isinstance(value, int) and value in agent_ids else None
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def strip_fence(text):
    """Remove one surrounding markdown code fence; nothing else is relaxed."""
    text = (text or '').strip()
    if text.startswith('```') and text.endswith('```') and len(text) > 6:
        body = text[3:-3]
        text = body.split('\n', 1)[1] if body.startswith(('json', 'JSON')) and '\n' in body else body
    return text.strip()


def last_json_object(text):
    """The last decodable JSON object in free-form working, or None."""
    decoder = json.JSONDecoder()
    for start in reversed([i for i, c in enumerate(text) if c == '{']):
        try:
            data, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def validate_worked(text, agent_ids=None):
    """Same strict value validation as `validate`, applied to the last JSON object of the response."""
    data = last_json_object(text or '')
    return None if data is None else validate(json.dumps(data), agent_ids)


class BudgetStop(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


class SmokeStop(RuntimeError):
    pass


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus can briefly deny atomic replacement. Retry disk I/O,
    # never the paid inference, and leave the original budget intact on failure.
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


class Client:
    def __init__(self, root, *, cache_only=False, smoke_calls=None, journal='budget.json', limit_usd=BUDGET,
                 prices=(INPUT_PRICE, OUTPUT_PRICE), timeout=90):
        self.root = Path(root)
        self.directory = self.root / 'outputs/v2/llm_cache'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budget_path = self.directory / journal
        self.limit, self.prices, self.timeout = limit_usd, prices, timeout
        # Trials may run in parallel threads: the lock guards the spending journal and counters, never the HTTP call.
        self.lock, self.stopped = threading.Lock(), None
        self.budget = json.loads(self.budget_path.read_text()) if self.budget_path.exists() else {
            'actual_usd': 0.0, 'reserved_unknown_usd': 0.0, 'calls': 0, 'limit_usd': limit_usd, 'call_cap': CALL_CAP}
        self.cache_only, self.smoke_calls = cache_only, smoke_calls
        self.new_calls = 0
        self.records = []
        self.key = None
        if not cache_only:
            # Only this explicitly authorized file is read. Never persist or print its content.
            for line in (self.root / '.env').read_text(encoding='utf-8-sig').splitlines():
                name, sep, value = line.partition('=')
                if sep and name.strip() == 'OPENROUTER_API_KEY':
                    self.key = value.strip().strip('"').strip("'")
            if not self.key:
                raise RuntimeError('OPENROUTER_API_KEY is absent from repository .env')

    def request(self, request, *, agent_ids=None, metadata=None, worked=False, fenced=False):
        encoded = serialize(request)
        key = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        path = self.directory / f'{key}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            for record in cached['attempts']:
                if record['cost_usd'] is None:
                    record = {**record, 'prompt_tokens': None, 'completion_tokens': None}
                self.records.append({**record, **(metadata or {}), 'cached': True, 'request_hash': key})
            return cached['value']
        if self.cache_only:
            raise CacheMiss(f'No cached response for request {key}')
        attempts = []
        value = None
        # Persist a reservation before the request; an interrupted/ambiguous request remains charged
        # at its conservative upper bound until manually reconciled, never retried for free.
        reserve = ((len(encoded.encode('utf-8')) + 4096) * self.prices[0]
                   + request.get('max_tokens', MAX_OUTPUT) * self.prices[1])
        for attempt in range(2):
            with self.lock:
                if self.stopped:
                    raise self.stopped
                if self.smoke_calls is not None and self.new_calls >= self.smoke_calls:
                    raise SmokeStop('Twenty-call smoke limit reached')
                if (self.budget['actual_usd'] + self.budget['reserved_unknown_usd'] + reserve > self.limit
                        or self.budget['calls'] >= CALL_CAP):
                    raise BudgetStop('Hard USD/call cap would be exceeded')
                self.budget['reserved_unknown_usd'] += reserve
                self.budget['calls'] += 1
                save_json(self.budget_path, self.budget)
                self.new_calls += 1
            started = time.perf_counter()
            record = {'attempt': attempt+1, 'prompt_tokens': None, 'completion_tokens': None,
                      'cost_usd': None, 'valid': False, 'failed': False, 'error': None}
            try:
                http = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                    data=encoded.encode('utf-8'), headers={'Authorization': 'Bearer '+self.key,
                                                         'Content-Type': 'application/json'})
                with urllib.request.urlopen(http, timeout=self.timeout) as response:
                    answer = json.load(response)
                usage = answer.get('usage', {})
                prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
                cost = usage.get('cost')
                if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                    # No trustworthy billed cost: stop the entire run, retaining the reservation.
                    raise BudgetStop('API omitted valid usage.cost; conservative reservation retained')
                record.update(prompt_tokens=prompt or 0, completion_tokens=completion or 0, cost_usd=cost)
                reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                if reasoning:
                    record['reasoning_tokens'] = reasoning
                choice = answer.get('choices', [{}])[0]
                content = choice.get('message', {}).get('content') or ''
                if worked:
                    # A response cut off by max_tokens is invalid even if an earlier JSON object appears in the working.
                    truncated = choice.get('finish_reason') == 'length'
                    value = None if truncated else validate_worked(content, agent_ids)
                    record.update(finish_reason=choice.get('finish_reason'), truncated=truncated)
                else:
                    value = validate(strip_fence(content) if fenced else content, agent_ids)
                    if fenced and strip_fence(content) != content.strip():
                        record['fence_stripped'] = True
                    if choice.get('finish_reason') == 'length':
                        record.update(finish_reason='length', truncated=True)
                record.update(valid=value is not None, response=content, generation_id=answer.get('id'))
                with self.lock:
                    self.budget['reserved_unknown_usd'] -= reserve
                    self.budget['actual_usd'] += cost
                    try:
                        save_json(self.budget_path, self.budget)
                    except OSError:
                        self.stopped = BudgetStop('Cannot persist spending journal; stopping without another API call')
                        raise self.stopped from None
                    if cost > reserve:
                        self.stopped = BudgetStop('Provider exceeded conservative reservation; stop immediately')
                        raise self.stopped
            except BudgetStop as error:
                self.stopped = self.stopped or error
                raise
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, AttributeError) as error:
                record.update(failed=True, error=type(error).__name__)
                # Never log raw network errors, headers, request objects or credentials.
            finally:
                record['latency_seconds'] = time.perf_counter()-started
                attempts.append(record)
                with self.lock:
                    self.records.append({**record, **(metadata or {}), 'cached': False, 'request_hash': key})
            if value is not None:
                break
            if attempt == 0:
                time.sleep(1)
        save_json(path, {'request': request, 'value': value, 'attempts': attempts})
        return value


def design():
    return [('main', 8, 30, seed) for seed in range(5)] + [
        ('sweep', n, 20, seed) for n in (8, 16, 24) for seed in range(5)]


def plan(root):
    calls = prompt_tokens = managed = 0
    for study, n, expected, seed in design():
        scenario = generate_scenario(ScenarioConfig(n_agents=n, expected_tasks=expected), seed)
        managed += len(scenario.tasks)
        for mechanism in ('cnp', 'greedy'):
            def hook(ctx, task, now):
                nonlocal calls, prompt_tokens
                agents = ctx.candidates(task, now)[:3] if mechanism == 'cnp' else [None]
                for agent in agents:
                    request = make_request(ctx, task, now, agent)
                    calls += 1
                    prompt_tokens += sum(count_tokens(m['content'])+8 for m in request['messages'])+8
                eligible = agents if mechanism == 'cnp' else ctx.agents
                ctx.assign(min(eligible, key=lambda a: (bid(a, task, now), a.id)), task, now)
            simulate_v2(scenario, mechanism, k=3, context_hook=hook)
    estimate = prompt_tokens*INPUT_PRICE + calls*MAX_OUTPUT*OUTPUT_PRICE
    result = {'model': MODEL, 'temperature': 0, 'design': design(), 'managed_scenario_tasks': managed,
              'planned_calls': calls, 'planned_prompt_tokens_formula_trajectories': prompt_tokens,
              'completion_cap_per_call': MAX_OUTPUT, 'estimated_usd': estimate,
              'estimated_usd_with_one_retry_every_call': 2*estimate,
              'input_usd_per_million': .15, 'output_usd_per_million': .60,
              'note': 'Prompt estimate follows formula queues; live queues can differ. Hard guard reserves UTF-8 byte upper bound plus overhead.'}
    save_json(Path(root)/'outputs/v2/results/llm_plan.json', result)
    return result


def run(root, *, cache_only=False, smoke=False):
    lock = Path(root)/'outputs/v2/llm_cache/live.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    if not cache_only:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise BudgetStop('Another live run holds live.lock; do not run two spenders concurrently') from None
        os.write(descriptor, str(os.getpid()).encode('ascii'))
    try:
        return _run(root, cache_only=cache_only, smoke=smoke)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink()


def _run(root, *, cache_only=False, smoke=False):
    root = Path(root)
    client = Client(root, cache_only=cache_only, smoke_calls=20 if smoke else None)
    rows = []
    if not cache_only:
        p = json.loads((root/'outputs/v2/results/llm_plan.json').read_text())
        if p['estimated_usd'] >= 2:
            raise BudgetStop('Dry-run estimate must be below USD 2')
        if not smoke:
            smoke_result = json.loads((root/'outputs/v2/results/llm_smoke.json').read_text())
            if smoke_result['new_calls'] != 20 or not smoke_result['valid_calls']:
                raise BudgetStop('Complete and inspect the 20-call smoke test first')
    status = 'complete'
    try:
        for study, n, expected, seed in design():
            scenario = generate_scenario(ScenarioConfig(n_agents=n, expected_tasks=expected), seed)
            for mechanism in ('cnp', 'greedy'):
                label = 'cnp' if mechanism == 'cnp' else 'centralized'
                formula = simulate(scenario, mechanism, k=3)
                rows.append({'study': study, 'n_agents': n, 'seed': seed, 'condition': 'formula_'+label,
                             **formula.metrics, 'prompt_tokens_per_task': 0, 'completion_tokens_per_task': 0,
                             'calls_per_task': 0, 'latency_per_task': 0, 'invalid_response_rate': 0,
                             'routing_agreement': 1, 'bid_relative_error': 0})
                start_record = len(client.records)
                agreements, bid_errors, signed_errors = [], [], []
                def hook(ctx, task, now):
                    candidates = ctx.candidates(task, now)[:3] if mechanism == 'cnp' else ctx.agents
                    formula_choice = min(candidates, key=lambda a: (bid(a, task, now), a.id)).id
                    meta = {'study': study, 'n_agents': n, 'seed': seed,
                            'condition': 'llm_'+label, 'task_id': task.id}
                    if mechanism == 'cnp':
                        offers = []
                        for agent in candidates:
                            ctx.ledger.record('cfp', 'manager', agent.id, task.id)
                            value = client.request(make_request(ctx, task, now, agent), metadata={**meta, 'agent_id': agent.id})
                            if value is not None:
                                truth = bid(agent, task, now)
                                bid_errors.append(abs(value-truth)/truth if truth else abs(value-truth))
                                signed_errors.append((value-truth)/truth if truth else value-truth)
                                offers.append((value, agent.id))
                                ctx.ledger.record('bid', agent.id, 'manager', task.id, payload={'task_id': task.id, 'cost': value})
                        winner = min(offers)[1] if offers else None
                    else:
                        for agent in ctx.agents:
                            ctx.ledger.record('centralized_full_query', 'manager', agent.id, task.id)
                            ctx.ledger.record('centralized_full_response', agent.id, 'manager', task.id)
                        winner = client.request(make_request(ctx, task, now), agent_ids=[a.id for a in ctx.agents], metadata=meta)
                    agreements.append(int(winner == formula_choice))
                    if winner is None:
                        ctx.unserved.append(task.id)
                    else:
                        ctx.assign(ctx.agents[winner], task, now)
                result = simulate_v2(scenario, mechanism, k=3, context_hook=hook)
                calls = client.records[start_record:]
                total = len(scenario.tasks)
                row = {'study': study, 'n_agents': n, 'seed': seed, 'condition': 'llm_'+label,
                       **result.metrics,
                       'prompt_tokens_per_task': sum(c['prompt_tokens'] or 0 for c in calls)/total,
                       'completion_tokens_per_task': sum(c['completion_tokens'] or 0 for c in calls)/total,
                       'calls_per_task': len(calls)/total,
                       'cache_hits': sum(c['cached'] for c in calls),
                       'latency_per_task': (sum(c['latency_seconds'] for c in calls) + sum(c['attempt'] == 2 for c in calls))/total,
                       'invalid_response_rate': sum(not c['valid'] for c in calls)/len(calls),
                       'routing_agreement': sum(agreements)/len(agreements),
                       'bid_relative_error': sum(bid_errors)/len(bid_errors) if bid_errors else 0,
                       'bid_signed_relative_errors': signed_errors,
                       'formula_makespan': formula.metrics['makespan'], 'formula_mean_response': formula.metrics['mean_response']}
                rows.append(row)
                print(f'LLM {study} n={n} seed={seed} {label}: {len(calls)} calls; cumulative USD {client.budget["actual_usd"]:.6f}', flush=True)
    except (SmokeStop, BudgetStop, CacheMiss) as error:
        status = str(error)
    output = root/'outputs/v2/results'
    summary = {'status': status, 'model': MODEL, 'temperature': 0, 'budget': client.budget,
               'new_calls': client.new_calls, 'valid_calls': sum(c['valid'] for c in client.records),
               'invalid_or_failed_calls': sum(not c['valid'] for c in client.records),
               'failed_calls': sum(c['failed'] for c in client.records),
               'calls_without_usage': sum(c['cost_usd'] is None for c in client.records), 'trials': rows,
               'limitations': 'Small arithmetic-routing replication, not a semantic LLM-agent benchmark. Sequential API calls; wall latency sums observed attempts including invalids and one-second retry backoffs. Cache replay reports original API latency and usage. Per-condition logical calls can reuse identical requests; cumulative actual spending counts each fresh request only.',
               'routing_agreement_definition': 'formula decision on the LLM trajectory local state and same shortlist',
               'bid_error_definition': 'mean absolute relative error over valid individual bids',
               'fallback': 'one retry; then exclude invalid bidder; unserved if all invalid; centralized invalid after retry leaves task unserved',
               'aggregate': {}}
    summary['reported_call_cost_usd'] = sum(c['cost_usd'] or 0 for c in client.records)
    unique = {(c['request_hash'], c['attempt']): c for c in client.records}
    summary['logical_call_attempts'] = len(client.records)
    summary['unique_cached_attempts'] = len(unique)
    summary['unique_failed_attempts'] = sum(not c['valid'] for c in unique.values())
    summary['unique_calls_without_usage'] = sum(c['cost_usd'] is None for c in unique.values())
    summary['unique_reported_cost_usd'] = sum(c['cost_usd'] or 0 for c in unique.values())
    summary['journal_cost_without_call_usage_usd'] = client.budget['actual_usd'] - summary['unique_reported_cost_usd']
    # Pooled over every valid logical LLM bid; signed error = (LLM bid - formula bid)/formula bid.
    signed = [e for r in rows for e in r.get('bid_signed_relative_errors', [])]
    if signed:
        absolute = sorted(abs(e) for e in signed)
        middle = len(absolute) // 2
        median = absolute[middle] if len(absolute) % 2 else (absolute[middle-1]+absolute[middle])/2
        ordered = sorted(signed)
        signed_median = ordered[middle] if len(ordered) % 2 else (ordered[middle-1]+ordered[middle])/2
        summary['bid_error_distribution'] = {
            'valid_bids': len(signed), 'mean_absolute': sum(absolute)/len(absolute), 'median_absolute': median,
            'share_within_10_percent': sum(e <= .1 for e in absolute)/len(absolute),
            'share_within_25_percent': sum(e <= .25 for e in absolute)/len(absolute),
            'median_signed': signed_median, 'share_overestimates': sum(e > 0 for e in signed)/len(signed),
            'share_zero_bids': sum(e == -1 for e in signed)/len(signed),
            'definition': 'pooled valid logical LLM bids in llm_cnp trials; relative to the formula bid on the same state'}
    metrics = ('makespan', 'mean_response', 'prompt_tokens_per_task', 'completion_tokens_per_task',
               'calls_per_task', 'latency_per_task', 'invalid_response_rate', 'routing_agreement', 'bid_relative_error', 'unserved_rate')
    for study, n in sorted({(r['study'], r['n_agents']) for r in rows}):
        key = f'{study}/n{n}'
        summary['aggregate'][key] = {condition: {m: summarize([r[m] for r in rows if r['study'] == study and r['n_agents'] == n and r['condition'] == condition]) for m in metrics}
                                      for condition in sorted({r['condition'] for r in rows if r['study'] == study and r['n_agents'] == n})}
    prefix = 'llm_smoke' if smoke else 'llm_results'
    save_json(output/f'{prefix}.json', summary)
    save_json(output/f'{prefix}_calls.json', client.records)
    if rows:
        write_csv(output/f'{prefix}_raw.csv', rows)
    return summary


# ---- Worked-reasoning variant, extra models and zero-cost shortlist references ------------------------
# Same scenarios, seeds, shortlist and scheduler as the one-shot run; only the model and output contract
# differ. References: `shortlist_random` (seeded uniform pick from the same k=3 shortlist) and
# `shortlist_top1` (first shortlist entry); neither sends bids or calls a model.
# Model ids, prices and selection reasons come from the public OpenRouter catalogue fetched 2026-11-12
# (https://openrouter.ai/api/v1/models); prices are USD per million tokens.
TOTAL_PROJECT_LIMIT_USD = 8.0
MODEL_SPECS = {
    'gpt-4o-mini': {
        'id': MODEL, 'prices': (.15, .60), 'prefix': 'llm_worked', 'journal': WORKED_JOURNAL, 'variants': ('worked',),
        'references': True, 'fenced': False, 'limit_usd': WORKED_BUDGET, 'estimate_limit_usd': WORKED_ESTIMATE_LIMIT,
        'main': True, 'tokenizer_factor': 1.0, 'workers': 8,
        'why': 'Model of the existing one-shot run; the worked variant is paired with it.'},
    'gemini-3.1-flash-lite': {
        'id': 'google/gemini-3.1-flash-lite', 'prices': (.25, 1.5), 'prefix': 'llm_gemini_3_1_flash_lite',
        'journal': 'budget_gemini_3_1_flash_lite.json', 'variants': ('oneshot', 'worked'), 'references': False, 'fenced': True,
        'limit_usd': 3.5, 'estimate_limit_usd': 3.0, 'main': False, 'tokenizer_factor': 1.2, 'workers': 8,
        'why': 'Current non-preview small Google Flash-class chat model that lists temperature support; billed reasoning tokens were zero '
               'in probe calls with and without a reasoning-off request, so no reasoning switch is sent. It replaces claude-haiku-4.5 '
               '(USD 1/5 per million), whose roughly USD 6 estimate for the worked sweep would have exceeded the USD 8 total; the newer '
               'claude-haiku-5.5 does not list temperature support and cannot run at temperature 0.'},
    'llama-3.3-70b': {
        'id': 'meta-llama/llama-3.3-70b-instruct', 'prices': (.71, .71), 'prefix': 'llm_llama_3_3_70b',
        'journal': 'budget_llama_3_3_70b.json', 'variants': ('oneshot', 'worked'), 'references': False, 'fenced': True,
        'limit_usd': 3.0, 'estimate_limit_usd': 2.5, 'main': False, 'tokenizer_factor': 1.1, 'workers': 8,
        'why': 'Widely deployed open-weight dense chat model that one organisation can host on a small GPU node; no reasoning mode, so '
               'billed tokens are comparable.'},
}
WORKED_DEFAULTS = {'bid_completion': 900, 'central_completion_per_agent': 210, 'central_completion_base': 150,
                   'oneshot_bid': 16, 'oneshot_central': 8}
VARIANT_LABEL = {'oneshot': 'llm_', 'worked': 'llm_worked_'}


def spec_design(spec, include_main):
    return [item for item in design() if include_main or item[0] != 'main']


def key_usage(root):
    """Cumulative USD usage of the OpenRouter key. Only the numeric usage field is read or returned."""
    key = None
    for line in (Path(root)/'.env').read_text(encoding='utf-8-sig').splitlines():
        name, sep, value = line.partition('=')
        if sep and name.strip() == 'OPENROUTER_API_KEY':
            key = value.strip().strip('"').strip("'")
    http = urllib.request.Request('https://openrouter.ai/api/v1/key', headers={'Authorization': 'Bearer '+key})
    with urllib.request.urlopen(http, timeout=30) as response:
        usage = json.load(response).get('data', {}).get('usage')
    return float(usage) if isinstance(usage, (int, float)) else None


def calibrated_completion(root, spec):
    """Completion-token assumptions for the dry run; measured from the saved smoke test when it exists (x1.5 safety)."""
    path = Path(root)/f'outputs/v2/results/{spec["prefix"]}_smoke_calls.json'
    assumptions = dict(WORKED_DEFAULTS, source='defaults')
    if path.exists():
        calls = [c for c in json.loads(path.read_text()) if c.get('completion_tokens')]
        def mean(rows):
            return sum(rows)/len(rows)
        groups = {'bid_completion': [c['completion_tokens'] for c in calls if c['condition'] == 'llm_worked_cnp'],
                  'oneshot_bid': [c['completion_tokens'] for c in calls if c['condition'] == 'llm_cnp'],
                  'oneshot_central': [c['completion_tokens'] for c in calls if c['condition'] == 'llm_centralized'],
                  'central_completion_per_agent': [c['completion_tokens']/c['n_agents'] for c in calls
                                                   if c['condition'] == 'llm_worked_centralized']}
        for name, values in groups.items():
            if values:
                assumptions[name] = 1.5*mean(values)
        if groups['central_completion_per_agent']:
            assumptions['central_completion_base'] = 0
        assumptions['source'] = 'smoke-test means x1.5 safety factor, defaults where unmeasured'
    return assumptions


def plan_model(root, name, include_main=None):
    spec = MODEL_SPECS[name]
    include_main = spec['main'] if include_main is None else include_main
    assumptions = calibrated_completion(root, spec)
    in_price, out_price = (p/1e6 for p in spec['prices'])
    calls = prompt_tokens = completion_tokens = ceiling = managed = 0
    for study, n, expected, seed in spec_design(spec, include_main):
        scenario = generate_scenario(ScenarioConfig(n_agents=n, expected_tasks=expected), seed)
        managed += len(scenario.tasks)
        for mechanism in ('cnp', 'greedy'):
            for variant in spec['variants']:
                def hook(ctx, task, now):
                    nonlocal calls, prompt_tokens, completion_tokens, ceiling
                    agents = ctx.candidates(task, now)[:3] if mechanism == 'cnp' else [None]
                    for agent in agents:
                        request = make_request(ctx, task, now, agent, variant, spec)
                        calls += 1
                        prompt_tokens += spec['tokenizer_factor']*(sum(count_tokens(m['content'])+8 for m in request['messages'])+8)
                        if variant == 'worked':
                            completion_tokens += (assumptions['bid_completion'] if agent is not None else
                                                  assumptions['central_completion_base'] + assumptions['central_completion_per_agent']*n)
                        else:
                            completion_tokens += assumptions['oneshot_bid' if agent is not None else 'oneshot_central']
                        ceiling += request['max_tokens']
                    eligible = agents if mechanism == 'cnp' else ctx.agents
                    ctx.assign(min(eligible, key=lambda a: (bid(a, task, now), a.id)), task, now)
                simulate_v2(scenario, mechanism, k=3, context_hook=hook)
    estimate = prompt_tokens*in_price + completion_tokens*out_price
    result = {'model': spec['id'], 'temperature': 0, 'variants': spec['variants'], 'include_main': include_main,
              'design': spec_design(spec, include_main), 'managed_scenario_tasks': managed,
              'planned_calls': calls, 'planned_prompt_tokens': prompt_tokens,
              'tokenizer_factor': spec['tokenizer_factor'], 'assumed_completion_tokens': completion_tokens,
              'completion_assumptions': assumptions,
              'max_tokens': {'bid': WORKED_MAX_BID, 'centralized': WORKED_MAX_CENTRAL, 'oneshot': MAX_OUTPUT},
              'estimated_usd': estimate, 'estimated_usd_with_one_retry_every_call': 2*estimate,
              'ceiling_usd_every_call_hits_max_tokens': prompt_tokens*in_price + ceiling*out_price,
              'estimate_limit_usd': spec['estimate_limit_usd'], 'additional_spend_limit_usd': spec['limit_usd'],
              'total_project_limit_usd': TOTAL_PROJECT_LIMIT_USD,
              'prices_usd_per_million': {'input': spec['prices'][0], 'output': spec['prices'][1], 'catalogue_date': '2026-11-12'},
              'note': 'Prompt tokens use o200k_base counts times the tokenizer factor; live queues can differ. Hard guard: separate journal, '
                      'reservation = UTF-8 byte upper bound plus max_tokens, per-model USD ceiling including smoke test, and the key-level total.'}
    save_json(Path(root)/f'outputs/v2/results/{spec["prefix"]}_plan.json', result)
    return result


def run_model(root, name, *, cache_only=False, smoke=False, include_main=None, workers=None):
    lock = Path(root)/'outputs/v2/llm_cache/live.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    if not cache_only:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise BudgetStop('Another live run holds live.lock; do not run two spenders concurrently') from None
        os.write(descriptor, str(os.getpid()).encode('ascii'))
    try:
        return _run_model(root, name, cache_only=cache_only, smoke=smoke, include_main=include_main, workers=workers)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink()


def _trial_row(study, n, seed, condition, result, calls, total, agreements, bid_errors, signed_errors, formula):
    prompt = sum(c['prompt_tokens'] or 0 for c in calls)/total
    completion = sum(c['completion_tokens'] or 0 for c in calls)/total
    return {'study': study, 'n_agents': n, 'seed': seed, 'condition': condition, **result.metrics,
            'prompt_tokens_per_task': prompt, 'completion_tokens_per_task': completion, 'total_tokens_per_task': prompt+completion,
            'calls_per_task': len(calls)/total, 'cache_hits': sum(c['cached'] for c in calls),
            'latency_per_task': (sum(c['latency_seconds'] for c in calls) + sum(c['attempt'] == 2 for c in calls))/total,
            'invalid_response_rate': sum(not c['valid'] for c in calls)/len(calls) if calls else 0,
            'truncated_response_rate': sum(bool(c.get('truncated')) for c in calls)/len(calls) if calls else 0,
            'routing_agreement': sum(agreements)/len(agreements) if agreements else 1,
            'bid_relative_error': sum(bid_errors)/len(bid_errors) if bid_errors else 0,
            'bid_signed_relative_errors': signed_errors,
            'formula_makespan': formula.metrics['makespan'], 'formula_mean_response': formula.metrics['mean_response']}


def _run_model(root, name, *, cache_only=False, smoke=False, include_main=None, workers=None):
    root = Path(root)
    spec = MODEL_SPECS[name]
    output = root/'outputs/v2/results'
    previous = output/f'{spec["prefix"]}_results.json'
    if include_main is None:
        # Cache replay reproduces the design that was actually run.
        include_main = (json.loads(previous.read_text()).get('include_main', spec['main'])
                        if cache_only and previous.exists() else spec['main'])
    client = Client(root, cache_only=cache_only, smoke_calls=20 if smoke else None, journal=spec['journal'],
                    limit_usd=spec['limit_usd'], prices=tuple(p/1e6 for p in spec['prices']), timeout=420)
    rows, usage_before, usage_after = [], None, None
    status = 'complete'
    if not cache_only:
        p = json.loads((output/f'{spec["prefix"]}_plan.json').read_text())
        if p['estimated_usd'] > spec['estimate_limit_usd']:
            raise BudgetStop(f'Dry-run estimate must not exceed USD {spec["estimate_limit_usd"]}')
        usage_before = key_usage(root)
        if not smoke:
            if usage_before is not None and usage_before + p['estimated_usd'] > TOTAL_PROJECT_LIMIT_USD:
                raise BudgetStop('Estimate would take total key usage over the project limit; model skipped')
            smoke_result = json.loads((output/f'{spec["prefix"]}_smoke.json').read_text())
            if smoke_result['new_calls'] != 20 or not smoke_result['valid_calls']:
                raise BudgetStop('Complete and inspect the 20-call smoke test first')

    def trial(study, n, seed, mechanism, variant, scenario):
        """One scenario under cnp, greedy (centralized) or a zero-cost reference condition."""
        label = {'cnp': 'cnp', 'greedy': 'centralized'}.get(mechanism)
        condition = VARIANT_LABEL[variant]+label if label else mechanism
        worked = variant == 'worked'
        agreements, bid_errors, signed_errors = [], [], []

        def hook(ctx, task, now):
            candidates = ctx.agents if mechanism == 'greedy' else ctx.candidates(task, now)[:3]
            formula_choice = min(candidates, key=lambda a: (bid(a, task, now), a.id)).id
            meta = {'study': study, 'n_agents': n, 'seed': seed, 'condition': condition, 'task_id': task.id}
            if mechanism == 'shortlist_top1':
                winner = candidates[0].id
            elif mechanism == 'shortlist_random':
                winner = candidates[int(ctx.rng.integers(len(candidates)))].id
            elif mechanism == 'cnp':
                offers = []
                for agent in candidates:
                    ctx.ledger.record('cfp', 'manager', agent.id, task.id)
                    value = client.request(make_request(ctx, task, now, agent, variant, spec),
                                           metadata={**meta, 'agent_id': agent.id}, worked=worked, fenced=spec['fenced'])
                    if value is not None:
                        truth = bid(agent, task, now)
                        bid_errors.append(abs(value-truth)/truth if truth else abs(value-truth))
                        signed_errors.append((value-truth)/truth if truth else value-truth)
                        offers.append((value, agent.id))
                        ctx.ledger.record('bid', agent.id, 'manager', task.id, payload={'task_id': task.id, 'cost': value})
                winner = min(offers)[1] if offers else None
            else:
                for agent in ctx.agents:
                    ctx.ledger.record('centralized_full_query', 'manager', agent.id, task.id)
                    ctx.ledger.record('centralized_full_response', agent.id, 'manager', task.id)
                winner = client.request(make_request(ctx, task, now, variant=variant, model=spec), agent_ids=[a.id for a in ctx.agents],
                                        metadata=meta, worked=worked, fenced=spec['fenced'])
            agreements.append(int(winner == formula_choice))
            if winner is None:
                ctx.unserved.append(task.id)
            else:
                ctx.assign(ctx.agents[winner], task, now)

        base = 'greedy' if mechanism == 'greedy' else 'cnp'
        result = simulate_v2(scenario, base, k=3, context_hook=hook)
        calls = [c for c in list(client.records) if label and c['condition'] == condition and c['study'] == study
                 and c['n_agents'] == n and c['seed'] == seed]
        return _trial_row(study, n, seed, condition, result, calls, len(scenario.tasks),
                          agreements, bid_errors, signed_errors, simulate(scenario, base, k=3))

    def formula_rows(study, n, seed, scenario):
        out = []
        for mechanism in ('cnp', 'greedy'):
            formula = simulate(scenario, mechanism, k=3)
            out.append({'study': study, 'n_agents': n, 'seed': seed,
                        'condition': 'formula_'+('cnp' if mechanism == 'cnp' else 'centralized'), **formula.metrics,
                        'prompt_tokens_per_task': 0, 'completion_tokens_per_task': 0, 'total_tokens_per_task': 0,
                        'calls_per_task': 0, 'latency_per_task': 0, 'invalid_response_rate': 0, 'truncated_response_rate': 0,
                        'routing_agreement': 1, 'bid_relative_error': 0})
        return out

    try:
        if smoke:
            # Twenty calls spanning every prompt and the largest team: bids on the main design, then centralized at n=24.
            stages, limit = [], 0
            for mechanism, case in (('cnp', ('main', 8, 30, 0)), ('greedy', ('sweep', 24, 20, 0))):
                for variant in spec['variants']:
                    limit += 20//(2*len(spec['variants']))
                    stages.append((case, mechanism, variant, limit))
            for (study, n, expected, seed), mechanism, variant, limit in stages:
                client.smoke_calls = limit
                try:
                    trial(study, n, seed, mechanism, variant, generate_scenario(ScenarioConfig(n_agents=n, expected_tasks=expected), seed))
                except SmokeStop:
                    pass
            status = 'smoke complete'
        else:
            slots, pool = [], ThreadPoolExecutor(max_workers=workers or spec['workers'])
            try:
                for study, n, expected, seed in spec_design(spec, include_main):
                    scenario = generate_scenario(ScenarioConfig(n_agents=n, expected_tasks=expected), seed)
                    slots.extend(formula_rows(study, n, seed, scenario))
                    jobs = ([(m, 'worked') for m in ('shortlist_random', 'shortlist_top1')] if spec['references'] else [])
                    jobs += [(m, v) for v in spec['variants'] for m in ('cnp', 'greedy')]
                    slots.extend(pool.submit(trial, study, n, seed, m, v, scenario) for m, v in jobs)
                for slot in slots:
                    try:
                        rows.append(slot.result() if hasattr(slot, 'result') else slot)
                    except (SmokeStop, BudgetStop, CacheMiss) as error:
                        status = status if status != 'complete' else str(error)
                        client.stopped = client.stopped or error
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    except (SmokeStop, BudgetStop, CacheMiss) as error:
        status = str(error)
    if not cache_only:
        try:
            usage_after = key_usage(root)
        except (urllib.error.URLError, OSError, ValueError):
            usage_after = None
    records = list(client.records)
    unique = {(c['request_hash'], c['attempt']): c for c in records}
    summary = {'status': status, 'model': spec['id'], 'model_key': name, 'variants': spec['variants'], 'temperature': 0,
               'include_main': include_main, 'prices_usd_per_million': {'input': spec['prices'][0], 'output': spec['prices'][1]},
               'budget': client.budget, 'key_usage_before_usd': usage_before, 'key_usage_after_usd': usage_after,
               'new_calls': client.new_calls, 'valid_calls': sum(c['valid'] for c in records),
               'invalid_or_failed_calls': sum(not c['valid'] for c in records),
               'truncated_calls': sum(bool(c.get('truncated')) for c in records),
               'failed_calls': sum(c['failed'] for c in records),
               'reasoning_tokens': sum(c.get('reasoning_tokens', 0) for c in records),
               'calls_without_usage': sum(c['cost_usd'] is None for c in records), 'trials': rows,
               'logical_call_attempts': len(records), 'unique_cached_attempts': len(unique),
               'unique_failed_attempts': sum(not c['valid'] for c in unique.values()),
               'unique_truncated_attempts': sum(bool(c.get('truncated')) for c in unique.values()),
               'unique_calls_without_usage': sum(c['cost_usd'] is None for c in unique.values()),
               'unique_reported_cost_usd': sum(c['cost_usd'] or 0 for c in unique.values()),
               'max_tokens': {'bid': WORKED_MAX_BID, 'centralized': WORKED_MAX_CENTRAL, 'oneshot': MAX_OUTPUT},
               'why_chosen': spec['why'],
               'limitations': 'Same small arithmetic-routing replication as the one-shot run. Worked variant: brief step-by-step working '
                              'before the final JSON line. Sequential calls within a trial, trials in parallel threads; wall latency sums observed '
                              'attempts including invalids and one-second retry backoffs, so it excludes queueing between trials. Provider routing '
                              'on OpenRouter may differ between calls.',
               'fallback': 'worked: last JSON object parsed and strictly validated; a response cut off by max_tokens (finish_reason length) is '
                           'invalid. one-shot other models: a markdown code fence around the bare JSON object is stripped before the same strict '
                           'validation. One retry; then exclude invalid bidder; unserved if all invalid; centralized invalid after retry leaves task unserved',
               'aggregate': {}}
    metrics = ('makespan', 'mean_response', 'prompt_tokens_per_task', 'completion_tokens_per_task', 'total_tokens_per_task',
               'calls_per_task', 'latency_per_task', 'invalid_response_rate', 'truncated_response_rate', 'routing_agreement',
               'bid_relative_error', 'unserved_rate')
    for study, n in sorted({(r['study'], r['n_agents']) for r in rows}):
        group = [r for r in rows if r['study'] == study and r['n_agents'] == n]
        summary['aggregate'][f'{study}/n{n}'] = {c: {m: summarize([r[m] for r in group if r['condition'] == c]) for m in metrics}
                                                 for c in sorted({r['condition'] for r in group})}
    if not cache_only:
        # Replays rewrite the result files, so the key-level usage readings live in an append-only log.
        log_path = output/'llm_key_usage.json'
        log = json.loads(log_path.read_text()) if log_path.exists() else []
        log.append({'model': name, 'invocation': ('smoke (20 calls)' if smoke else 'main+sweep' if include_main else 'sweep'),
                    'key_usage_before_usd': usage_before, 'key_usage_after_usd': usage_after, 'new_calls': client.new_calls})
        save_json(log_path, log)
    prefix = spec['prefix']+('_smoke' if smoke else '_results')
    save_json(output/f'{prefix}.json', summary)
    save_json(output/f'{prefix}_calls.json', records)
    if rows:
        write_csv(output/f'{prefix}_raw.csv', rows)
    return summary
