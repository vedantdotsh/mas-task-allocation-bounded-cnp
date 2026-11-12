import io
import json

import pytest

from mas_sim.llm import BudgetStop, Client, validate


@pytest.mark.parametrize('text,ids,expected', [('{"bid":12.5}', None, 12.5),
    ('{"bid":-1}', None, None), ('{"bid":true}', None, None), ('{"bid":NaN}', None, None),
    ('{"bid":"12"}', None, None), ('not json', None, None),
    ('{"agent_id":2}', [0, 1], None), ('{"agent_id":1}', [0, 1], 1)])
def test_validation(text, ids, expected):
    assert validate(text, ids) == expected


def test_budget_cache_and_invalid_retry(tmp_path, monkeypatch):
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY=test-placeholder')
    answers = iter(['invalid', '{"bid":12}'])
    def response(*args, **kwargs):
        return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'cost': .000021},
                                     'choices': [{'message': {'content': next(answers)}}]}).encode())
    monkeypatch.setattr('urllib.request.urlopen', response)
    monkeypatch.setattr('time.sleep', lambda _: None)
    client = Client(tmp_path)
    request = {'messages': [{'content': 'test'}]}
    assert client.request(request) == 12
    assert client.budget['calls'] == 2
    assert client.budget['actual_usd'] == pytest.approx(.000042)
    assert client.records[0]['valid'] is False
    cached = Client(tmp_path, cache_only=True)
    assert cached.request(request) == 12
    assert cached.new_calls == 0
    client.budget['actual_usd'] = 3.99999
    with pytest.raises(BudgetStop):
        client.request({'messages': []})
    assert client.budget['calls'] == 2


def test_missing_usage_stops_and_retains_reservation(tmp_path, monkeypatch):
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY=test-placeholder')
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(b'{"usage":{}}'))
    client = Client(tmp_path)
    with pytest.raises(BudgetStop):
        client.request({'messages': []})
    assert client.budget['calls'] == 1
    assert client.budget['reserved_unknown_usd'] > 0


def test_disk_journal_failure_does_not_retry_paid_request(tmp_path, monkeypatch):
    import mas_sim.llm as llm
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY=test-placeholder')
    requests = []
    def response(*args, **kwargs):
        requests.append(1)
        return io.BytesIO(json.dumps({'usage': {'prompt_tokens':100,'completion_tokens':10,'cost':.000021},
                                     'choices':[{'message':{'content':'{"bid":12}'}}]}).encode())
    save = llm.save_json
    def broken_save(path, value):
        if value.get('actual_usd', 0) > 0:
            raise PermissionError('simulated Windows journal lock')
        save(path,value)
    monkeypatch.setattr('urllib.request.urlopen',response)
    monkeypatch.setattr(llm,'save_json',broken_save)
    client=Client(tmp_path)
    with pytest.raises(BudgetStop, match='journal'):
        client.request({'messages':[]})
    assert len(requests)==1
    assert client.records[0]['cost_usd']==.000021


@pytest.mark.parametrize('text,ids,expected', [
    ('Waiting 0. Travel 4.5. Total 12.5\n{"bid": 12.5}', None, 12.5),
    ('Draft {"bid": 3} corrected:\n```json\n{"bid": 12.5}\n```', None, 12.5),
    ('Agent 2 has the smallest duration.\n{"agent_id": 2}', [0, 1, 2], 2),
    ('no JSON at all', None, None), ('{"bid": 5}\nthen prose', None, 5.0), ('{"bid": -3}', None, None),
    ('{"agent_id": 7}', [0, 1], None), ('{"bid": true}', None, None), ('', None, None)])
def test_worked_parser_takes_last_object_with_strict_validation(text, ids, expected):
    from mas_sim.llm import validate_worked
    assert validate_worked(text, ids) == expected


def test_fence_stripping_is_the_only_relaxation():
    from mas_sim.llm import strip_fence, validate
    assert validate(strip_fence('```json\n{"bid": 4}\n```')) == 4
    assert validate(strip_fence('```\n{"bid": 4}\n```')) == 4
    assert validate(strip_fence('The answer is {"bid": 4}')) is None


def test_worked_truncation_is_invalid_and_retried(tmp_path, monkeypatch):
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY=test-placeholder')
    answers = iter([('{"bid": 3}', 'length'), ('working\n{"bid": 12}', 'stop')])
    def response(*args, **kwargs):
        content, reason = next(answers)
        return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'cost': .00002},
                                     'choices': [{'message': {'content': content}, 'finish_reason': reason}]}).encode())
    monkeypatch.setattr('urllib.request.urlopen', response)
    monkeypatch.setattr('time.sleep', lambda _: None)
    client = Client(tmp_path, journal='budget_test.json', limit_usd=2.0)
    assert client.request({'messages': [], 'max_tokens': 100}, worked=True) == 12
    assert [r['truncated'] for r in client.records] == [True, False]
    assert [r['valid'] for r in client.records] == [False, True]
    assert (tmp_path/'outputs/v2/llm_cache/budget_test.json').exists()
    assert not (tmp_path/'outputs/v2/llm_cache/budget.json').exists()


def test_parallel_requests_keep_the_journal_consistent(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY=test-placeholder')
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(
        {'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'cost': .00001},
         'choices': [{'message': {'content': '{"bid": 1}'}, 'finish_reason': 'stop'}]}).encode()))
    client = Client(tmp_path, journal='budget_test.json', limit_usd=2.0)
    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(lambda i: client.request({'messages': [{'content': str(i)}]}), range(40)))
    assert values == [1.0]*40
    assert client.budget['calls'] == 40 and client.budget['reserved_unknown_usd'] == pytest.approx(0, abs=1e-12)
    assert client.budget['actual_usd'] == pytest.approx(40*.00001)


def test_oneshot_requests_are_unchanged_by_the_new_variants():
    """The original one-shot gpt-4o-mini run must replay byte-for-byte: same request hash as the saved cache entry."""
    import hashlib
    from pathlib import Path
    from mas_sim.llm import make_request
    from mas_sim.scenario import ScenarioConfig, generate_scenario
    from mas_sim.v2 import simulate_v2
    from mas_sim.wire import serialize
    cache = Path(__file__).resolve().parents[1]/'outputs/v2/llm_cache'
    if not cache.exists():
        pytest.skip('no saved LLM cache')
    seen = []
    def hook(ctx, task, now):
        agents = ctx.candidates(task, now)[:3]
        if not seen:
            seen.extend(hashlib.sha256(serialize(make_request(ctx, task, now, a)).encode()).hexdigest() for a in agents)
        ctx.assign(agents[0], task, now)
    simulate_v2(generate_scenario(ScenarioConfig(n_agents=8, expected_tasks=30), 0), 'cnp', k=3, context_hook=hook)
    assert seen and all((cache/f'{h}.json').exists() for h in seen)
