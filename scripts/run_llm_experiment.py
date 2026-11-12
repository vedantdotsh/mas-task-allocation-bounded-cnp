import argparse
import json

from _bootstrap import ROOT
from mas_sim.llm import MODEL_SPECS, plan, plan_model, run, run_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--model', choices=sorted(MODEL_SPECS), help='worked variant / extra model; omit for the original one-shot gpt-4o-mini run')
    parser.add_argument('--with-main', action='store_true', help='also run the 8-agent, 30-task main design (extra models only)')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    if args.model:
        main = True if args.with_main else None
        result = (plan_model(ROOT, args.model, main) if args.dry_run else
                  run_model(ROOT, args.model, cache_only=args.cache_only, smoke=args.smoke, include_main=main, workers=args.workers))
    else:
        result = plan(ROOT) if args.dry_run else run(ROOT, cache_only=args.cache_only, smoke=args.smoke)
    keys = ('status', 'budget', 'new_calls', 'valid_calls', 'invalid_or_failed_calls', 'truncated_calls', 'key_usage_before_usd', 'key_usage_after_usd')
    print(json.dumps(result if args.dry_run else {k: result[k] for k in keys if k in result}, indent=2))
