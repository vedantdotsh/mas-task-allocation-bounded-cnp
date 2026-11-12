"""Default v2 reproduction. The live component reads cache and never spends money."""
import argparse
from _bootstrap import OUTPUTS, ROOT
from mas_sim.experiments import run_paper_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['legacy', 'v2'], default='v2')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.version == 'v2':
        from mas_sim.experiments_v2 import run_experiments
        from mas_sim.report_v2 import sensitivity, generate_figures, write_report
        from mas_sim.example import write_example
        from mas_sim.llm import MODEL_SPECS, run, plan, run_model
        run_experiments(OUTPUTS/'v2', workers=args.workers)
        plan(ROOT)
        live = run(ROOT, cache_only=True)
        if live['status'] != 'complete':
            raise RuntimeError('LLM cache is incomplete: '+live['status'])
        for name, spec in MODEL_SPECS.items():
            # Worked variant and extra models: replay only the runs that exist, from cache.
            if (ROOT/f'outputs/v2/results/{spec["prefix"]}_results.json').exists():
                replay = run_model(ROOT, name, cache_only=True)
                if replay['status'] != 'complete':
                    raise RuntimeError(f'LLM cache for {name} is incomplete: '+replay['status'])
        sensitivity(OUTPUTS/'v2')
        write_example(ROOT/'docs/running_example.md')
        write_report(OUTPUTS/'v2', ROOT/'docs/results_v2.md')
        paths = generate_figures(OUTPUTS/'v2')
        print(f'Completed v2 E1–E6, cached LLM replay and {len(paths)} figure files; no API calls.')
        return
    for number in range(1, 6):
        run_paper_experiment(number, OUTPUTS)
    from mas_sim.plotting import generate_all_figures
    from mas_sim.reproduction import write_reproduction
    paths = generate_all_figures(OUTPUTS)
    write_reproduction(OUTPUTS)
    print(f"Completed five experiments and {len(paths)} figure files.")


if __name__ == "__main__":
    main()
