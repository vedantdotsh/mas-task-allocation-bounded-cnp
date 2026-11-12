"""v2 experiment 1; 30 paired trials per point."""
import argparse
from _bootstrap import OUTPUTS
from mas_sim.experiments import run_paper_experiment
from mas_sim.experiments_v2 import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v2','legacy'], default='v2')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.version == 'legacy':
        return run_paper_experiment(1, OUTPUTS)
    return run_experiments(OUTPUTS/'v2', workers=args.workers, numbers=[1])


if __name__ == '__main__':
    main()
