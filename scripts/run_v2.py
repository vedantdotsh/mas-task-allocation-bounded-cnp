import argparse

from _bootstrap import OUTPUTS
from mas_sim.experiments_v2 import run_experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--experiments', type=int, nargs='+', default=list(range(1, 7)))
    args = parser.parse_args()
    run_experiments(OUTPUTS / 'v2', args.trials, args.workers, args.experiments)
