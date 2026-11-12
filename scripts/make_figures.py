"""Regenerate figures from saved outputs only."""
import argparse
from _bootstrap import OUTPUTS
from mas_sim.plotting import generate_all_figures
from mas_sim.report_v2 import generate_figures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v2','legacy'], default='v2')
    args = parser.parse_args()
    paths = generate_figures(OUTPUTS/'v2') if args.version == 'v2' else generate_all_figures(OUTPUTS)
    for path in paths:
        print(path)
