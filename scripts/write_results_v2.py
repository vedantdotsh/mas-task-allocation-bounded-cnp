from _bootstrap import OUTPUTS, ROOT
from mas_sim.example import write_example
from mas_sim.report_v2 import generate_figures, sensitivity, write_report


if __name__ == '__main__':
    sensitivity(OUTPUTS/'v2')
    write_example(ROOT/'docs/running_example.md')
    write_report(OUTPUTS/'v2', ROOT/'docs/results_v2.md')
    print(f'Wrote generated tables, example and {len(generate_figures(OUTPUTS/"v2"))} figure files.')
