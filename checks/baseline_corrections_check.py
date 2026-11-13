"""Supersedes the exploratory duplicate SSI/CBBA implementations.

Checks production experiment outputs against the preserved published evidence.
Run after scripts/run_experiments.py; no experiments or paid APIs are invoked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from correction_report import verify_results

if __name__ == "__main__":
    result = verify_results()
    print("PASS: all unaffected published values exact; k-ablation exact; "
          "fresh manager matches greedy; preserved result and LLM hashes match.")
    print(f"Changed SSI/CBBA leaves: {result['changed_baseline_leaf_count']}; "
          f"unchanged leaves: {result['unchanged_leaf_count']}")
