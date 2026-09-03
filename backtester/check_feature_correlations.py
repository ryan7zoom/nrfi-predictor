"""
Checks whether any feature actually correlates with the NRFI outcome.
Run this after a backtest to see if the model has any real signal to work
with, before assuming a training/model problem.

Uses only numpy (no scipy dependency) so it doesn't need a new package.
"""

import json
import sys
import numpy as np

from model import FEATURE_COLUMNS


def load_results(path: str = "backtester/data/backtest_results.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def check_correlations(results: list[dict]) -> None:
    labeled = [r for r in results if r.get("actual_label") is not None]
    if not labeled:
        print("No labeled games to check.")
        return

    print(f"Checking correlations across {len(labeled)} labeled games.\n")
    print(f"{'Feature':<32} {'Correlation':>12} {'Non-null':>10}")
    print("-" * 56)

    for col in FEATURE_COLUMNS:
        vals = []
        labels = []
        for r in labeled:
            v = r.get(col)
            if v is not None:
                vals.append(float(v))
                labels.append(r["actual_label"])

        if len(vals) < 10:
            print(f"{col:<32} {'n/a':>12} {len(vals):>10}")
            continue

        corr = pearson_corr(np.array(vals), np.array(labels))
        print(f"{col:<32} {corr:>12.4f} {len(vals):>10}")

    print()
    print("Correlations near 0 for every feature means none of them carry")
    print("real signal for NRFI on their own. Correlations don't need to be")
    print("large (0.05-0.15 is meaningful for a noisy sports outcome), but")
    print("if literally everything is under ~0.02, the features aren't")
    print("giving the model anything to learn from.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "backtester/data/backtest_results.json"
    results = load_results(path)
    check_correlations(results)
