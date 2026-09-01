"""
Scores backtest predictions against actual outcomes. This is the piece that
was missing: backtest.py only produced predictions, it never checked if
they were any good.

Run after backtest.py has produced backtester/data/backtest_results.json.
"""

import json
import sys

BRIER_BASELINE = 0.25  # a model that always guesses 50% scores this


def load_results(path: str = "backtester/data/backtest_results.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("actual_label") is not None]

    if not scored:
        return {"error": "No games with known outcomes to score against."}

    n = len(scored)
    correct = 0
    brier_sum = 0.0
    predicted_nrfi_count = 0
    actual_nrfi_count = 0

    for r in scored:
        pred = r["predicted_nrfi_prob"]
        actual = r["actual_label"]

        predicted_class = 1 if pred >= 0.5 else 0
        if predicted_class == actual:
            correct += 1

        brier_sum += (pred - actual) ** 2

        if predicted_class == 1:
            predicted_nrfi_count += 1
        if actual == 1:
            actual_nrfi_count += 1

    accuracy = correct / n
    brier_score = brier_sum / n

    return {
        "num_games_scored": n,
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier_score, 4),
        "brier_vs_baseline": round(BRIER_BASELINE - brier_score, 4),
        "predicted_nrfi_rate": round(predicted_nrfi_count / n, 4),
        "actual_nrfi_rate": round(actual_nrfi_count / n, 4),
    }


def print_report(stats: dict) -> None:
    if "error" in stats:
        print(stats["error"])
        return

    print(f"Games scored:        {stats['num_games_scored']}")
    print(f"Accuracy:            {stats['accuracy'] * 100:.1f}%")
    print(f"Brier score:         {stats['brier_score']}  (lower is better, 0.25 = always guessing 50%)")
    print(f"Beat 50/50 baseline: {'yes' if stats['brier_vs_baseline'] > 0 else 'no'} (by {stats['brier_vs_baseline']})")
    print(f"Predicted NRFI rate: {stats['predicted_nrfi_rate'] * 100:.1f}%")
    print(f"Actual NRFI rate:    {stats['actual_nrfi_rate'] * 100:.1f}%")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "backtester/data/backtest_results.json"
    results = load_results(path)
    stats = evaluate(results)
    print_report(stats)
