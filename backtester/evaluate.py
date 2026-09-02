"""
Scores backtest predictions against actual outcomes, split into an early
window and a later window so you can see whether the model actually gets
better once it has more games of history to learn from.
"""

import json
import sys
from datetime import datetime

BRIER_BASELINE = 0.25  # a model that always guesses 50% scores this
DEFAULT_SPLIT_DAYS = 45  # first N days = "early", everything after = "later"


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

    return {
        "num_games_scored": n,
        "accuracy": round(correct / n, 4),
        "brier_score": round(brier_sum / n, 4),
        "brier_vs_baseline": round(BRIER_BASELINE - (brier_sum / n), 4),
        "predicted_nrfi_rate": round(predicted_nrfi_count / n, 4),
        "actual_nrfi_rate": round(actual_nrfi_count / n, 4),
    }


def split_by_date(results: list[dict], split_days: int) -> tuple[list[dict], list[dict]]:
    """First split_days days of the range = early, rest = later."""
    scored = [r for r in results if r.get("actual_label") is not None]
    if not scored:
        return [], []

    dates = sorted(datetime.strptime(r["date"], "%Y-%m-%d") for r in scored)
    start = dates[0]
    cutoff = start.replace(day=start.day)  # keep type consistent
    from datetime import timedelta
    cutoff = start + timedelta(days=split_days)

    early = [r for r in scored if datetime.strptime(r["date"], "%Y-%m-%d") < cutoff]
    later = [r for r in scored if datetime.strptime(r["date"], "%Y-%m-%d") >= cutoff]
    return early, later


def print_report(label: str, stats: dict) -> None:
    print(f"--- {label} ---")
    if "error" in stats:
        print(stats["error"])
        print()
        return

    print(f"Games scored:        {stats['num_games_scored']}")
    print(f"Accuracy:            {stats['accuracy'] * 100:.1f}%")
    print(f"Brier score:         {stats['brier_score']}  (lower is better, 0.25 = always guessing 50%)")
    print(f"Beat 50/50 baseline: {'yes' if stats['brier_vs_baseline'] > 0 else 'no'} (by {stats['brier_vs_baseline']})")
    print(f"Predicted NRFI rate: {stats['predicted_nrfi_rate'] * 100:.1f}%")
    print(f"Actual NRFI rate:    {stats['actual_nrfi_rate'] * 100:.1f}%")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "backtester/data/backtest_results.json"
    split_days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SPLIT_DAYS

    results = load_results(path)

    overall_stats = evaluate(results)
    print_report("Overall (entire range)", overall_stats)

    early, later = split_by_date(results, split_days)
    print_report(f"Early window (first {split_days} days)", evaluate(early))
    print_report(f"Later window (after day {split_days})", evaluate(later))
