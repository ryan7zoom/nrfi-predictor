"""
Expanding-window walk-forward backtest.

Two lineup modes: "perfect" (real batting order, needs game already played)
vs "realistic" (last-completed-game proxy). Comparing the two shows the
accuracy cost of lineup timing.
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlb_api  # noqa: E402
import pipeline  # noqa: E402
import cache  # noqa: E402
from model import train_model, predict  # noqa: E402


def daterange(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def _get_pitcher_hand(player_id: int) -> str:
    """Returns 'vl' if pitcher throws left, 'vr' otherwise (this is the
    hand code the opposing batters' splits should use). Throwing hand
    never changes, so this is cached permanently (as_of_date='static')
    instead of being re-fetched from the API on every call."""

    def _compute():
        data = mlb_api._get(f"{mlb_api.BASE_V1}/people/{player_id}")
        people = data.get("people", [])
        if not people:
            return "vr"
        throws = people[0].get("pitchHand", {}).get("code", "R")
        return "vl" if throws == "L" else "vr"

    return cache.get_or_compute("pitcher_hand", str(player_id), "static", _compute)


_schedule_cache_by_date = {}


def _get_schedule_cached(date: str) -> list[dict]:
    """In-memory cache for the current process, avoids re-fetching the same
    date's schedule repeatedly when many teams' lookback windows overlap."""
    if date not in _schedule_cache_by_date:
        _schedule_cache_by_date[date] = mlb_api.get_schedule(date)
    return _schedule_cache_by_date[date]


def _find_last_completed_game(team_id: int, before_date: str) -> int | None:
    end = datetime.strptime(before_date, "%Y-%m-%d")
    for days_back in range(1, 15):
        check_date = (end - timedelta(days=days_back)).strftime("%Y-%m-%d")
        games = _get_schedule_cached(check_date)
        for g in games:
            if g["teams"]["home"]["team"]["id"] == team_id or g["teams"]["away"]["team"]["id"] == team_id:
                if g.get("status", {}).get("abstractGameState") == "Final":
                    return g["gamePk"]
    return None


def build_features_for_date(eval_date: str, lineup_mode: str = "realistic") -> list[dict]:
    games = mlb_api.get_schedule(eval_date)
    results = []

    for g in games:
        game_pk = g["gamePk"]
        home_team_id = g["teams"]["home"]["team"]["id"]
        away_team_id = g["teams"]["away"]["team"]["id"]

        home_pitcher = g["teams"]["home"].get("probablePitcher")
        away_pitcher = g["teams"]["away"].get("probablePitcher")
        if not home_pitcher or not away_pitcher:
            continue

        home_pitcher_id = home_pitcher["id"]
        away_pitcher_id = away_pitcher["id"]
        home_pitcher_hand = _get_pitcher_hand(home_pitcher_id)
        away_pitcher_hand = _get_pitcher_hand(away_pitcher_id)

        if lineup_mode == "perfect":
            home_top3 = mlb_api.get_starting_batting_order(game_pk, "home")[:3]
            away_top3 = mlb_api.get_starting_batting_order(game_pk, "away")[:3]
        else:
            home_last_game = _find_last_completed_game(home_team_id, eval_date)
            away_last_game = _find_last_completed_game(away_team_id, eval_date)
            home_top3 = mlb_api.get_starting_batting_order(home_last_game, "home")[:3] if home_last_game else []
            away_top3 = mlb_api.get_starting_batting_order(away_last_game, "away")[:3] if away_last_game else []

        features = pipeline.build_game_features(
            game_pk=game_pk, home_team_id=home_team_id, away_team_id=away_team_id,
            home_pitcher_id=home_pitcher_id, away_pitcher_id=away_pitcher_id,
            home_pitcher_hand=home_pitcher_hand, away_pitcher_hand=away_pitcher_hand,
            home_top3_batter_ids=home_top3, away_top3_batter_ids=away_top3,
            as_of_date=eval_date, is_historical=True,
        )

        if g.get("status", {}).get("abstractGameState") == "Final":
            runs = mlb_api.get_first_inning_runs(game_pk)
            features["nrfi_label"] = 1 if (runs["home"] == 0 and runs["away"] == 0) else 0

        results.append(features)

    return results


def run_backtest(start_date: str, end_date: str, lineup_mode: str = "realistic", output_path: str = None) -> list[dict]:
    all_features = []
    all_predictions = []

    for eval_date in daterange(start_date, end_date):
        todays_features = build_features_for_date(eval_date, lineup_mode=lineup_mode)
        if not todays_features:
            continue

        if len(all_features) >= 50:
            model = train_model(all_features)
            preds = predict(model, todays_features)
        else:
            preds = [0.5] * len(todays_features)

        for feat, pred in zip(todays_features, preds):
            all_predictions.append({
                "game_pk": feat["game_pk"],
                "date": eval_date,
                "predicted_nrfi_prob": pred,
                "actual_label": feat.get("nrfi_label"),
            })

        all_features.extend(todays_features)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(all_predictions, f, indent=2)

    return all_predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--lineup-mode", default="realistic", choices=["realistic", "perfect"])
    parser.add_argument("--output", default="backtester/data/backtest_results.json")
    args = parser.parse_args()

    results = run_backtest(args.start, args.end, args.lineup_mode, args.output)
    print(f"Backtest complete: {len(results)} predictions written to {args.output}")
