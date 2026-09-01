"""
Two-tier live prediction: projected (last-completed-game lineup proxy) and
confirmed (scraped lineup, falls back to projected if scraping is empty).
Outputs structured JSON, not a bare number.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlb_api  # noqa: E402
import pipeline  # noqa: E402
import lineup_scraper  # noqa: E402
from model import train_model, predict  # noqa: E402
from backtest import _get_pitcher_hand, _find_last_completed_game  # noqa: E402


def load_training_data(path: str = "backtester/data/backtest_results.json") -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def match_confirmed_lineup(scraped_lineups: list[dict], home_team_name: str, away_team_name: str) -> dict | None:
    for lu in scraped_lineups:
        if lu["home_team"] == home_team_name and lu["away_team"] == away_team_name:
            return lu
    return None


def predict_today(output_path: str = "docs/predictions.json") -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    games = mlb_api.get_schedule(today)

    training_data = load_training_data()
    model = train_model(training_data) if training_data else None

    scraped_lineups = []
    try:
        scraped_lineups = lineup_scraper.get_confirmed_lineups()
    except Exception:
        pass

    results = []

    for g in games:
        game_pk = g["gamePk"]
        home_team_id = g["teams"]["home"]["team"]["id"]
        away_team_id = g["teams"]["away"]["team"]["id"]
        home_team_name = g["teams"]["home"]["team"].get("name", "")
        away_team_name = g["teams"]["away"]["team"].get("name", "")

        home_pitcher = g["teams"]["home"].get("probablePitcher")
        away_pitcher = g["teams"]["away"].get("probablePitcher")
        if not home_pitcher or not away_pitcher:
            continue

        home_pitcher_id = home_pitcher["id"]
        away_pitcher_id = away_pitcher["id"]
        home_pitcher_hand = _get_pitcher_hand(home_pitcher_id)
        away_pitcher_hand = _get_pitcher_hand(away_pitcher_id)

        confirmed = match_confirmed_lineup(scraped_lineups, home_team_name, away_team_name)
        # TODO: scraper returns names only, not player IDs. Falls back to
        # projected tier until name-to-ID matching is wired up.
        tier = "projected"

        home_last_game = _find_last_completed_game(home_team_id, today)
        away_last_game = _find_last_completed_game(away_team_id, today)
        home_top3 = mlb_api.get_starting_batting_order(home_last_game, "home")[:3] if home_last_game else []
        away_top3 = mlb_api.get_starting_batting_order(away_last_game, "away")[:3] if away_last_game else []

        features = pipeline.build_game_features(
            game_pk=game_pk, home_team_id=home_team_id, away_team_id=away_team_id,
            home_pitcher_id=home_pitcher_id, away_pitcher_id=away_pitcher_id,
            home_pitcher_hand=home_pitcher_hand, away_pitcher_hand=away_pitcher_hand,
            home_top3_batter_ids=home_top3, away_top3_batter_ids=away_top3,
            as_of_date=today, is_historical=False,
        )

        prob = predict(model, [features])[0]

        stale = max(features["home_lineup_stale_fraction"], features["away_lineup_stale_fraction"])
        confidence = "low" if stale > 0.3 else ("medium" if stale > 0 else "high")

        results.append({
            "game_pk": game_pk,
            "date": today,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "nrfi_probability": round(prob, 4),
            "tier": tier,
            "confidence": confidence,
            "features": features,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "games": results}, f, indent=2)

    return results


if __name__ == "__main__":
    predict_today()
