"""Orchestrates feature engines into one feature vector per game."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlb_api  # noqa: E402
import weather_api  # noqa: E402
import cache  # noqa: E402
from park_data import get_park_info  # noqa: E402
from feature_engines.pitcher_quality import compute_pitcher_quality_features  # noqa: E402
from feature_engines.offense_strength import compute_top3_offense_feature  # noqa: E402
from feature_engines.park_weather import compute_park_weather_features  # noqa: E402
from feature_engines.lineup_confidence import compute_lineup_freshness, compute_sample_weight  # noqa: E402


def _season_from_date(date_str: str) -> int:
    return int(date_str[:4])


def get_pitcher_features(player_id: int, as_of_date: str) -> dict:
    season = _season_from_date(as_of_date)

    def _compute():
        splits = mlb_api.get_pitcher_game_log(player_id, season)
        return compute_pitcher_quality_features(splits, as_of_date)

    return cache.get_or_compute("pitcher_quality", f"{player_id}_{season}", as_of_date, _compute)


def get_offense_features(team_id: int, top3_batter_ids: list[int], vs_hand: str, as_of_date: str) -> dict:
    season = _season_from_date(as_of_date)

    def _compute():
        splits = []
        for pid in top3_batter_ids:
            try:
                s = mlb_api.get_batter_hitting_splits(pid, season, vs_hand)
            except RuntimeError:
                s = None
            splits.append(s)
        return compute_top3_offense_feature(splits)

    key = f"{team_id}_{'_'.join(map(str, top3_batter_ids))}_{vs_hand}_{season}"
    return cache.get_or_compute("offense_strength", key, as_of_date, _compute)


def get_projected_lineup(team_id: int, last_completed_game_pk: int, home_or_away: str) -> list[int]:
    return mlb_api.get_starting_batting_order(last_completed_game_pk, home_or_away)


def get_weather_features(home_team_id: int, as_of_date: str, is_historical: bool, hour: int = 19) -> dict:
    """is_historical=True uses the archive API (backtest-safe)."""
    info = get_park_info(home_team_id)

    def _compute():
        if is_historical:
            return weather_api.get_weather_at_hour(info.lat, info.lon, as_of_date, hour)
        return weather_api.get_current_weather(info.lat, info.lon)

    key = f"{home_team_id}_{hour}_{'hist' if is_historical else 'live'}"
    weather = cache.get_or_compute("weather", key, as_of_date, _compute)
    return compute_park_weather_features(home_team_id, weather)


def build_game_features(
    game_pk: int,
    home_team_id: int,
    away_team_id: int,
    home_pitcher_id: int,
    away_pitcher_id: int,
    home_pitcher_hand: str,
    away_pitcher_hand: str,
    home_top3_batter_ids: list[int],
    away_top3_batter_ids: list[int],
    as_of_date: str,
    is_historical: bool,
) -> dict:
    """home/away_pitcher_hand is the OPPOSING pitcher's hand ('vl'/'vr'),
    used to pick which split applies to that team's batters."""
    home_pitcher_feat = get_pitcher_features(home_pitcher_id, as_of_date)
    away_pitcher_feat = get_pitcher_features(away_pitcher_id, as_of_date)

    home_offense_feat = get_offense_features(home_team_id, home_top3_batter_ids, away_pitcher_hand, as_of_date)
    away_offense_feat = get_offense_features(away_team_id, away_top3_batter_ids, home_pitcher_hand, as_of_date)

    park_weather_feat = get_weather_features(home_team_id, as_of_date, is_historical)

    home_lineup_conf = compute_lineup_freshness(home_top3_batter_ids, home_team_id, as_of_date if is_historical else None)
    away_lineup_conf = compute_lineup_freshness(away_top3_batter_ids, away_team_id, as_of_date if is_historical else None)

    home_weight = compute_sample_weight(home_pitcher_feat["num_starts_season"])
    away_weight = compute_sample_weight(away_pitcher_feat["num_starts_season"])
    sample_weight = min(home_weight, away_weight)

    return {
        "game_pk": game_pk,
        "as_of_date": as_of_date,
        "home_pitcher_season_fip": home_pitcher_feat["season_fip"],
        "home_pitcher_recent_fip_delta": home_pitcher_feat["recent_fip_delta"],
        "away_pitcher_season_fip": away_pitcher_feat["season_fip"],
        "away_pitcher_recent_fip_delta": away_pitcher_feat["recent_fip_delta"],
        "home_top3_woba": home_offense_feat["top3_woba_avg"],
        "away_top3_woba": away_offense_feat["top3_woba_avg"],
        "park_factor": park_weather_feat["park_factor"],
        "temperature_f": park_weather_feat["temperature_f"],
        "wind_effect": park_weather_feat["wind_effect"],
        "park_factor_x_wind": park_weather_feat["park_factor_x_wind"],
        "is_roofed": park_weather_feat["is_roofed"],
        "home_lineup_stale_fraction": home_lineup_conf["stale_fraction"],
        "away_lineup_stale_fraction": away_lineup_conf["stale_fraction"],
        "sample_weight": sample_weight,
    }
