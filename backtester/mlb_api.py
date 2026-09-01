"""
MLB Stats API client. No key required. Endpoints/fields below verified live.
"""

import time
import requests

BASE_V1 = "https://statsapi.mlb.com/api/v1"
BASE_V1_1 = "https://statsapi.mlb.com/api/v1.1"

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0


def _get(url: str, params: dict | None = None) -> dict:
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", BACKOFF_BASE_SECONDS * (2 ** attempt)))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError(f"Failed GET {url} after {MAX_RETRIES} attempts") from last_exc


def get_schedule(date: str) -> list[dict]:
    data = _get(
        f"{BASE_V1}/schedule",
        params={"sportId": 1, "date": date, "hydrate": "team,linescore,probablePitcher"},
    )
    dates = data.get("dates", [])
    if not dates:
        return []
    return dates[0].get("games", [])


def get_pitcher_game_log(player_id: int, season: int) -> list[dict]:
    """Fields confirmed live: strikeOuts, baseOnBalls, hitByPitch, homeRuns,
    inningsPitched, earnedRuns, battersFaced. Empty list if no games (injury etc)."""
    data = _get(
        f"{BASE_V1}/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": season},
    )
    stats = data.get("stats", [])
    if not stats:
        return []
    return stats[0].get("splits", [])


def get_linescore(game_pk: int) -> dict:
    return _get(f"{BASE_V1}/game/{game_pk}/linescore")


def get_first_inning_runs(game_pk: int) -> dict:
    linescore = get_linescore(game_pk)
    innings = linescore.get("innings", [])
    if not innings:
        return {"home": 0, "away": 0}
    first = innings[0]
    return {
        "home": first.get("home", {}).get("runs", 0) or 0,
        "away": first.get("away", {}).get("runs", 0) or 0,
    }


def get_team_hitting_splits(team_id: int, season: int, vs_hand: str) -> dict | None:
    """vs_hand: 'vl' or 'vr'"""
    data = _get(
        f"{BASE_V1}/teams/{team_id}/stats",
        params={"stats": "statSplits", "group": "hitting", "season": season, "sitCodes": vs_hand},
    )
    stats = data.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return None
    return stats[0]["splits"][0]["stat"]


def get_batter_hitting_splits(player_id: int, season: int, vs_hand: str) -> dict | None:
    """vs_hand: 'vl' or 'vr'"""
    data = _get(
        f"{BASE_V1}/people/{player_id}/stats",
        params={"stats": "statSplits", "group": "hitting", "season": season, "sitCodes": vs_hand},
    )
    stats = data.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return None
    return stats[0]["splits"][0]["stat"]


def get_boxscore(game_pk: int) -> dict:
    return _get(f"{BASE_V1}/game/{game_pk}/boxscore")


def get_starting_batting_order(game_pk: int, home_or_away: str) -> list[int]:
    """Returns starter player IDs in order 1-9 (battingOrder ending in '00')."""
    box = get_boxscore(game_pk)
    team_data = box.get("teams", {}).get(home_or_away, {})
    players = team_data.get("players", {})

    starters = []
    for _, pdata in players.items():
        batting_order = pdata.get("battingOrder")
        if batting_order and batting_order.endswith("00"):
            starters.append((int(batting_order), pdata["person"]["id"]))

    starters.sort(key=lambda x: x[0])
    return [pid for _, pid in starters]


def get_active_roster(team_id: int, as_of_date: str | None = None) -> list[dict]:
    """as_of_date confirmed to work for historical lookups (verified via roster diff test)."""
    params = {"rosterType": "active"}
    if as_of_date:
        params["date"] = as_of_date
    data = _get(f"{BASE_V1}/teams/{team_id}/roster", params=params)
    return data.get("roster", [])


def is_player_on_active_roster(player_id: int, team_id: int, as_of_date: str | None = None) -> bool:
    roster = get_active_roster(team_id, as_of_date)
    for entry in roster:
        if entry.get("person", {}).get("id") == player_id and entry.get("status", {}).get("code") == "A":
            return True
    return False


def get_team_venue_info(team_id: int) -> dict:
    """/venues/{id} alone returns null coords, confirmed by test. Use this instead."""
    data = _get(f"{BASE_V1}/teams/{team_id}", params={"hydrate": "venue(location)"})
    teams = data.get("teams", [])
    if not teams:
        return {}
    return teams[0].get("venue", {})


def get_live_feed(game_pk: int) -> dict:
    """Must use v1.1. v1 errors on this endpoint, confirmed by test."""
    return _get(f"{BASE_V1_1}/game/{game_pk}/feed/live")
