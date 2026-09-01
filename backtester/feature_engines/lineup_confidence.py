"""
Lineup freshness: checks if projected top-3 hitters are actually on the
active roster. Continuous confidence signal, not a hard filter.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlb_api  # noqa: E402
import cache  # noqa: E402


def _get_active_roster_cached(team_id: int, as_of_date: str | None) -> list[dict]:
    """The roster fetch itself is the expensive part (one full team roster
    per call). Cache it per (team, date) so 3 batters on the same team in
    the same game share one fetch instead of each triggering their own,
    and so the same team's roster isn't re-fetched from scratch on every
    game they play across a backtest."""
    cache_date = as_of_date if as_of_date else "live"

    def _compute():
        return mlb_api.get_active_roster(team_id, as_of_date)

    return cache.get_or_compute("active_roster", str(team_id), cache_date, _compute)


def compute_lineup_freshness(player_ids: list[int], team_id: int, as_of_date: str | None = None) -> dict:
    """as_of_date: pass for backtesting, omit for live predictions."""
    if not player_ids:
        return {"num_stale": 0, "stale_fraction": 0.0, "is_fresh": True}

    roster = _get_active_roster_cached(team_id, as_of_date)
    active_ids = {
        entry.get("person", {}).get("id")
        for entry in roster
        if entry.get("status", {}).get("code") == "A"
    }

    stale = sum(1 for pid in player_ids if pid not in active_ids)

    return {
        "num_stale": stale,
        "stale_fraction": stale / len(player_ids),
        "is_fresh": stale == 0,
    }


def compute_sample_weight(num_starts: int, target_starts: int = 8, min_weight: float = 0.1) -> float:
    """Ramps from min_weight to 1.0 as starts accumulate, caps at 1.0.
    min_weight keeps early-season games (0 starts) from having zero weight,
    which would otherwise make the whole training batch unusable."""
    return max(min_weight, min(1.0, num_starts / target_starts))
