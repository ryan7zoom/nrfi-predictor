"""
Lineup freshness: checks if projected top-3 hitters are actually on the
active roster. Continuous confidence signal, not a hard filter.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlb_api import is_player_on_active_roster  # noqa: E402


def compute_lineup_freshness(player_ids: list[int], team_id: int, as_of_date: str | None = None) -> dict:
    """as_of_date: pass for backtesting, omit for live predictions."""
    if not player_ids:
        return {"num_stale": 0, "stale_fraction": 0.0, "is_fresh": True}

    stale = sum(1 for pid in player_ids if not is_player_on_active_roster(pid, team_id, as_of_date))

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
