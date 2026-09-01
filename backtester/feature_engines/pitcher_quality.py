"""
Pitcher quality: season FIP prior + recent 1st-inning ERA delta over a
rolling window. Uses full-game-log fields (confirmed available), not
play-by-play 1st-inning-only parsing (too noisy at that sample size).
"""

from datetime import datetime

FIP_CONSTANT = 3.10


def _innings_to_outs(ip_str: str) -> int:
    if not ip_str:
        return 0
    whole, _, frac = ip_str.partition(".")
    return int(whole) * 3 + (int(frac) if frac else 0)


def _outs_to_innings_float(outs: int) -> float:
    return outs / 3.0


def compute_fip(stat: dict) -> float | None:
    """Single game-log stat dict, raw MLB API shape (inningsPitched as '5.1')."""
    outs = _innings_to_outs(stat.get("inningsPitched", "0.0"))
    return compute_fip_from_totals({
        "homeRuns": stat.get("homeRuns", 0) or 0,
        "baseOnBalls": stat.get("baseOnBalls", 0) or 0,
        "hitByPitch": stat.get("hitByPitch", 0) or 0,
        "strikeOuts": stat.get("strikeOuts", 0) or 0,
        "outs": outs,
    })


def compute_fip_from_totals(totals: dict) -> float | None:
    """Aggregated totals shape (plain numeric 'outs', from aggregate_game_logs)."""
    innings = _outs_to_innings_float(totals.get("outs", 0))
    if innings == 0:
        return None

    hr = totals.get("homeRuns", 0) or 0
    bb = totals.get("baseOnBalls", 0) or 0
    hbp = totals.get("hitByPitch", 0) or 0
    k = totals.get("strikeOuts", 0) or 0

    return ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / innings + FIP_CONSTANT


def aggregate_game_logs(splits: list[dict], as_of_date: str, window: int | None = None) -> dict:
    """Only includes games strictly before as_of_date (no look-ahead)."""
    cutoff = datetime.strptime(as_of_date, "%Y-%m-%d")
    eligible = [s for s in splits if datetime.strptime(s["date"], "%Y-%m-%d") < cutoff]
    eligible.sort(key=lambda s: s["date"])

    if window:
        eligible = eligible[-window:]

    totals = {"homeRuns": 0, "baseOnBalls": 0, "hitByPitch": 0, "strikeOuts": 0,
              "earnedRuns": 0, "outs": 0, "battersFaced": 0}
    for s in eligible:
        stat = s["stat"]
        totals["homeRuns"] += stat.get("homeRuns", 0) or 0
        totals["baseOnBalls"] += stat.get("baseOnBalls", 0) or 0
        totals["hitByPitch"] += stat.get("hitByPitch", 0) or 0
        totals["strikeOuts"] += stat.get("strikeOuts", 0) or 0
        totals["earnedRuns"] += stat.get("earnedRuns", 0) or 0
        totals["outs"] += _innings_to_outs(stat.get("inningsPitched", "0.0"))
        totals["battersFaced"] += stat.get("battersFaced", 0) or 0

    totals["num_starts"] = len(eligible)
    totals["innings"] = _outs_to_innings_float(totals["outs"])
    return totals


def compute_pitcher_quality_features(splits: list[dict], as_of_date: str, recent_window: int = 10) -> dict:
    season_totals = aggregate_game_logs(splits, as_of_date, window=None)
    recent_totals = aggregate_game_logs(splits, as_of_date, window=recent_window)

    season_fip = compute_fip_from_totals(season_totals) if season_totals["innings"] > 0 else None
    recent_fip = compute_fip_from_totals(recent_totals) if recent_totals["innings"] > 0 else None

    delta = None
    if season_fip is not None and recent_fip is not None:
        delta = recent_fip - season_fip

    return {
        "season_fip": season_fip,
        "recent_fip": recent_fip,
        "recent_fip_delta": delta,
        "num_starts_season": season_totals["num_starts"],
    }
