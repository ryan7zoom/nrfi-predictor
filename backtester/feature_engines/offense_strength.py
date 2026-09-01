"""
Top-3 hitters' offense vs. opposing pitcher's handedness. Uses individual
batter splits (confirmed available), not team-level approximation.
"""


def compute_woba_approx(stat: dict) -> float | None:
    """OBP/SLG weighted proxy, not true wOBA. Refine if backtest shows it's off."""
    obp = stat.get("obp")
    slg = stat.get("slg")
    if obp is None or slg is None:
        return None
    try:
        obp_f = float(obp)
        slg_f = float(slg)
    except (TypeError, ValueError):
        return None
    return 0.6 * obp_f + 0.4 * slg_f


def compute_top3_offense_feature(top3_splits: list[dict | None]) -> dict:
    """top3_splits entries may be None (no qualifying split data yet)."""
    wobas = []
    for stat in top3_splits:
        if stat is None:
            continue
        w = compute_woba_approx(stat)
        if w is not None:
            wobas.append(w)

    if not wobas:
        return {"top3_woba_avg": None, "num_batters_with_data": 0}

    return {
        "top3_woba_avg": sum(wobas) / len(wobas),
        "num_batters_with_data": len(wobas),
    }


def compute_team_1st_inning_rate(games_scored: list[int], games_total: int) -> float | None:
    if games_total == 0:
        return None
    return sum(games_scored) / games_total
