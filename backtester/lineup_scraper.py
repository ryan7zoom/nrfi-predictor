"""
Scrapes confirmed starting lineups from MLB.com. Lineups typically post
<1hr before first pitch, so this only helps the "confirmed" prediction tier.
"""

import requests
from bs4 import BeautifulSoup

LINEUPS_URL = "https://www.mlb.com/starting-lineups"


def get_confirmed_lineups() -> list[dict]:
    """Returns [] if lineups aren't posted yet, caller should fall back to
    the projected tier in that case."""
    resp = requests.get(LINEUPS_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    matchups = soup.select("div.starting-lineups__matchup")
    results = []

    for matchup in matchups:
        away_team_el = matchup.select_one(".starting-lineups__team-name--away")
        home_team_el = matchup.select_one(".starting-lineups__team-name--home")

        away_team = away_team_el.get_text(strip=True) if away_team_el else None
        home_team = home_team_el.get_text(strip=True) if home_team_el else None

        # page has duplicate full-name/abbreviated entries, first 9 are real
        away_players = matchup.select(".starting-lineups__team--away .starting-lineups__player")
        home_players = matchup.select(".starting-lineups__team--home .starting-lineups__player")

        away_lineup = [p.get_text(strip=True) for p in away_players[:9]]
        home_lineup = [p.get_text(strip=True) for p in home_players[:9]]

        if away_team and home_team and len(away_lineup) == 9 and len(home_lineup) == 9:
            results.append({
                "away_team": away_team,
                "home_team": home_team,
                "away_lineup": away_lineup,
                "home_lineup": home_lineup,
            })

    return results
