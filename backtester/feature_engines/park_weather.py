"""Park factor + weather feature engine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from park_data import get_park_info, wind_effect  # noqa: E402


def compute_park_weather_features(home_team_id: int, weather: dict) -> dict:
    """Roofed stadiums always get wind_effect=0 (no live roof-status field exists)."""
    info = get_park_info(home_team_id)

    wind_dir = weather.get("wind_direction_deg")
    wind_speed = weather.get("wind_speed_mph") or 0.0

    if info.is_roofed or wind_dir is None:
        wind_effect_value = 0.0
    else:
        wind_effect_value = wind_effect(wind_dir, wind_speed, home_team_id)

    return {
        "park_factor": info.park_factor,
        "temperature_f": weather.get("temperature_f"),
        "wind_speed_mph": wind_speed,
        "wind_effect": wind_effect_value,
        "park_factor_x_wind": info.park_factor * wind_effect_value,
        "is_roofed": info.is_roofed,
        "is_irregular_wind_park": info.is_irregular_wind,
    }
