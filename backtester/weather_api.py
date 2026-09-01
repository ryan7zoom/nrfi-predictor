"""Open-Meteo weather client. No API key required."""

import time
import requests

LIVE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0


def _get(url: str, params: dict) -> dict:
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError(f"Failed GET {url} after {MAX_RETRIES} attempts") from last_exc


def get_current_weather(lat: float, lon: float) -> dict:
    data = _get(
        LIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        },
    )
    current = data.get("current", {})
    return {
        "temperature_f": current.get("temperature_2m"),
        "wind_speed_mph": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "precipitation": current.get("precipitation"),
    }


def get_historical_weather(lat: float, lon: float, date: str) -> dict:
    """date: 'YYYY-MM-DD'. Archive goes back to 1940."""
    data = _get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": date, "end_date": date,
            "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation",
        },
    )
    return data.get("hourly", {})


def get_weather_at_hour(lat: float, lon: float, date: str, hour: int) -> dict:
    hourly = get_historical_weather(lat, lon, date)
    times = hourly.get("time", [])
    if not times:
        return {"temperature_f": None, "wind_speed_mph": None, "wind_direction_deg": None, "precipitation": None}

    idx = hour if hour < len(times) else 0
    return {
        "temperature_f": _safe_get(hourly.get("temperature_2m"), idx),
        "wind_speed_mph": _safe_get(hourly.get("windspeed_10m"), idx),
        "wind_direction_deg": _safe_get(hourly.get("winddirection_10m"), idx),
        "precipitation": _safe_get(hourly.get("precipitation"), idx),
    }


def _safe_get(lst, idx):
    if lst is None or idx >= len(lst):
        return None
    return lst[idx]
