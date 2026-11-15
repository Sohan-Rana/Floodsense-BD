"""Live rainfall inputs from the free Open-Meteo forecast API (no API key required)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

from .config import DISTRICTS
from .features import estimate_soil_moisture

API_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherInputs:
    rain_24h: float
    rain_72h: float
    rain_7d: float
    forecast_rain_48h: float
    soil_moisture: float


def summarize_hourly(precip: list[float | None], now_index: int) -> WeatherInputs:
    """Aggregate an hourly precipitation series (mm) around the current hour index."""
    p = np.array([0.0 if v is None else float(v) for v in precip])
    past = p[: now_index + 1]
    future = p[now_index + 1 : now_index + 49]

    def last(hours: int) -> float:
        return float(past[-hours:].sum())

    n_days = len(past) // 24
    daily = past[len(past) - n_days * 24 :].reshape(n_days, 24).sum(axis=1) if n_days else np.array([])
    return WeatherInputs(
        rain_24h=last(24),
        rain_72h=last(72),
        rain_7d=last(168),
        forecast_rain_48h=float(future.sum()),
        soil_moisture=estimate_soil_moisture(daily.tolist()),
    )


def fetch_weather(district_key: str, timeout: float = 10.0) -> WeatherInputs:
    """Fetch 7 past days + 3 forecast days of hourly rain for a pilot district."""
    d = DISTRICTS[district_key]
    resp = requests.get(
        API_URL,
        params={
            "latitude": d.lat,
            "longitude": d.lon,
            "hourly": "precipitation",
            "past_days": 7,
            "forecast_days": 3,
            "timezone": "Asia/Dhaka",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    times = payload["hourly"]["time"]
    precip = payload["hourly"]["precipitation"]
    # Current hour = last hour that is not after "now"; past_days=7 means 168 past hourly rows.
    now_index = min(7 * 24, len(times) - 1)
    return summarize_hourly(precip, now_index)
