"""Feature definitions shared by training, inference and the dashboard."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

FEATURES = [
    "rain_24h",  # mm, rainfall in the last 24 h
    "rain_72h",  # mm, rainfall in the last 72 h
    "rain_7d",  # mm, rainfall in the last 7 days
    "forecast_rain_48h",  # mm, forecast rainfall for the next 48 h
    "river_level_ratio",  # river level divided by the local danger level (1.0 = danger level)
    "river_trend_24h",  # change of river_level_ratio over 24 h
    "soil_moisture",  # 0-1 relative soil wetness
    "elevation_m",  # static: mean elevation
    "river_dist_km",  # static: distance to the nearest major river
]

TARGET = "flood_next_48h"


def estimate_soil_moisture(daily_rain: Sequence[float]) -> float:
    """Crude soil wetness proxy (0-1) from a daily rainfall history (oldest -> newest)."""
    series = pd.Series(list(daily_rain), dtype=float)
    if series.empty:
        return 0.3
    smoothed = float(series.ewm(span=10).mean().iloc[-1])
    return float(np.clip(0.2 + 0.05 * smoothed, 0.1, 1.0))


def estimate_river_ratio(rain_7d: float) -> float:
    """Very rough fallback when no gauge reading is available.

    Prefer a real water-level reading (for example from FFWC/BWDB) whenever possible.
    """
    return float(np.clip(0.55 + 0.0035 * rain_7d, 0.1, 1.8))


def validate_frame(df: pd.DataFrame, with_target: bool = False) -> None:
    """Raise ValueError if required columns are missing."""
    required = list(FEATURES) + ([TARGET] if with_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
