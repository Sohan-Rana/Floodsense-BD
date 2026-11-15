"""Synthetic training data generator.

Real, labelled flood histories at district/day resolution are hard to obtain, so the
prototype ships with a physically-motivated synthetic generator. It lets the full pipeline
(training, evaluation, dashboard, alerts) run end to end out of the box.

IMPORTANT: metrics measured on synthetic data say nothing about real-world accuracy.
Swap in real observations (BMD/Open-Meteo rainfall, BWDB/FFWC river levels, Sentinel-1
flood extents) with the same column names to obtain meaningful results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DISTRICTS, District
from .features import FEATURES, TARGET


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _simulate_district(d: District, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    doy = dates.day_of_year.to_numpy()

    # Monsoon seasonality peaking in mid-July.
    monsoon = np.exp(-(((doy - 200) / 55.0) ** 2))

    wet = rng.random(n) < (0.08 + 0.65 * monsoon)
    amount = rng.gamma(0.9, 14.0 * (0.4 + 1.4 * monsoon) * d.rain_factor)
    rain = pd.Series(np.where(wet, amount, 0.0), index=dates)

    rain_72h = rain.rolling(3, min_periods=1).sum().to_numpy()
    rain_7d = rain.rolling(7, min_periods=1).sum().to_numpy()

    # Forecast = next two days of rain with multiplicative forecast error.
    forecast_noise = rng.normal(1.0, 0.2, n).clip(0.3, 1.7)
    forecast = (rain.shift(-1).fillna(0.0) + rain.shift(-2).fillna(0.0)).to_numpy() * forecast_noise

    # Upstream inflow: smoothed, season-dependent positive series.
    upstream = (
        pd.Series(rng.gamma(1.2, 8.0 * (0.2 + 1.5 * monsoon))).ewm(span=10).mean().to_numpy()
    )

    river = 0.35 + 0.0035 * rain_7d + 0.045 * upstream + rng.normal(0.0, 0.03, n)
    river = np.clip(river, 0.1, 1.8)
    river_trend = np.diff(river, prepend=river[0])

    soil = np.clip(0.2 + 0.05 * rain.ewm(span=10).mean().to_numpy(), 0.1, 1.0)

    score = (
        7.0 * (river - 0.95)
        + 0.020 * forecast
        + 0.006 * rain_72h
        + 3.0 * river_trend
        + 1.5 * (soil - 0.6)
        - 0.06 * (d.elevation_m - 10.0)
        - 0.05 * (d.river_dist_km - 5.0)
        + rng.normal(0.0, 0.6, n)
        - 1.5
    )
    flood = (rng.random(n) < _sigmoid(score)).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "district": d.key,
            "rain_24h": rain.to_numpy(),
            "rain_72h": rain_72h,
            "rain_7d": rain_7d,
            "forecast_rain_48h": forecast,
            "river_level_ratio": river,
            "river_trend_24h": river_trend,
            "soil_moisture": soil,
            "elevation_m": d.elevation_m,
            "river_dist_km": d.river_dist_km,
            TARGET: flood,
        }
    )


def generate_synthetic(years: int = 4, seed: int = 42, start: str = "2019-01-01") -> pd.DataFrame:
    """Generate a daily district-level dataset with a binary 48 h flood label."""
    if years < 1:
        raise ValueError("years must be >= 1")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=int(365 * years), freq="D")
    frames = [_simulate_district(d, dates, rng) for d in DISTRICTS.values()]
    df = pd.concat(frames, ignore_index=True)
    return df[["date", "district", *FEATURES, TARGET]]
