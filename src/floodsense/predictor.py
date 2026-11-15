"""High-level inference API combining model, risk level, impact and alert text."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .alerts import build_alert_bn, build_alert_en
from .config import DISTRICTS, HORIZON_HOURS
from .features import FEATURES, estimate_river_ratio
from .model import TrainedModel
from .risk import impact_assessment, risk_level


@dataclass(frozen=True)
class Prediction:
    district: str
    probability: float
    level: str
    exposed_population: int
    exposed_cropland_ha: int
    alert_bn: str
    alert_en: str


def predict(
    model: TrainedModel,
    district_key: str,
    rain_24h: float,
    rain_72h: float,
    rain_7d: float,
    forecast_rain_48h: float,
    river_level_ratio: float | None = None,
    river_trend_24h: float = 0.0,
    soil_moisture: float = 0.5,
) -> Prediction:
    if district_key not in DISTRICTS:
        raise KeyError(f"Unknown district '{district_key}'. Choose from: {', '.join(DISTRICTS)}")
    d = DISTRICTS[district_key]
    if river_level_ratio is None:
        river_level_ratio = estimate_river_ratio(rain_7d)

    row = pd.DataFrame(
        [
            {
                "rain_24h": rain_24h,
                "rain_72h": rain_72h,
                "rain_7d": rain_7d,
                "forecast_rain_48h": forecast_rain_48h,
                "river_level_ratio": river_level_ratio,
                "river_trend_24h": river_trend_24h,
                "soil_moisture": soil_moisture,
                "elevation_m": d.elevation_m,
                "river_dist_km": d.river_dist_km,
            }
        ]
    )[FEATURES]
    prob = float(model.estimator.predict_proba(row)[0, 1])
    level = risk_level(prob)
    impact = impact_assessment(district_key, prob)
    return Prediction(
        district=district_key,
        probability=prob,
        level=level,
        exposed_population=impact.exposed_population,
        exposed_cropland_ha=impact.exposed_cropland_ha,
        alert_bn=build_alert_bn(district_key, level, HORIZON_HOURS),
        alert_en=build_alert_en(district_key, level, HORIZON_HOURS),
    )
