"""Static configuration: pilot districts and risk thresholds.

District attributes (elevation, river distance, population, cropland, lowland share,
rainfall factor) are rounded, illustrative values for prototyping. Replace them with
official figures (BBS, BWDB, SRTM-derived statistics) before any real-world use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class District:
    key: str
    name_en: str
    name_bn: str
    lat: float
    lon: float
    elevation_m: float
    river_dist_km: float
    population: int
    cropland_ha: int
    lowland_share: float  # share of the district that is low-lying / flood-prone (0-1)
    rain_factor: float  # relative rainfall intensity used by the synthetic generator


DISTRICTS: dict[str, District] = {
    d.key: d
    for d in [
        District("sunamganj", "Sunamganj", "সুনামগঞ্জ", 25.07, 91.40, 8, 3, 2_500_000, 200_000, 0.75, 1.35),
        District("sylhet", "Sylhet", "সিলেট", 24.90, 91.87, 35, 4, 3_500_000, 150_000, 0.45, 1.30),
        District("kurigram", "Kurigram", "কুড়িগ্রাম", 25.81, 89.64, 28, 2, 2_100_000, 170_000, 0.60, 1.00),
        District("sirajganj", "Sirajganj", "সিরাজগঞ্জ", 24.45, 89.70, 14, 2, 3_100_000, 190_000, 0.65, 0.95),
        District("gaibandha", "Gaibandha", "গাইবান্ধা", 25.33, 89.53, 26, 3, 2_400_000, 170_000, 0.55, 0.95),
        District("netrokona", "Netrokona", "নেত্রকোণা", 24.88, 90.73, 12, 5, 2_300_000, 180_000, 0.60, 1.20),
    ]
}

# Forecast horizon the model is trained for.
HORIZON_HOURS = 48

# Probability -> risk level thresholds.
LOW_MAX = 0.25
MEDIUM_MAX = 0.55

# Operating threshold used for binary evaluation (low on purpose: missing a flood is worse
# than a false alarm, so recall is prioritised).
ALERT_THRESHOLD = 0.35

LEVELS = ("Low", "Medium", "High")
