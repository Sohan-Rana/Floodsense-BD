"""Risk levels and a simple community / agricultural exposure estimate."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DISTRICTS, LOW_MAX, MEDIUM_MAX


def risk_level(probability: float) -> str:
    """Map a flood probability (0-1) to Low / Medium / High."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if probability < LOW_MAX:
        return "Low"
    if probability < MEDIUM_MAX:
        return "Medium"
    return "High"


@dataclass(frozen=True)
class Impact:
    exposed_population: int
    exposed_cropland_ha: int


def impact_assessment(district_key: str, probability: float) -> Impact:
    """Indicative exposure = district totals x lowland share x scaled flood probability.

    This is a screening-level heuristic, not a hydrodynamic impact model.
    """
    d = DISTRICTS[district_key]
    exposure = d.lowland_share * min(1.0, probability * 1.2)
    return Impact(
        exposed_population=int(round(d.population * exposure, -3)),
        exposed_cropland_ha=int(round(d.cropland_ha * exposure, -2)),
    )
