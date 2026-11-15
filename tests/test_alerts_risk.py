import pytest

from floodsense.alerts import build_alert_bn, build_alert_en, to_bangla_digits
from floodsense.config import DISTRICTS
from floodsense.risk import impact_assessment, risk_level


def test_risk_levels():
    assert risk_level(0.0) == "Low"
    assert risk_level(0.3) == "Medium"
    assert risk_level(0.9) == "High"
    with pytest.raises(ValueError):
        risk_level(1.5)


def test_impact_scales_with_probability():
    low = impact_assessment("sunamganj", 0.1)
    high = impact_assessment("sunamganj", 0.9)
    assert high.exposed_population > low.exposed_population
    assert high.exposed_cropland_ha > low.exposed_cropland_ha
    assert high.exposed_population <= DISTRICTS["sunamganj"].population


def test_bangla_digits():
    assert to_bangla_digits(48) == "৪৮"


@pytest.mark.parametrize("level", ["Low", "Medium", "High"])
def test_alerts_contain_district_and_disclaimer(level):
    bn = build_alert_bn("sylhet", level)
    en = build_alert_en("sylhet", level)
    assert "সিলেট" in bn and "৪৮" in bn and "৯৯৯" in bn
    assert "Sylhet" in en and "999" in en


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        build_alert_bn("sylhet", "Extreme")
