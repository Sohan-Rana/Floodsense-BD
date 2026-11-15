import pandas as pd

from floodsense.config import DISTRICTS
from floodsense.features import FEATURES, TARGET
from floodsense.synthetic import generate_synthetic


def test_shape_and_columns():
    df = generate_synthetic(years=2, seed=1)
    assert len(df) == 2 * 365 * len(DISTRICTS)
    assert set(FEATURES + [TARGET, "date", "district"]) <= set(df.columns)
    assert not df[FEATURES].isna().any().any()


def test_prevalence_is_realistic():
    df = generate_synthetic(years=4, seed=42)
    rate = df[TARGET].mean()
    assert 0.02 < rate < 0.35


def test_deterministic_with_seed():
    a = generate_synthetic(years=1, seed=7)
    b = generate_synthetic(years=1, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_floods_concentrate_in_monsoon():
    df = generate_synthetic(years=4, seed=42)
    month = df["date"].dt.month
    monsoon = df[month.isin([6, 7, 8, 9])][TARGET].mean()
    dry = df[month.isin([12, 1, 2])][TARGET].mean()
    assert monsoon > dry
