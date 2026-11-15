import pytest

from floodsense.model import ensure_model, load_model, save_model, time_split, train_model
from floodsense.predictor import predict
from floodsense.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def trained():
    df = generate_synthetic(years=3, seed=3)
    return df, train_model(df, seed=3)


def test_time_split_has_no_leakage(trained):
    df, _ = trained
    train_df, test_df = time_split(df)
    assert train_df["date"].max() < test_df["date"].min()


def test_metrics_are_sensible(trained):
    _, tm = trained
    assert tm.metrics["roc_auc"] > 0.8
    assert tm.metrics["recall"] > 0.5
    assert set(tm.importances) and abs(sum(tm.importances.values()) - 1.0) < 1e-6


def test_save_and_load_roundtrip(trained, tmp_path):
    _, tm = trained
    save_model(tm, tmp_path)
    loaded = load_model(tmp_path)
    assert loaded.metrics == tm.metrics


def test_ensure_model_trains_when_missing(tmp_path):
    tm = ensure_model(tmp_path / "fresh", years=2, seed=5)
    assert (tmp_path / "fresh" / "floodsense_rf.joblib").exists()
    assert tm.metrics["n_train"] > 0


def test_heavy_rain_scores_higher_than_dry(trained):
    _, tm = trained
    dry = predict(tm, "sunamganj", 0, 2, 10, 3, river_level_ratio=0.4, soil_moisture=0.3)
    wet = predict(tm, "sunamganj", 80, 200, 350, 130, river_level_ratio=1.2, river_trend_24h=0.1, soil_moisture=0.95)
    assert wet.probability > dry.probability
    assert wet.level == "High"
    assert dry.level == "Low"


def test_unknown_district_raises(trained):
    _, tm = trained
    with pytest.raises(KeyError):
        predict(tm, "atlantis", 1, 1, 1, 1)
