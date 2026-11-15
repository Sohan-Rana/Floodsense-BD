from floodsense import open_meteo
from floodsense.open_meteo import summarize_hourly


def test_summarize_hourly_windows():
    # 7 past days (168 h) at 1 mm/h, then 48 h forecast at 2 mm/h.
    precip = [1.0] * 168 + [1.0] + [2.0] * 48
    w = summarize_hourly(precip, now_index=168)
    assert w.rain_24h == 24.0
    assert w.rain_72h == 72.0
    assert w.rain_7d == 168.0
    assert w.forecast_rain_48h == 96.0
    assert 0.1 <= w.soil_moisture <= 1.0


def test_none_values_are_treated_as_zero():
    w = summarize_hourly([None] * 200, now_index=100)
    assert w.rain_24h == 0.0


def test_fetch_weather_parses_response(monkeypatch):
    precip = [0.5] * 169 + [1.0] * 72
    times = [f"t{i}" for i in range(len(precip))]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hourly": {"time": times, "precipitation": precip}}

    def fake_get(url, params, timeout):
        assert params["latitude"] and params["longitude"]
        return FakeResponse()

    monkeypatch.setattr(open_meteo.requests, "get", fake_get)
    w = open_meteo.fetch_weather("sylhet")
    assert w.forecast_rain_48h == 48.0
    assert w.rain_24h == 12.0
