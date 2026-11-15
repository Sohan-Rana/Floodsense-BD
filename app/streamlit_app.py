"""FloodSense BD dashboard. Run with: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from floodsense.config import DISTRICTS
from floodsense.model import ensure_model
from floodsense.open_meteo import fetch_weather
from floodsense.predictor import predict

LEVEL_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}

st.set_page_config(page_title="FloodSense BD", page_icon="🌊", layout="wide")


@st.cache_resource(show_spinner="Preparing model...")
def get_model():
    return ensure_model()


model = get_model()

st.title("🌊 FloodSense BD")
st.caption("AI-powered flood risk forecasting and community early warning · From Prediction to Protection")

# ---- Sidebar inputs -------------------------------------------------------------------
st.sidebar.header("Inputs")
mode = st.sidebar.radio("Rainfall source", ["Manual scenario", "Live weather (Open-Meteo)"])

manual = {}
if mode == "Manual scenario":
    manual = dict(
        rain_24h=st.sidebar.slider("Rain, last 24 h (mm)", 0, 250, 40),
        rain_72h=st.sidebar.slider("Rain, last 72 h (mm)", 0, 500, 110),
        rain_7d=st.sidebar.slider("Rain, last 7 days (mm)", 0, 900, 250),
        forecast_rain_48h=st.sidebar.slider("Forecast rain, next 48 h (mm)", 0, 400, 80),
        soil_moisture=st.sidebar.slider("Soil moisture (0-1)", 0.1, 1.0, 0.7),
    )

river_ratio = st.sidebar.slider(
    "River level / danger level", 0.2, 1.6, 0.85, 0.01,
    help="1.0 = danger level. Use a real gauge reading (for example from FFWC/BWDB) when available.",
)
river_trend = st.sidebar.slider("River trend over 24 h", -0.2, 0.3, 0.02, 0.01)

# ---- Predictions ----------------------------------------------------------------------
rows, predictions, errors = [], {}, []
for key, d in DISTRICTS.items():
    try:
        if mode == "Manual scenario":
            inputs = manual
        else:
            w = fetch_weather(key)
            inputs = dict(
                rain_24h=w.rain_24h, rain_72h=w.rain_72h, rain_7d=w.rain_7d,
                forecast_rain_48h=w.forecast_rain_48h, soil_moisture=w.soil_moisture,
            )
        p = predict(model, key, river_level_ratio=river_ratio, river_trend_24h=river_trend, **inputs)
    except Exception as exc:  # network or API problems should not crash the dashboard
        errors.append(f"{d.name_en}: {exc}")
        continue
    predictions[key] = p
    rows.append(
        {
            "District": d.name_en,
            "Risk": p.level,
            "Flood probability": round(p.probability, 3),
            "Exposed people (est.)": p.exposed_population,
            "Exposed cropland ha (est.)": p.exposed_cropland_ha,
        }
    )

for msg in errors:
    st.warning(f"Live weather unavailable: {msg}")
if not predictions:
    st.stop()

# ---- Map + table ----------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Risk map")
    fmap = folium.Map(location=[25.0, 90.6], zoom_start=7, tiles="OpenStreetMap")
    for key, p in predictions.items():
        d = DISTRICTS[key]
        folium.CircleMarker(
            location=[d.lat, d.lon],
            radius=14,
            color=LEVEL_COLORS[p.level],
            fill=True,
            fill_opacity=0.6,
            tooltip=f"{d.name_en} ({d.name_bn}): {p.level} ({p.probability:.0%})",
        ).add_to(fmap)
    st_folium(fmap, height=450, use_container_width=True, returned_objects=[])

with right:
    st.subheader("District overview")
    st.dataframe(pd.DataFrame(rows).sort_values("Flood probability", ascending=False), hide_index=True)
    st.subheader("What drives the model")
    st.bar_chart(pd.Series(model.importances, name="importance"))

# ---- Alerts ---------------------------------------------------------------------------
st.subheader("Early warning")
choice = st.selectbox(
    "District", list(predictions), format_func=lambda k: f"{DISTRICTS[k].name_en} ({DISTRICTS[k].name_bn})"
)
p = predictions[choice]
c1, c2, c3 = st.columns(3)
c1.metric("Flood probability (48 h)", f"{p.probability:.0%}")
c2.metric("Risk level", p.level)
c3.metric("Exposed people (est.)", f"{p.exposed_population:,}")
tab_bn, tab_en = st.tabs(["বাংলা", "English"])
tab_bn.info(p.alert_bn)
tab_en.info(p.alert_en)

st.caption(
    "Prototype for decision support only. It does not replace official FFWC/BWDB forecasts. "
    "The bundled model is trained on synthetic data until real observations are plugged in."
)
