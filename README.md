# 🌊 FloodSense BD

**AI-powered flood forecasting and community early warning for Bangladesh.**
*From Prediction to Protection.*

### 🚀 [Live Demo](https://floodsense-bd-dqzkbgynfmmauz9smkfwjk.streamlit.app/)

[![CI](https://github.com/Sohan-Rana/Floodsense-BD/actions/workflows/ci.yml/badge.svg)](https://github.com/Sohan-Rana/Floodsense-BD/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![FloodSense BD dashboard](docs/screenshot-dashboard.png)

## The problem

Bangladesh is among the most flood-prone countries in the world. Official forecasts exist, but they are technical and rarely reach vulnerable households in plain Bangla with clear actions to take.

## What FloodSense BD does

FloodSense BD combines rainfall, river level, soil wetness and terrain data with machine learning to estimate **where** flooding is likely, **when** (next 48 hours) and **how severe** the impact could be. It then turns the result into plain-language **Bangla and English early warnings** for vulnerable communities.

- **Recall-first:** the alert threshold is 0.35, because a missed flood costs more than a false alarm.
- **Honest by design:** the bundled model is trained on synthetic data and is clearly labelled as such (see [limitations](#data-and-honest-limitations)).
- **Open and modular:** data, model, risk and alert layers can each be replaced independently.

## Who it is for

- Vulnerable households and farmers in flood-prone districts
- Local disaster-management committees
- NGOs planning pre-positioning and relief
- Later: telecom operators (SMS delivery) and agricultural insurers

## Results (pilot, synthetic data)

| Metric | Value |
|---|---|
| Precision | _fill from `floodsense train`_ |
| Recall | _fill from `floodsense train`_ |
| ROC-AUC | _fill from `floodsense train`_ |

> These numbers come from synthetic data and do **not** describe real-world accuracy. Validation on real FFWC / Sentinel-1 data is the next milestone.

> **Not a replacement for official forecasts.** FloodSense BD is a decision-support layer that connects scientific forecasting, geospatial risk visualisation, localised impact intelligence and community-friendly alerts. Always follow warnings from the Flood Forecasting and Warning Centre (FFWC) and local authorities.

---

## Features

| Feature | What it does |
|---|---|
| AI flood prediction | Random-forest classifier outputs a 48 h flood probability per district |
| Risk mapping | Interactive map with Low / Medium / High zones |
| Live rainfall | Pulls past and forecast rainfall from the free Open-Meteo API |
| Early warning | Clear alerts in **Bangla and English** with practical actions |
| Impact assessment | Indicative exposed population and cropland per district |
| Risk dashboard | Streamlit app with map, table, feature importance and alerts |
| CLI | `train`, `predict`, `demo` and `districts` commands |

## Architecture

```
Data sources         Weather · River levels · Satellite · Terrain
      │
Processing           Python · Pandas · feature engineering
      │
ML engine            Flood probability · risk scoring · evaluation
      │
FloodSense platform  Interactive map · dashboard · Bangla alerts
      │
Community impact     Preparedness · resilience · informed decisions
```

More detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

```bash
git clone https://github.com/Sohan-Rana/Floodsense-BD.git
cd Floodsense-BD

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[app,dev]"

floodsense train                   # generates data, trains, prints metrics
floodsense demo                    # risk for all pilot districts in 3 scenarios
streamlit run app/streamlit_app.py # open the dashboard
pytest                             # run the tests
```

### Predict for one district

```bash
floodsense predict --district sunamganj \
  --rain-24h 70 --rain-72h 180 --rain-7d 320 --forecast-48h 120 --river-ratio 1.1
```

### Use it as a library

```python
from floodsense.model import ensure_model
from floodsense.predictor import predict

model = ensure_model()
p = predict(model, "sylhet", rain_24h=60, rain_72h=150, rain_7d=300,
            forecast_rain_48h=100, river_level_ratio=1.05, soil_moisture=0.9)
print(p.level, f"{p.probability:.0%}")
print(p.alert_bn)
```

## Project structure

```
Floodsense-BD/
├── app/streamlit_app.py     # dashboard (map, table, alerts)
├── src/floodsense/
│   ├── config.py            # pilot districts, thresholds
│   ├── features.py          # feature definitions
│   ├── synthetic.py         # synthetic data generator
│   ├── open_meteo.py        # live rainfall client
│   ├── model.py             # training, evaluation, persistence
│   ├── risk.py              # risk levels, impact estimate
│   ├── alerts.py            # Bangla / English alerts
│   ├── predictor.py         # inference API
│   └── cli.py               # command line interface
├── tests/                   # pytest suite
├── docs/ARCHITECTURE.md
└── .github/workflows/ci.yml
```

## Pilot districts

Sunamganj, Sylhet, Kurigram, Sirajganj, Gaibandha and Netrokona. Add more in `src/floodsense/config.py`.

## Model and evaluation

- **Model:** `RandomForestClassifier` with balanced class weights.
- **Target:** flood within the next 48 hours (binary), converted to Low / Medium / High by probability thresholds.
- **Split:** time-aware (test period strictly after training period) to avoid leakage.
- **Metrics:** precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix. The alert threshold is intentionally low (0.35) because **recall matters most**: a missed flood is worse than a false alarm.

## Data and honest limitations

- **The bundled model is trained on synthetic data.** Labelled flood histories at district/day resolution are hard to obtain, so a physically-motivated generator lets the whole pipeline run out of the box. **Metrics on synthetic data do not describe real-world accuracy.**
- District attributes (elevation, population, cropland and similar) are rounded illustrative values. Replace them with official figures.
- River level must come from a gauge reading (for example FFWC/BWDB). Without one, a crude rainfall-based fallback is used.
- Impact estimates are screening-level heuristics, not hydrodynamic models.

To get real results, supply a DataFrame with the same columns (see `features.py`) built from, for example:

| Need | Candidate sources |
|---|---|
| Rainfall | Open-Meteo, NASA POWER, GPM IMERG, BMD |
| River levels | FFWC / BWDB |
| Flood extent labels | Sentinel-1 SAR, Global Flood Database |
| Terrain | SRTM / Copernicus DEM |

## Responsible AI

- Positioned as a complement to official FFWC/BWDB warnings, never a replacement.
- Recall-first threshold, with probability calibration and uncertainty bands planned.
- Synthetic data and illustrative district attributes are disclosed everywhere they appear.
- Planned per-district performance checks so no region is systematically under-warned.
- Only public environmental data is used. No personal data is collected.

## Roadmap

- [ ] Train on real historical data with Sentinel-1 flood-extent labels
- [ ] Upazila-level resolution and GeoPandas-based risk polygons
- [ ] 24 h and 72 h horizons
- [ ] SMS / Telegram alert delivery
- [ ] Probability calibration and uncertainty bands
- [ ] Validation with local disaster-management partners

## Contributing

Issues and pull requests are welcome. Please run `pytest` before submitting.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Sohan Rana** (SR), Software Engineering, Daffodil International University · Co-founder and COO, Honqio IT
