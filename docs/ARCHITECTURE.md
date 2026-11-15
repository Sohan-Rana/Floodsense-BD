# FloodSense BD Architecture

## Data flow

1. **Inputs.** Rainfall (observed and forecast), river level ratio, soil wetness and static terrain attributes.
2. **Feature layer** (`features.py`). One shared list of nine features is used by training, inference and the dashboard, so they cannot drift apart.
3. **Model** (`model.py`). A random forest estimates the probability of a flood in the next 48 hours. Training uses a time-aware split.
4. **Risk layer** (`risk.py`). Probability becomes Low / Medium / High and an indicative exposure estimate.
5. **Alert layer** (`alerts.py`). Level and district become a Bangla and English message with practical actions and a pointer to official sources.
6. **Delivery.** CLI, Python API (`predictor.py`) and the Streamlit dashboard.

## Design decisions

- **Recall first.** The binary alert threshold is 0.35 so the system errs towards warning.
- **Honest data story.** `synthetic.py` exists so the pipeline is runnable and testable; it is documented everywhere as non-real data.
- **Replaceable inputs.** Anything that produces a DataFrame with the columns in `features.py` plus `date` and `flood_next_48h` can be trained on without changing other code.
- **Fail soft.** Live weather errors show a warning in the dashboard instead of crashing it.

## Extending

| Goal | Where |
|---|---|
| Add districts | `config.py` (`DISTRICTS`) |
| Add a feature | `features.py` (`FEATURES`), then data generator or loader and `predictor.py` |
| Change thresholds | `config.py` (`LOW_MAX`, `MEDIUM_MAX`, `ALERT_THRESHOLD`) |
| New alert channel | Consume `Prediction.alert_bn` / `alert_en` from `predictor.predict` |
| Real data | Replace `generate_synthetic` with a loader returning the same columns |
