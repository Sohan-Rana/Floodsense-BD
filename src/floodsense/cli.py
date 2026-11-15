"""Command line interface: ``floodsense train | predict | demo | districts``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DISTRICTS
from .model import DEFAULT_MODEL_DIR, ensure_model, save_model, train_model
from .predictor import predict
from .synthetic import generate_synthetic

SCENARIOS = {
    "dry spell": dict(rain_24h=0, rain_72h=2, rain_7d=10, forecast_rain_48h=3, river_level_ratio=0.45, soil_moisture=0.3),
    "normal monsoon": dict(rain_24h=15, rain_72h=45, rain_7d=120, forecast_rain_48h=40, river_level_ratio=0.8, soil_moisture=0.6),
    "heavy rain": dict(rain_24h=70, rain_72h=180, rain_7d=320, forecast_rain_48h=120, river_level_ratio=1.1, river_trend_24h=0.08, soil_moisture=0.9),
}


def _cmd_train(args: argparse.Namespace) -> int:
    df = generate_synthetic(years=args.years, seed=args.seed)
    tm = train_model(df, seed=args.seed)
    path = save_model(tm, args.model_dir)
    print(f"Saved model to {path}")
    print(json.dumps(tm.metrics, indent=2))
    print("Top features:", ", ".join(list(tm.importances)[:4]))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    tm = ensure_model(args.model_dir)
    p = predict(
        tm,
        args.district,
        rain_24h=args.rain_24h,
        rain_72h=args.rain_72h,
        rain_7d=args.rain_7d,
        forecast_rain_48h=args.forecast_48h,
        river_level_ratio=args.river_ratio,
        river_trend_24h=args.river_trend,
        soil_moisture=args.soil_moisture,
    )
    print(f"Flood probability: {p.probability:.0%}  |  Risk level: {p.level}")
    print(f"Indicative exposure: ~{p.exposed_population:,} people, ~{p.exposed_cropland_ha:,} ha cropland")
    print("\n" + p.alert_bn)
    print("\n" + p.alert_en)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    tm = ensure_model(args.model_dir)
    for name, inputs in SCENARIOS.items():
        print(f"\n=== Scenario: {name} ===")
        for key, d in DISTRICTS.items():
            p = predict(tm, key, **inputs)
            print(f"{d.name_en:<10} {p.probability:6.0%}  {p.level}")
    return 0


def _cmd_districts(_: argparse.Namespace) -> int:
    for key, d in DISTRICTS.items():
        print(f"{key:<10} {d.name_en} ({d.name_bn})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floodsense", description="FloodSense BD command line tools")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Where the model is stored")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Generate synthetic data, train and save the model")
    t.add_argument("--years", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="Predict flood risk for one district")
    p.add_argument("--district", required=True, choices=list(DISTRICTS))
    p.add_argument("--rain-24h", type=float, required=True)
    p.add_argument("--rain-72h", type=float, required=True)
    p.add_argument("--rain-7d", type=float, required=True)
    p.add_argument("--forecast-48h", type=float, required=True)
    p.add_argument("--river-ratio", type=float, default=None, help="River level / danger level")
    p.add_argument("--river-trend", type=float, default=0.0)
    p.add_argument("--soil-moisture", type=float, default=0.5)
    p.set_defaults(func=_cmd_predict)

    dm = sub.add_parser("demo", help="Run built-in scenarios for all pilot districts")
    dm.set_defaults(func=_cmd_demo)

    ds = sub.add_parser("districts", help="List pilot districts")
    ds.set_defaults(func=_cmd_districts)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
