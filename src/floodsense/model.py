"""Model training, evaluation and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import ALERT_THRESHOLD
from .features import FEATURES, TARGET, validate_frame
from .synthetic import generate_synthetic

DEFAULT_MODEL_DIR = Path("models")
MODEL_FILE = "floodsense_rf.joblib"
METRICS_FILE = "metrics.json"


@dataclass
class TrainedModel:
    estimator: RandomForestClassifier
    metrics: dict
    importances: dict


def time_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so that the test period is strictly after the training period."""
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * (1.0 - test_fraction))]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = ALERT_THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": cm.tolist(),
    }


def train_model(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42) -> TrainedModel:
    """Train a random-forest flood classifier with a time-aware train/test split."""
    validate_frame(df, with_target=True)
    if "date" not in df.columns:
        raise ValueError("DataFrame needs a 'date' column for the time-aware split")

    train_df, test_df = time_split(df, test_fraction)
    est = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    est.fit(train_df[FEATURES], train_df[TARGET])

    proba = est.predict_proba(test_df[FEATURES])[:, 1]
    metrics = evaluate(test_df[TARGET].to_numpy(), proba)
    metrics.update(
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        positive_rate_test=float(test_df[TARGET].mean()),
        data_source="synthetic or user-provided (see README)",
    )
    importances = {
        name: float(score)
        for name, score in sorted(zip(FEATURES, est.feature_importances_), key=lambda kv: -kv[1])
    }
    return TrainedModel(est, metrics, importances)


def save_model(tm: TrainedModel, directory: str | Path = DEFAULT_MODEL_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    joblib.dump(
        {"estimator": tm.estimator, "metrics": tm.metrics, "importances": tm.importances, "features": FEATURES},
        path,
    )
    (directory / METRICS_FILE).write_text(
        json.dumps({"metrics": tm.metrics, "importances": tm.importances}, indent=2), encoding="utf-8"
    )
    return path


def load_model(directory: str | Path = DEFAULT_MODEL_DIR) -> TrainedModel:
    payload = joblib.load(Path(directory) / MODEL_FILE)
    if payload["features"] != FEATURES:
        raise ValueError("Saved model was trained with a different feature set; please retrain.")
    return TrainedModel(payload["estimator"], payload["metrics"], payload["importances"])


def ensure_model(directory: str | Path = DEFAULT_MODEL_DIR, years: int = 4, seed: int = 42) -> TrainedModel:
    """Load the saved model, or train one on synthetic data if none exists yet."""
    try:
        return load_model(directory)
    except (FileNotFoundError, ValueError):
        tm = train_model(generate_synthetic(years=years, seed=seed), seed=seed)
        save_model(tm, directory)
        return tm
