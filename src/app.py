"""Minimal deployment POC for the persisted volume-forecast model.

Serves `models/final_volume_forecast_model.joblib` (produced by notebook 03's
`persist_final_model` and re-registered by notebook 07's `log_experiment_to_mlflow`)
behind a small FastAPI surface: `/health`, `/model/metadata`, and `/forecast`.

This is a proof-of-concept for how the artifact *could* be served, not a
production system: there is no authentication, no live/current ticket feed, no
autoscaling, and no monitoring integration beyond the historical-backtest
metrics already recorded in `models/final_volume_forecast_model_metadata.json`.
Every response repeats that this is historical, to avoid the endpoint being
mistaken for a real-time forecasting service.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_volume_forecast_model.joblib"
METADATA_PATH = MODELS_DIR / "final_volume_forecast_model_metadata.json"

try:
    from src.issues_capstone import REPORTS as _REPORTS
except ImportError:  # pragma: no cover - fallback if imported outside package context
    _REPORTS = ROOT / "output" / "issue_helpdesk" / "volume_forecast"

DAILY_COUNTS_PATH = _REPORTS / "daily_ticket_counts.csv"

app = FastAPI(
    title="Help Desk Ticket Volume Forecast \u2014 Deployment POC",
    version="1.0.0",
    description=(
        "Serves the notebook 03 validation-selected, full-history-refit forecast "
        "model. Historical backtest only: source data ends March 2023, and no "
        "current ticket feed is connected."
    ),
)


def _load_metadata() -> dict:
    if not METADATA_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "No persisted model found. Run notebook 03 "
                "(src.issues_capstone.run_volume_forecast) to create "
                f"{MODEL_PATH.relative_to(ROOT)}."
            ),
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_artifact():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "No persisted model found. Run notebook 03 "
                "(src.issues_capstone.run_volume_forecast) to create "
                f"{MODEL_PATH.relative_to(ROOT)}."
            ),
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_history() -> pd.Series:
    if not DAILY_COUNTS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"No {DAILY_COUNTS_PATH.relative_to(ROOT)} found. Run notebook 01/03 first.",
        )
    daily = pd.read_csv(DAILY_COUNTS_PATH, parse_dates=["date_created"])
    return pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        name="tickets",
    )


@app.get("/health")
def health():
    metadata = _load_metadata() if METADATA_PATH.is_file() else None
    return {
        "status": "ok" if metadata else "no_model_persisted",
        "service": "help-desk-ticket-volume-forecast-poc",
        "model_name": metadata["model_name"] if metadata else None,
        "note": "Historical-backtest artifact only; not a live production endpoint.",
    }


@app.get("/model/metadata")
def model_metadata():
    return _load_metadata()


@app.get("/forecast")
def forecast(
    horizon_days: int = Query(
        default=30,
        ge=1,
        le=90,
        description="Number of future daily counts to forecast, starting the day after the training history ends.",
    )
):
    artifact = _load_artifact()
    history = _load_history()
    metadata = _load_metadata()
    predicted = artifact.predict(history, horizon_days)
    start_date = pd.Timestamp(history.index.max()) + pd.Timedelta(days=1)
    dates = pd.date_range(start_date, periods=horizon_days, freq="D", tz="UTC")
    return {
        "model_name": artifact.model_name,
        "horizon_days": horizon_days,
        "history_ends": history.index.max().date().isoformat(),
        "trained_on_date_end": metadata["trained_on_date_end"],
        "forecast": [
            {"date": date.date().isoformat(), "forecast_tickets": round(float(value), 2)}
            for date, value in zip(dates, predicted)
        ],
        "limitations": metadata["limitations"],
        "caveat": (
            "This forecast is generated from the same historical series the model was "
            "trained on (ending March 2023), not a live/current ticket feed. It "
            "demonstrates that the persisted artifact is servable, not that it is "
            "validated for current operations."
        ),
    }
