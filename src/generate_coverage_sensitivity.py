"""Stress-test zero-record dates without changing the canonical experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.issues_capstone import (
    CONFIG, FORECAST_CALENDAR_POLICY, REPORTS, _fit_final_model, _metrics,
)


MODELS = (
    "Seasonal naive (same weekday last week)",
    "ETS (weekly additive)",
    "XGBoost autoregression (28-day lags)",
)
OUTPUT = REPORTS / "coverage_sensitivity"


def validate_series(series: pd.Series) -> None:
    if not isinstance(series.index, pd.DatetimeIndex) or series.empty:
        raise ValueError("A nonempty daily DatetimeIndex series is required.")
    expected = pd.date_range(series.index[0], periods=len(series), freq="D")
    if not series.index.equals(expected):
        raise ValueError("Dates must be unique, ordered and consecutive calendar days.")
    if not np.isfinite(series.to_numpy(dtype=float)).all() or (series < 0).any():
        raise ValueError("Daily counts must be finite and nonnegative.")


def causal_gap_scenario(series: pd.Series) -> pd.Series:
    """Replace zero-record days using only earlier observed positive counts.

    This deliberately severe missing-record scenario is not a correction:
    genuine zero-demand dates are also replaced. Never use future observations
    or previously imputed values to choose a replacement.
    """
    validate_series(series)
    scenario = series.astype(float).copy()
    for position in np.flatnonzero(series.to_numpy() == 0):
        prior = series.iloc[max(0, position - 56):position]
        observed = prior[prior > 0]
        weekday = observed[observed.index.dayofweek == series.index[position].dayofweek]
        candidates = weekday.iloc[-4:] if not weekday.empty else observed
        if candidates.empty:
            raise ValueError("A zero-record date has no observed positive count in the prior 56 days.")
        scenario.iloc[position] = float(candidates.median())
    return scenario


def validate_predictions(predictions: pd.DataFrame, series: pd.Series) -> None:
    validate_series(series)
    required = {
        "origin_date", "forecast_date", "forecast_window_days", "horizon_days",
        "split", "model", "actual_tickets", "forecast_tickets",
    }
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError("Canonical predictions are empty or missing required columns.")
    if set(predictions["split"]) != {"validation", "test"}:
        raise ValueError("Both validation and test predictions are required.")
    if not np.isfinite(predictions[["actual_tickets", "forecast_tickets"]].to_numpy()).all():
        raise ValueError("Canonical counts and forecasts must be finite.")
    expected = predictions["origin_date"] + pd.to_timedelta(predictions["horizon_days"], unit="D")
    if not expected.equals(predictions["forecast_date"]):
        raise ValueError("Forecast dates do not follow the recorded origin and lead day.")
    actual = series.reindex(pd.DatetimeIndex(predictions["forecast_date"])).to_numpy()
    if not np.array_equal(actual, predictions["actual_tickets"].to_numpy()):
        raise ValueError("Canonical prediction targets differ from the daily series.")
    keys = ["origin_date", "forecast_window_days", "split", "model"]
    for (_, horizon, _, _), group in predictions.groupby(keys):
        if sorted(group["horizon_days"].tolist()) != list(range(1, int(horizon) + 1)):
            raise ValueError("Each origin/model/window must contain every lead exactly once.")
    if predictions.loc[predictions["split"].eq("validation"), "forecast_date"].max() >= (
        predictions.loc[predictions["split"].eq("test"), "forecast_date"].min()
    ):
        raise ValueError("Validation and test target dates must be disjoint.")


def scenario_predictions(baseline: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    result = baseline.copy()
    seed = int(CONFIG["ticket_volume_forecasting"]["xgboost"]["random_state"])
    for origin, group in result.groupby("origin_date"):
        history = scenario.loc[:origin]
        if history.empty or history.index[-1] != origin:
            raise ValueError("Forecast origin must exist in the training series.")
        horizon = int(group["horizon_days"].max())
        for model, rows in group.groupby("model"):
            artifact = _fit_final_model(history, model, seed)
            forecast = np.asarray(artifact.predict(history, horizon), dtype=float)
            if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
                raise ValueError(f"Invalid scenario forecast for {model} at {origin}.")
            result.loc[rows.index, "forecast_tickets"] = np.maximum(
                0.0, forecast[rows["horizon_days"].to_numpy(dtype=int) - 1],
            )
    result["scenario"] = "causal_missing_record_stress"
    return result


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "split", "forecast_window_days", "model"]
    for (scenario, split, horizon, model), group in predictions.groupby(keys):
        populations = {"observed_positive_dates_only": group[group["actual_tickets"] > 0]}
        if scenario == "canonical_zero_fill":
            populations["zero_assumed_all_dates"] = group
        for population, scored in populations.items():
            if scored.empty:
                raise ValueError(f"No scored targets for {scenario}/{split}/{horizon}/{model}.")
            rows.append({
                "scenario": scenario, "split": split, "forecast_window_days": int(horizon),
                "model": model, "scored_population": population,
                "forecast_target_pairs": len(scored),
                "unique_target_dates": scored["forecast_date"].nunique(),
                "excluded_zero_record_pairs": len(group) - len(scored),
                **_metrics(scored["actual_tickets"], scored["forecast_tickets"]),
            })
    return pd.DataFrame(rows)


def generate(output: Path = OUTPUT) -> dict:
    paths = {
        "daily_counts": REPORTS / "daily_ticket_counts.csv",
        "predictions": REPORTS / "volume_forecast_rolling_predictions.csv",
        "experiment": REPORTS / "volume_forecast_experiment.json",
    }
    manifest = json.loads(paths["experiment"].read_text(encoding="utf-8"))
    if manifest.get("forecast_calendar_policy") != FORECAST_CALENDAR_POLICY:
        raise ValueError("Re-run Notebook 03: canonical forecast calendar policy is stale.")
    daily = pd.read_csv(paths["daily_counts"])
    series = pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        name="tickets",
    )
    baseline = pd.read_csv(paths["predictions"])
    for column in ("origin_date", "forecast_date"):
        baseline[column] = pd.to_datetime(baseline[column], utc=True)
    baseline = baseline[baseline["model"].isin(MODELS)].copy().reset_index(drop=True)
    if set(baseline["model"]) != set(MODELS):
        raise ValueError("All three stress-test model families must have canonical predictions.")
    validate_predictions(baseline, series)
    scenario = causal_gap_scenario(series)
    baseline["scenario"] = "canonical_zero_fill"
    alternative = scenario_predictions(baseline, scenario)
    keep = [
        "scenario", "origin_date", "forecast_date", "forecast_window_days", "horizon_days",
        "split", "model", "actual_tickets", "forecast_tickets",
    ]
    predictions = pd.concat([baseline[keep], alternative[keep]], ignore_index=True)
    metrics = summarize_predictions(predictions)
    comparison = metrics[metrics["scored_population"].eq("observed_positive_dates_only")]
    validation = comparison[comparison["split"].eq("validation")]
    ranks = validation.sort_values("MAE").groupby(
        ["scenario", "forecast_window_days"], as_index=False,
    ).first()
    summary = {
        "forecast_calendar_policy": FORECAST_CALENDAR_POLICY,
        "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        "model_configuration": CONFIG["ticket_volume_forecasting"],
        "zero_record_days": int(series.eq(0).sum()),
        "calendar_days": len(series),
        "recorded_ticket_total": float(series.sum()),
        "stress_scenario_total": float(scenario.sum()),
        "method": (
            "Treat every zero-record day as unknown. Replace training-history zeros with the "
            "median of the last four observed positive same-weekday counts within the prior "
            "56 days; if absent, use prior observed positive counts within 56 days. "
            "No future or imputed values enter replacement statistics. Refit the same ETS, "
            "XGBoost and seasonal-naive families on the canonical rolling origins. "
            "Compare scenarios only on the identical positive-record targets."
        ),
        "best_validation_model_among_three_on_positive_dates": ranks[
            ["scenario", "forecast_window_days", "model", "MAE"]
        ].to_dict(orient="records"),
        "limitations": [
            "Post-hoc sensitivity, not a new independent holdout or model-selection experiment.",
            "Missing dates are not proven missing; real zero demand is deliberately replaced.",
            "Positive-record dates can still be incomplete; their counts are not certified ground truth.",
            "Excluding zero-record targets changes the estimand and can induce selection bias.",
            "No true labels exist for the assumed missing days; imputed targets are never scored.",
            "Ridge is outside this bounded check; no assertion of four-model rank robustness.",
            "Overlapping 30-day windows reuse targets; pair counts are not independent samples.",
            "Intervals are not recalibrated; canonical model selection and API artifact are unchanged.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date_created": series.index, "recorded_tickets": series.to_numpy(),
        "stress_training_tickets": scenario.to_numpy(), "zero_record_date": series.eq(0).to_numpy(),
    }).to_csv(output / "daily_scenarios.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = generate()
    print(json.dumps({
        "output": str(OUTPUT),
        "zero_record_days": summary["zero_record_days"],
        "validation_comparison": summary["best_validation_model_among_three_on_positive_dates"],
    }, indent=2))
