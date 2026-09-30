"""Read-only diagnostics for the persisted primary ETS model and saved backtest.

Run from the project root: .\\.venv\\Scripts\\python.exe -m src.generate_ets_diagnostics
Only output\\issue_helpdesk\\volume_forecast\\ets_diagnostics is written.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import issues_capstone as core


MODEL_NAME = "ETS (weekly additive)"
OUTPUT = core.REPORTS / "ets_diagnostics"


def validate_history(history: pd.Series) -> None:
    """Require the complete daily, nonnegative history used by this count model."""
    if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("History must be a Series with a DatetimeIndex.")
    if len(history) < 28:
        raise ValueError("History must contain at least 28 daily observations.")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("History must contain finite nonnegative counts.")
    expected = pd.date_range(history.index[0], periods=len(history), freq="D")
    if not history.index.equals(expected):
        raise ValueError("History must be ordered, unique and uninterrupted daily observations.")


def component_evidence(history: pd.Series, fitted, horizon: int = 30):
    """Export actual Holt-Winters states, not an independently fitted decomposition."""
    validate_history(history)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    model = fitted.model
    if (model.trend, model.seasonal, model.seasonal_periods, model.damped_trend) != (
        "add", "add", 7, False,
    ):
        raise ValueError("Diagnostics require the canonical undamped additive weekly ETS.")
    if fitted.params["use_boxcox"] or fitted.params["remove_bias"]:
        raise ValueError("Transformed or bias-adjusted ETS is not the canonical specification.")
    endog = np.asarray(model.endog, dtype=float).reshape(-1)
    if endog.shape != history.shape or not np.array_equal(endog, history.to_numpy(dtype=float)):
        raise ValueError("Saved ETS observations do not match daily history.")
    if not pd.DatetimeIndex(fitted.fittedvalues.index).equals(history.index):
        raise ValueError("Saved ETS dates do not match daily history.")
    level = np.asarray(fitted.level, dtype=float)
    trend = np.asarray(fitted.trend, dtype=float)
    season = np.asarray(fitted.season, dtype=float)
    previous_level = np.r_[fitted.params["initial_level"], level[:-1]]
    previous_trend = np.r_[fitted.params["initial_trend"], trend[:-1]]
    previous_season = np.r_[fitted.params["initial_seasons"], season[:-7]]
    reconstructed = previous_level + previous_trend + previous_season
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    if not np.allclose(reconstructed, predicted, atol=1e-7, rtol=1e-7):
        raise ValueError("Fitted component reconstruction does not match saved ETS.")
    components = pd.DataFrame({
        "date": history.index,
        "actual_tickets": history.to_numpy(dtype=float),
        "fitted_tickets_raw": predicted,
        "level_after_observation": level,
        "trend_after_observation": trend,
        "seasonal_after_observation": season,
        "prediction_previous_level": previous_level,
        "prediction_previous_trend": previous_trend,
        "prediction_seasonal_lag7": previous_season,
        "residual_actual_minus_fitted": history.to_numpy(dtype=float) - predicted,
    })
    leads = np.arange(1, horizon + 1)
    # Explain the cycle actually used by the installed library. Its forecast
    # boundary can reuse an older seasonal state rather than the final update.
    effective_week = np.asarray(fitted.forecast(7)) - level[-1] - np.arange(1, 8) * trend[-1]
    future = pd.DataFrame({
        "date": core.forecast_dates(history, horizon),
        "lead_day": leads,
        "terminal_level": level[-1],
        "trend_contribution": leads * trend[-1],
        "seasonal_contribution": np.resize(effective_week, horizon),
        "last_updated_seasonal_state": np.resize(season[-7:], horizon),
    })
    future["forecast_seasonal_minus_last_updated"] = (
        future["seasonal_contribution"] - future["last_updated_seasonal_state"]
    )
    future["forecast_tickets_raw"] = (
        future["terminal_level"] + future["trend_contribution"] + future["seasonal_contribution"]
    )
    if not np.allclose(future["forecast_tickets_raw"], fitted.forecast(horizon), atol=1e-7):
        raise ValueError("Forecast component reconstruction does not match saved ETS.")
    future["forecast_tickets_clipped"] = future["forecast_tickets_raw"].clip(lower=0)
    return components, future


def residual_evidence(residuals, max_lag: int = 28):
    """Descriptive ACF and unadjusted Ljung-Box, not an independence certificate."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Residuals must be a finite one-dimensional array with at least 3 values.")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not 1 <= max_lag < len(values):
        raise ValueError("Maximum lag must be a positive integer smaller than the sample.")
    summary = {
        "observations": len(values),
        "mean_actual_minus_fitted": float(values.mean()),
        "MAE": float(np.abs(values).mean()),
        "RMSE": float(np.sqrt(np.mean(values ** 2))),
        "sample_standard_deviation": float(values.std(ddof=1)),
    }
    lags = np.arange(1, max_lag + 1)
    if np.ptp(values) == 0:
        raise ValueError("Autocorrelation is undefined for constant residuals.")
    correlations = acf(values, nlags=max_lag, fft=True)[1:]
    tests = acorr_ljungbox(values, lags=lags, model_df=0, return_df=True)
    table = pd.DataFrame({
        "lag_days": lags,
        "acf": correlations,
        "ljung_box_statistic": tests["lb_stat"].to_numpy(),
        "ljung_box_pvalue_unadjusted_model_df0": tests["lb_pvalue"].to_numpy(),
        "approximate_white_noise_95_bound": 1.96 / np.sqrt(len(values)),
    })
    return summary, table


def test_error_evidence(predictions: pd.DataFrame, horizon: int = 30):
    """Preserve canonical origin/lead weighting and nonnegative forecast clipping."""
    required = {
        "model", "split", "forecast_window_days", "origin_date", "forecast_date",
        "horizon_days", "actual_tickets", "forecast_tickets", "forecast_tickets_raw",
    }
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions.columns))}")
    if horizon != 30:
        raise ValueError("This diagnostic is restricted to the primary 30-day horizon.")
    frame = predictions.loc[
        predictions["model"].eq(MODEL_NAME) & predictions["split"].eq("test")
        & predictions["forecast_window_days"].eq(horizon)
    ].copy()
    if frame.empty:
        raise ValueError("No primary ETS test forecasts found.")
    for column in ("origin_date", "forecast_date"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if frame[column].isna().any():
            raise ValueError("Test dates must not be missing.")
    numeric = frame[["actual_tickets", "forecast_tickets", "forecast_tickets_raw", "horizon_days"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Test forecasts, actuals and leads must be finite.")
    if (frame[["actual_tickets", "forecast_tickets"]] < 0).any().any():
        raise ValueError("Actual counts and clipped forecasts must be nonnegative.")
    if not np.allclose(frame["forecast_tickets"], frame["forecast_tickets_raw"].clip(lower=0)):
        raise ValueError("Saved forecasts do not follow canonical nonnegative clipping.")
    if frame.duplicated(["origin_date", "horizon_days"]).any():
        raise ValueError("Duplicate origin/lead pairs in ETS test forecasts.")
    for _, group in frame.groupby("origin_date"):
        if sorted(group["horizon_days"].tolist()) != list(range(1, horizon + 1)):
            raise ValueError("Every test origin must contain all 30 forecast leads.")
    dates = frame["origin_date"] + pd.to_timedelta(frame["horizon_days"], unit="D")
    if not dates.equals(frame["forecast_date"]):
        raise ValueError("Test forecast dates must follow each origin by the stated lead.")
    frame = frame.sort_values(["origin_date", "horizon_days"])
    frame["error_actual_minus_forecast"] = frame["actual_tickets"] - frame["forecast_tickets"]
    frame["absolute_error"] = frame["error_actual_minus_forecast"].abs()
    summary = {
        "origin_lead_pairs": len(frame),
        "forecast_origins": int(frame["origin_date"].nunique()),
        "unique_target_dates": int(frame["forecast_date"].nunique()),
        "target_date_start": frame["forecast_date"].min().date().isoformat(),
        "target_date_end": frame["forecast_date"].max().date().isoformat(),
        "negative_raw_forecasts_clipped": int(frame["forecast_tickets_raw"].lt(0).sum()),
        **core._metrics(frame["actual_tickets"], frame["forecast_tickets"]),
    }
    by_lead = pd.DataFrame([
        {"lead_day": int(lead), "origin_lead_pairs": len(group),
         **core._metrics(group["actual_tickets"], group["forecast_tickets"])}
        for lead, group in frame.groupby("horizon_days")
    ])
    return frame, summary, by_lead


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _figures(components, future, correlations, errors, by_lead, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), layout="constrained")
        recent = components.tail(120)
        axes[0].plot(recent["date"], recent["actual_tickets"], color="0.6", label="Observed")
        axes[0].plot(recent["date"], recent["fitted_tickets_raw"], label="Fitted (in-sample)")
        axes[0].plot(future["date"], future["forecast_tickets_clipped"], color="tab:red",
                     label="30-day historical projection (clipped)")
        axes[0].set_title("Primary ETS: observed/fitted history and projection after 14 March 2023")
        axes[0].set_ylabel("Tickets/day")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[1].plot(components["date"], components["level_after_observation"], label="Level",
                     color="tab:blue")
        axes[1].set_ylabel("Level (tickets/day)")
        other = axes[1].twinx()
        other.plot(components["date"], components["trend_after_observation"], color="tab:orange")
        other.set_ylabel("Daily trend increment", color="tab:orange")
        axes[1].set_title("Full-history post-update states (raw ETS parameterization)")
        week = future.head(7)
        axes[2].bar(week["date"].dt.strftime("%a"), week["seasonal_contribution"], color="tab:green")
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_ylabel("Additive tickets/day")
        axes[2].set_title("Effective seven-day seasonal cycle used by the saved model's forecast")
        fig.savefig(output / "ets_components.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        axes[0, 0].plot(components["date"], components["residual_actual_minus_fitted"],
                        linewidth=0.6)
        axes[0, 0].axhline(0, color="black", linewidth=0.8)
        axes[0, 0].set_title("In-sample residuals: actual minus fitted (full history)")
        axes[0, 0].set_ylabel("Tickets/day error")
        axes[0, 1].bar(correlations["lag_days"], correlations["acf"])
        bound = correlations["approximate_white_noise_95_bound"].iloc[0]
        for sign in (-1, 1):
            axes[0, 1].axhline(sign * bound, color="tab:red", linestyle="--")
        axes[0, 1].set_title("In-sample ACF; dashed bounds are heuristic")
        axes[0, 1].set_xlabel("Lag (days)")
        axes[1, 0].hist(errors["error_actual_minus_forecast"], bins=40, color="tab:orange")
        axes[1, 0].axvline(0, color="black", linewidth=0.8)
        axes[1, 0].set_title("Held-out rolling test errors (overlapping windows)")
        axes[1, 0].set_xlabel("Actual minus clipped forecast")
        axes[1, 0].set_ylabel("Origin/lead pairs, not independent days")
        axes[1, 1].plot(by_lead["lead_day"], by_lead["MAE"], label="Test MAE")
        axes[1, 1].plot(by_lead["lead_day"], by_lead["signed_bias_actual_minus_forecast"],
                        label="Test signed bias")
        axes[1, 1].axhline(0, color="black", linewidth=0.8)
        axes[1, 1].set_title("Held-out errors by lead within primary 30-day windows")
        axes[1, 1].set_xlabel("Lead day")
        axes[1, 1].set_ylabel("Tickets/day error")
        axes[1, 1].legend()
        fig.savefig(output / "ets_residual_diagnostics.png", dpi=160)
        plt.close(fig)


def generate() -> dict:
    """Load existing artifacts only; never fit or persist a model."""
    import joblib
    import statsmodels

    paths = {
        "model": core.MODELS_DIR / "final_volume_forecast_model.joblib",
        "metadata": core.MODELS_DIR / "final_volume_forecast_model_metadata.json",
        "daily": core.REPORTS / "daily_ticket_counts.csv",
        "predictions": core.REPORTS / "volume_forecast_rolling_predictions.csv",
        "experiment": core.REPORTS / "volume_forecast_experiment.json",
        "config": core.CONFIG_PATH,
    }
    hashes = {name: _hash(path) for name, path in paths.items()}
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    experiment = json.loads(paths["experiment"].read_text(encoding="utf-8"))
    cfg = core.CONFIG["ticket_volume_forecasting"]
    if (
        metadata["model_name"] != MODEL_NAME or metadata["primary_horizon_days"] != 30
        or cfg["intended_planning_horizon_days"] != 30
        or cfg["ets"] != {"trend": "additive", "seasonal": "additive"}
        or cfg["weekly_seasonal_period_days"] != 7
        or experiment["selected_model"] != MODEL_NAME
        or any(record.get("forecast_calendar_policy") != core.FORECAST_CALENDAR_POLICY
               for record in (metadata, experiment))
    ):
        raise ValueError("Saved artifacts/config are not the current primary 30-day ETS experiment.")
    daily = pd.read_csv(paths["daily"])
    history = pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)), name="tickets",
    )
    validate_history(history)
    if (
        metadata["trained_on_rows"] != len(history)
        or metadata["trained_on_date_start"] != history.index[0].date().isoformat()
        or metadata["trained_on_date_end"] != history.index[-1].date().isoformat()
    ):
        raise ValueError("Saved metadata and daily history disagree.")
    # Older notebooks pickle the same core class under its top-level module name.
    sys.modules.setdefault("issues_capstone", core)
    artifact = joblib.load(paths["model"])
    if artifact.model_name != MODEL_NAME:
        raise ValueError("Persisted model is not ETS.")
    fitted = artifact.fitted["ets"]
    components, future = component_evidence(history, fitted)
    residual_summary, correlations = residual_evidence(components["residual_actual_minus_fitted"])
    errors, test_summary, by_lead = test_error_evidence(pd.read_csv(paths["predictions"]))
    actual = history.reindex(pd.DatetimeIndex(errors["forecast_date"])).to_numpy()
    if not np.array_equal(actual, errors["actual_tickets"].to_numpy()):
        raise ValueError("Saved test actual counts do not match daily history.")
    test_start = history.index[-int(cfg["final_test_days"])]
    if (errors["forecast_date"] < test_start).any():
        raise ValueError("Saved test predictions are outside the configured historical holdout.")
    optimizer = fitted.mle_retvals
    convergence = {key: optimizer.get(key) for key in ("success", "status", "message", "nit", "nfev")}
    limitations = [
        "The persisted model was refit on ALL history, including dates in the historical test. "
        "Its fitted residuals and states are in-sample explanation, not held-out performance.",
        "Held-out evidence comes ONLY from saved expanding-window test forecasts made with "
        "history available at each origin. The 30-day model was selected by validation MAE, not test MAE.",
        "Test statistics weight every origin/lead pair, matching canonical metrics. Overlapping "
        "30-day windows at seven-day strides reuse dates; errors are not independent observations.",
        "ACF bands +/-1.96/sqrt(n) and Ljung-Box p-values (model_df=0, no fitted-parameter "
        "or multiple-testing adjustment) are descriptive screening tools. "
        "Neither small mean error nor non-rejection establishes residual whiteness.",
        "Optimizer success refers only to the saved final fit, not all historical rolling fits; "
        "no historical optimizer logs are available here. Boundary parameters can limit adaptation.",
        "Level and seasonal offsets use the saved raw parameterization; seasonal offsets need "
        "not average zero. Their sum is interpretable, but their separate baselines are not unique.",
        "The final 30-day projection is after the March 2023 data endpoint, not a current forecast "
        "or a held-out test. No new uncertainty intervals are inferred from fitted residuals.",
        "Zero-arrival completeness is assumed, not independently verified. Historical holdout "
        "performance is not prospective/external validation and may not transfer under drift.",
        "These are additive ETS state contributions, not causal feature effects or SHAP explanations.",
        "The effective forward seasonal cycle is obtained from the saved model's first seven "
        "forecasts after subtracting level and trend, then checked against all 30 forecasts. "
        "It can differ from the final post-update seasonal states at the library's forecast "
        "boundary; the CSV records that difference explicitly. No model behavior is changed.",
    ]
    summary = {
        "model": MODEL_NAME, "primary_horizon_days": 30,
        "specification": {"trend": "additive", "seasonal": "additive", "seasonal_periods": 7,
                          "damped_trend": False, "initialization_method": "estimated"},
        "source_artifacts": {name: {"path": str(path.relative_to(core.ROOT)), "sha256": hashes[name]}
                             for name, path in paths.items()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "statsmodels": statsmodels.__version__},
        "history": {"start": history.index[0].date().isoformat(),
                    "end": history.index[-1].date().isoformat(), "days": len(history)},
        "parameters": fitted.params,
        "saved_final_fit_optimizer": convergence,
        "terminal_level": float(fitted.level.iloc[-1]),
        "terminal_daily_trend": float(fitted.trend.iloc[-1]),
        "terminal_week_seasonal_min": float(future["seasonal_contribution"].min()),
        "terminal_week_seasonal_max": float(future["seasonal_contribution"].max()),
        "max_effective_seasonal_difference_from_last_updated": float(
            future["forecast_seasonal_minus_last_updated"].abs().max()
        ),
        "in_sample_full_history_residuals": residual_summary,
        "held_out_primary_30_day_test": test_summary,
        "limitations": limitations,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "ets_fitted_components.csv": components,
        "ets_forecast_components_30d.csv": future,
        "ets_residual_autocorrelation.csv": correlations,
        "ets_test_errors_30d.csv": errors,
        "ets_test_errors_by_lead.csv": by_lead,
    }.items():
        table.to_csv(OUTPUT / name, index=False)
    (OUTPUT / "ets_diagnostics_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    lines = [
        "PRIMARY 30-DAY ETS: COMPONENT AND RESIDUAL EVIDENCE",
        r"Reproduce: .\.venv\Scripts\python.exe -m src.generate_ets_diagnostics",
        "",
        "No training is run. The saved final ETS and existing rolling predictions are read only.",
        "Level is the evolving baseline count; trend is the daily change in that baseline; "
        "the seven-day seasonal state adds a repeating weekday offset.",
        "One-step fitted value at t = level[t-1] + trend[t-1] + seasonal[t-7]. "
        "Post-update states at t already incorporate observation t, so summing those is NOT its fitted value.",
        "Future raw value at lead h = terminal level + h * terminal trend + repeated effective "
        "seven-day seasonal cycle used by the saved library forecast. Canonical evaluated forecasts "
        "are clipped below at zero. The effective cycle is not assumed equal to the final updated states.",
        f"Terminal level: {summary['terminal_level']:.6f}; daily trend: "
        f"{summary['terminal_daily_trend']:.6f}; seasonal range: "
        f"{summary['terminal_week_seasonal_min']:.3f} to {summary['terminal_week_seasonal_max']:.3f}.",
        f"Smoothing weights: alpha={fitted.params['smoothing_level']:.6f}, "
        f"beta={fitted.params['smoothing_trend']:.6f}, gamma={fitted.params['smoothing_seasonal']:.6f}. "
        "A zero beta means the estimated initial trend is carried forward without trend updates.",
        f"Saved final optimizer: {convergence}.",
        "",
        f"IN-SAMPLE: n={len(history)}, mean actual-minus-fitted={residual_summary['mean_actual_minus_fitted']:.4f}, "
        f"MAE={residual_summary['MAE']:.4f}, RMSE={residual_summary['RMSE']:.4f}.",
        "Positive signed error means underprediction; negative means overprediction.",
    ]
    for lag in (1, 7, 14, 28):
        row = correlations.loc[correlations["lag_days"].eq(lag)].iloc[0]
        lines.append(f"Lag {lag}: ACF={row['acf']:.5f}; unadjusted Ljung-Box cumulative p="
                     f"{row['ljung_box_pvalue_unadjusted_model_df0']:.6g}.")
    lines += [
        "",
        f"HELD-OUT: {test_summary['origin_lead_pairs']} origin/lead pairs, "
        f"{test_summary['forecast_origins']} origins, {test_summary['unique_target_dates']} unique target dates; "
        f"{test_summary['target_date_start']} to {test_summary['target_date_end']}.",
        f"Test MAE={test_summary['MAE']:.4f}, RMSE={test_summary['RMSE']:.4f}, "
        f"WAPE={test_summary['WAPE']:.2%}, sMAPE={test_summary['sMAPE']:.2%}, "
        f"bias(actual-forecast)={test_summary['signed_bias_actual_minus_forecast']:.4f}.",
        "", "LIMITATIONS", *[f"- {item}" for item in limitations],
        "", "OUTPUT GUIDE",
        "ets_components.png: recent fitted history, full-history level/trend and terminal weekday cycle.",
        "ets_residual_diagnostics.png: in-sample residuals/ACF versus held-out errors and lead metrics.",
        "ets_fitted_components.csv: full daily states, pre-update prediction contributions and residuals.",
        "ets_forecast_components_30d.csv: exact raw forecast sum and canonical clipped version.",
        "ets_residual_autocorrelation.csv: lags 1-28, ACF, heuristic bands and cumulative Ljung-Box.",
        "ets_test_errors_30d.csv: saved primary test rows plus signed/absolute errors.",
        "ets_test_errors_by_lead.csv: canonical error metrics separately for leads 1-30.",
        "ets_diagnostics_summary.json: parameters, convergence, source hashes, metrics and caveats.",
    ]
    (OUTPUT / "ets_diagnostics_readme.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _figures(components, future, correlations, errors, by_lead, OUTPUT)
    if hashes != {name: _hash(path) for name, path in paths.items()}:
        raise RuntimeError("A source artifact changed during diagnostics generation.")
    print(json.dumps(_json_safe({
        "output_directory": str(OUTPUT.relative_to(core.ROOT)),
        "in_sample": residual_summary, "held_out": test_summary, "optimizer": convergence,
    }), indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    generate()
