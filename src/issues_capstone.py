"""End-to-end analysis helpers for the historical Help Desk issues dataset."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project_config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / CONFIG["dataset"]
REPORTS = ROOT / CONFIG["ticket_volume_forecasting"]["output_directory"]
PROJECT_REPORTS = ROOT / "output" / "issue_helpdesk"
MODELS_DIR = ROOT / "models"
FORECAST_CALENDAR_POLICY = "day_after_history_end_v1"
PROJECT_REPORTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class VolumeForecastArtifact:
    """Picklable final forecast model, refit on the full available history.

    Holds only the fitted components needed to reproduce the validation-selected
    model's forecasts. It is not a live/production scoring service; see
    ``models/final_volume_forecast_model_metadata.json`` for the historical-backtest
    caveats that apply to every prediction it produces.
    """

    def __init__(self, model_name: str, feature_columns: list[str] | None = None, **fitted):
        self.model_name = model_name
        self.feature_columns = feature_columns
        self.fitted = fitted

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future daily counts given the most recent observed history."""
        dates = forecast_dates(history, horizon)
        if self.model_name == "Seasonal naive (same weekday last week)":
            return np.resize(history.iloc[-7:].to_numpy(dtype=float), horizon)
        if self.model_name == "ETS (weekly additive)":
            ets = self.fitted["ets"]
            return np.asarray(ets.forecast(horizon), dtype=float)
        if self.model_name == "XGBoost autoregression (28-day lags)":
            xgb = self.fitted["xgb"]
            rolling_values = history.to_numpy(dtype=float).tolist()
            predictions = []
            for date in dates:
                row = lag_feature_row(np.asarray(rolling_values), date)
                forecast = float(
                    xgb.predict(pd.DataFrame([row], columns=self.feature_columns))[0]
                )
                predictions.append(max(0.0, forecast))
                rolling_values.append(max(0.0, forecast))
            return np.asarray(predictions)
        if self.model_name == "Ridge (MI-selected features + PCA)":
            selector, scaler, pca, ridge = (
                self.fitted["selector"],
                self.fitted["scaler"],
                self.fitted["pca"],
                self.fitted["ridge"],
            )
            rolling_values = history.to_numpy(dtype=float).tolist()
            predictions = []
            for date in dates:
                row = pd.DataFrame(
                    [lag_feature_row(np.asarray(rolling_values), date)],
                    columns=self.feature_columns,
                )
                transformed = pca.transform(scaler.transform(selector.transform(row)))
                forecast = float(ridge.predict(transformed)[0])
                predictions.append(max(0.0, forecast))
                rolling_values.append(max(0.0, forecast))
            return np.asarray(predictions)
        raise ValueError(f"Unknown persisted model name: {self.model_name}")


def read_source(path: Path = SOURCE) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required issue dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def parse_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="mixed", errors="coerce", utc=True)


def source_and_ticket_data(path: Path = SOURCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = read_source(path)
    required = {
        "id",
        "issue_created",
        "issue_resolution_date",
        "issue_type",
        "issue_priority",
        "issue_resolution",
        "issue_status",
        "issue_proj",
        "wf_total_time",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source is missing required columns: {sorted(missing)}")

    data = source.copy()
    data["created_at"] = parse_timestamp(data["issue_created"])
    data["resolved_at"] = parse_timestamp(data["issue_resolution_date"])
    data["elapsed_wall_clock_hours"] = (
        data["resolved_at"] - data["created_at"]
    ).dt.total_seconds() / 3600
    priority_mapping = CONFIG["priority_sla_assessment"]["priority_mapping"]
    data["mapped_priority"] = data["issue_priority"].map(priority_mapping)
    data["sla_reference_hours"] = data["mapped_priority"].map(
        CONFIG["priority_sla_assessment"]["reference_targets_hours"]
    )
    data["sla_reference_met"] = (
        data["elapsed_wall_clock_hours"] <= data["sla_reference_hours"]
    )
    start = pd.Timestamp(CONFIG["analysis_scope"]["documented_start_date"], tz="UTC")
    ticket_data = data.loc[
        data["issue_type"].eq(CONFIG["analysis_scope"]["issue_type"])
        & data["created_at"].ge(start)
    ].copy()
    ticket_data["created_date"] = ticket_data["created_at"].dt.floor("D")
    ticket_data["created_year"] = ticket_data["created_at"].dt.year
    ticket_data["created_weekday"] = ticket_data["created_at"].dt.day_name()
    ticket_data["created_month"] = ticket_data["created_at"].dt.month
    ticket_data["elapsed_wall_clock_hours"] = (
        ticket_data["resolved_at"] - ticket_data["created_at"]
    ).dt.total_seconds() / 3600
    ticket_data["sla_reference_hours"] = ticket_data["mapped_priority"].map(
        CONFIG["priority_sla_assessment"]["reference_targets_hours"]
    )
    ticket_data["sla_reference_met"] = (
        ticket_data["elapsed_wall_clock_hours"]
        <= ticket_data["sla_reference_hours"]
    )
    return source, ticket_data


def daily_ticket_counts(ticket_data: pd.DataFrame) -> pd.DataFrame:
    created = ticket_data["created_date"].dropna()
    if created.empty:
        raise ValueError("No valid creation dates remain in the scoped Ticket data.")
    counts = created.value_counts().sort_index().rename("tickets")
    expected = pd.date_range(counts.index.min(), counts.index.max(), freq="D", tz="UTC")
    absent = expected.difference(counts.index)
    if len(absent) and not CONFIG["analysis_scope"]["complete_daily_coverage"]:
        raise ValueError(
            "Calendar dates without records exist; confirm coverage before zero-filling."
        )
    counts = counts.reindex(expected, fill_value=0)
    return counts.rename_axis("date_created").reset_index()


def _sla_cohort(ticket_data: pd.DataFrame) -> pd.DataFrame:
    duration = ticket_data["elapsed_wall_clock_hours"]
    return ticket_data.loc[
        ticket_data["issue_resolution"].eq("Done")
        & ticket_data["mapped_priority"].notna()
        & duration.notna()
        & np.isfinite(duration)
        & duration.ge(0)
    ].copy()


def run_data_understanding(path: Path = SOURCE) -> dict:
    source, tickets = source_and_ticket_data(path)
    profile = pd.DataFrame(
        {
            "column": source.columns,
            "dtype": [str(source[column].dtype) for column in source.columns],
            "rows_processed": len(source),
            "present_rows": [int(source[column].notna().sum()) for column in source.columns],
            "missing_rows": [int(source[column].isna().sum()) for column in source.columns],
            "missing_pct": [source[column].isna().mean() * 100 for column in source.columns],
            "unique_values": [int(source[column].nunique(dropna=True)) for column in source.columns],
        }
    )
    profile.to_csv(PROJECT_REPORTS / "raw_column_profile.csv", index=False)

    counts = daily_ticket_counts(tickets)
    counts.to_csv(REPORTS / "daily_ticket_counts.csv", index=False)
    span = pd.date_range(
        counts["date_created"].min(), counts["date_created"].max(), freq="D", tz="UTC"
    )
    covered = counts.set_index("date_created")["tickets"].reindex(span, fill_value=0)
    coverage = pd.DataFrame(
        [
            {
                "calendar_days": len(covered),
                "days_with_ticket_arrivals": int(covered.gt(0).sum()),
                "zero_ticket_days": int(covered.eq(0).sum()),
                "zero_ticket_day_pct": float(covered.eq(0).mean() * 100),
                "coverage_status": "assumed complete from dataset scope; not independently verified",
                "date_start_utc": span.min().date().isoformat(),
                "date_end_utc": span.max().date().isoformat(),
            }
        ]
    )
    coverage.to_csv(PROJECT_REPORTS / "daily_coverage_audit.csv", index=False)

    cohort = _sla_cohort(tickets)
    cohort["reference_breach"] = ~cohort["sla_reference_met"]
    sla = (
        cohort.groupby("mapped_priority", observed=True)
        .agg(
            assessed_tickets=("id", "size"),
            reference_met=("sla_reference_met", "sum"),
            median_elapsed_hours=("elapsed_wall_clock_hours", "median"),
            p90_elapsed_hours=("elapsed_wall_clock_hours", lambda values: values.quantile(0.9)),
            p95_elapsed_hours=("elapsed_wall_clock_hours", lambda values: values.quantile(0.95)),
            reference_hours=("sla_reference_hours", "first"),
        )
        .reset_index()
        .rename(columns={"mapped_priority": "priority"})
    )
    sla["reference_met_pct"] = sla["reference_met"] / sla["assessed_tickets"] * 100
    sla["reference_breach"] = sla["assessed_tickets"] - sla["reference_met"]
    total = {
        "priority": "Overall",
        "assessed_tickets": int(sla["assessed_tickets"].sum()),
        "reference_met": int(sla["reference_met"].sum()),
        "median_elapsed_hours": cohort["elapsed_wall_clock_hours"].median(),
        "p90_elapsed_hours": cohort["elapsed_wall_clock_hours"].quantile(0.9),
        "p95_elapsed_hours": cohort["elapsed_wall_clock_hours"].quantile(0.95),
        "reference_hours": np.nan,
        "reference_met_pct": cohort["sla_reference_met"].mean() * 100,
        "reference_breach": int((~cohort["sla_reference_met"]).sum()),
    }
    sla = pd.concat([sla, pd.DataFrame([total])], ignore_index=True)
    sla.to_csv(PROJECT_REPORTS / "sla_attainment_by_priority.csv", index=False)

    source_created = parse_timestamp(source["issue_created"])
    source_resolved = parse_timestamp(source["issue_resolution_date"])
    source_last_change = parse_timestamp(source["last_change_date"])
    quality = pd.DataFrame(
        [
            ("source_rows", len(source)),
            ("source_columns", len(source.columns)),
            ("unique_issue_ids", source["id"].nunique(dropna=True)),
            ("exact_duplicate_rows", source.duplicated().sum()),
            ("duplicate_project_issue_keys", source.duplicated(["issue_proj", "issue_num"]).sum()),
            ("documented_scope_start", CONFIG["analysis_scope"]["documented_start_date"]),
            ("records_before_documented_scope", int(
                (source_created < pd.Timestamp("2016-01-01", tz="UTC")).sum()
            )),
            ("created_after_resolution_rows", int((source_resolved < source_created).sum())),
            ("resolution_disposition_date_mismatch_rows", int(
                (
                    source["issue_resolution"].notna()
                    != source_resolved.notna()
                ).sum()
            )),
            ("resolution_after_last_change_rows", int((source_resolved > source_last_change).sum())),
            ("scoped_ticket_rows", len(tickets)),
            ("ticket_creation_dates_missing", int(tickets["created_at"].isna().sum())),
            ("resolved_date_missing_in_scoped_tickets", int(tickets["resolved_at"].isna().sum())),
            ("unknown_priority_in_scoped_tickets", int(tickets["issue_priority"].eq("unknown").sum())),
            ("sla_assessed_rows", len(cohort)),
            ("negative_elapsed_durations", int(tickets["elapsed_wall_clock_hours"].lt(0).sum())),
        ],
        columns=["quality_check", "value"],
    )
    quality.to_csv(PROJECT_REPORTS / "source_quality_summary.csv", index=False)

    durations = cohort["elapsed_wall_clock_hours"].dropna()
    q1, q3 = durations.quantile([0.25, 0.75])
    iqr = q3 - q1
    audit = pd.DataFrame(
        [
            {
                "rows": len(durations),
                "min_elapsed_hours": durations.min(),
                "median_elapsed_hours": durations.median(),
                "p95_elapsed_hours": durations.quantile(0.95),
                "max_elapsed_hours": durations.max(),
                "iqr_low_fence": q1 - 1.5 * iqr,
                "iqr_high_fence": q3 + 1.5 * iqr,
                "above_iqr_fence": int((durations > q3 + 1.5 * iqr).sum()),
                "wf_total_time_duration_spearman": cohort[
                    ["elapsed_wall_clock_hours", "wf_total_time"]
                ].corr(method="spearman").iloc[0, 1],
            }
        ]
    )
    audit.to_csv(PROJECT_REPORTS / "resolution_duration_audit.csv", index=False)

    manifest = {
        "source": str(path.resolve().relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_rows": len(source),
        "source_columns": list(source.columns),
        "project_goal": CONFIG["project_goal"],
        "analysis_scope": CONFIG["analysis_scope"],
        "project_workstreams": CONFIG["project_workstreams"],
        "sla_priority_mapping": CONFIG["priority_sla_assessment"]["priority_mapping"],
        "sla_reference_targets_hours": CONFIG["priority_sla_assessment"]["reference_targets_hours"],
        "sla_duration_definition": CONFIG["priority_sla_assessment"]["duration_definition"],
        "sla_assessed_rows": len(cohort),
        "ticket_scope_rows": len(tickets),
        "created_start": tickets["created_at"].min().isoformat(),
        "created_end": tickets["created_at"].max().isoformat(),
        "forecast_coverage_assumption": CONFIG["analysis_scope"]["coverage_assumption"],
        "notes": [
            "Source documentation states a 2016-01 to 2023-03 period, but the CSV includes earlier issue_created values; pre-2016 records are excluded from the primary scope.",
            "Unknown priority is excluded from priority-reference attainment.",
            "Reference values and the priority mapping are project assumptions, not verified service policy.",
            "Elapsed wall-clock duration is not established as a business-hours SLA clock.",
            "Dates without scoped Ticket records are treated as zero arrivals only under an explicit, unverified completeness assumption.",
        ],
    }
    (PROJECT_REPORTS / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    dictionary = pd.DataFrame(
        [
            {
                "column": column,
                "dtype": str(source[column].dtype),
                "rows_processed": len(source),
                "present_rows": int(source[column].notna().sum()),
                "missing_rows": int(source[column].isna().sum()),
                "unique_values": int(source[column].nunique(dropna=True)),
                "meaning_or_caveat": _column_meaning(column),
            }
            for column in source.columns
        ]
    )
    dictionary.to_csv(PROJECT_REPORTS / "data_dictionary.csv", index=False)
    return {
        "source_rows": len(source),
        "ticket_rows": len(tickets),
        "sla_assessed_rows": len(cohort),
        "sla_met": int(cohort["sla_reference_met"].sum()),
        "daily_observations": len(counts),
        "reports": PROJECT_REPORTS,
    }


def _column_meaning(column: str) -> str:
    meanings = {
        "id": "Unique source issue identifier.",
        "started": "Workflow-record start timestamp shown in the source record.",
        "ended": "Workflow-record end/last-change timestamp shown in the source record.",
        "issue_num": "Issue sequence number within its project; not globally unique.",
        "issue_reporter": "Masked user identifier of the issue reporter.",
        "issue_assignee": "Masked identifier of the last recorded assignee; missingness is substantial.",
        "issue_contr_count": "Number of users who contributed to the issue.",
        "issue_proj": "Masked project code associated with the issue.",
        "issue_comments_count": "Recorded issue comment count.",
        "issue_created": "Issue creation timestamp; parsed as UTC.",
        "issue_resolution_date": "Recorded issue resolution timestamp; may be missing.",
        "issue_type": "Issue kind; primary analysis includes exact value Ticket.",
        "issue_priority": "Recorded priority; unknown is excluded from reference assessment.",
        "issue_resolution": "Recorded resolution disposition; SLA cohort requires Done.",
        "issue_status": "Last recorded workflow status, not complete status history.",
        "wf_total_time": "Workflow-derived total processing seconds; diagnostic only.",
        "last_change_date": "Timestamp of the last recorded issue update.",
        "processing_steps": "Number of workflow steps recorded for the issue.",
    }
    if column.startswith("wfe_"):
        return "Count of transitions through the named workflow state."
    if column.startswith("wf_") and column != "wf_total_time":
        return "Workflow-derived elapsed seconds spent in the named state; sparse and not substituted for the SLA clock."
    return meanings.get(column, "")


def forecast_dates(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Use the same future calendar for backtests, tuning and persisted models."""
    if history.empty:
        raise ValueError("Forecast history must contain at least one observed day.")
    return pd.date_range(
        pd.Timestamp(history.index[-1]) + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
        tz="UTC",
    )


def lag_feature_row(history: np.ndarray, date: pd.Timestamp) -> dict[str, float]:
    if len(history) < 28:
        raise ValueError("At least 28 observed prior daily counts are required.")
    values = np.asarray(history, dtype=float)
    row = {f"lag_{lag}": float(values[-lag]) for lag in range(1, 29)}
    row["mean_7"] = float(values[-7:].mean())
    row["mean_28"] = float(values[-28:].mean())
    row["weekday_sin"] = float(np.sin(2 * np.pi * date.dayofweek / 7))
    row["weekday_cos"] = float(np.cos(2 * np.pi * date.dayofweek / 7))
    row["year_sin"] = float(np.sin(2 * np.pi * date.dayofyear / 365.25))
    row["year_cos"] = float(np.cos(2 * np.pi * date.dayofyear / 365.25))
    return row


def supervised_features(history: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    rows, targets = [], []
    for target_index in range(28, len(history)):
        rows.append(
            lag_feature_row(
                history.iloc[:target_index].to_numpy(dtype=float),
                pd.Timestamp(history.index[target_index]),
            )
        )
        targets.append(float(history.iloc[target_index]))
    return pd.DataFrame(rows), np.asarray(targets)


def run_eda() -> dict:
    source, tickets = source_and_ticket_data()
    daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
    by_type = (
        source.groupby("issue_type", dropna=False)
        .agg(rows=("id", "size"), projects=("issue_proj", "nunique"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    by_type.to_csv(PROJECT_REPORTS / "issue_type_summary.csv", index=False)
    tickets.groupby(["created_year", "issue_priority"], dropna=False).size().rename(
        "tickets"
    ).reset_index().to_csv(PROJECT_REPORTS / "tickets_by_year_priority.csv", index=False)
    tickets.groupby("created_weekday", observed=True).agg(
        tickets=("id", "size"),
        sla_eligible=("mapped_priority", "count"),
    ).reset_index().to_csv(PROJECT_REPORTS / "tickets_by_weekday.csv", index=False)
    duration_cohort = _sla_cohort(tickets)
    duration_cohort.groupby("mapped_priority", observed=True).agg(
        assessed=("id", "size"),
        median_elapsed_hours=("elapsed_wall_clock_hours", "median"),
        p90_elapsed_hours=("elapsed_wall_clock_hours", lambda values: values.quantile(0.9)),
        p95_elapsed_hours=("elapsed_wall_clock_hours", lambda values: values.quantile(0.95)),
    ).reset_index().to_csv(
        PROJECT_REPORTS / "resolution_duration_by_priority.csv", index=False
    )
    numeric_columns = [
        "issue_comments_count",
        "issue_contr_count",
        "processing_steps",
        "wf_total_time",
        "elapsed_wall_clock_hours",
    ]
    tickets[numeric_columns].corr(method="spearman").to_csv(
        PROJECT_REPORTS / "ticket_numeric_spearman_correlation.csv"
    )
    monthly = daily.copy()
    monthly["month"] = pd.to_datetime(monthly["date_created"], utc=True).dt.strftime("%Y-%m")
    monthly.groupby("month", as_index=False).agg(tickets=("tickets", "sum")).to_csv(
        PROJECT_REPORTS / "monthly_ticket_volume.csv", index=False
    )

    series = pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        name="tickets",
    )
    X, y = supervised_features(series)
    validation_start = series.index.max() - pd.Timedelta(days=729)
    feature_dates = series.index[28:]
    training_mask = feature_dates < validation_start
    X_train = X.loc[training_mask]
    y_train = y[training_mask]
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    selector = SelectKBest(
        score_func=lambda features, target: mutual_info_regression(
            features, target, random_state=42
        ),
        k=min(CONFIG["ticket_volume_forecasting"]["selected_feature_count"], X_train.shape[1]),
    )
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.get_support()]
    selection = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_information_score": selector.scores_,
            "selected_in_training_only": selector.get_support(),
        }
    ).sort_values("mutual_information_score", ascending=False)
    selection.to_csv(PROJECT_REPORTS / "forecast_training_feature_selection.csv", index=False)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=CONFIG["ticket_volume_forecasting"]["pca_explained_variance"], svd_solver="full")
    pca.fit(scaled)
    pd.DataFrame(
        [
            {
                "training_rows": len(X_train),
                "original_features": X_train.shape[1],
                "selected_features": len(selected),
                "pca_components_for_target_variance": pca.n_components_,
                "explained_variance_ratio": pca.explained_variance_ratio_.sum(),
                "fit_cutoff": validation_start.isoformat(),
                "purpose": "training-only exploratory reduction of past-count/calendar features; Ridge PCA model is evaluated separately",
            }
        ]
    ).to_csv(PROJECT_REPORTS / "forecast_pca_diagnostics.csv", index=False)
    return {
        "issue_types": by_type,
        "selected_features": selected.tolist(),
        "pca_components": int(pca.n_components_),
        "validation_start": validation_start,
        "ticket_rows": len(tickets),
    }


def _xgboost_autoregressive_forecast(
    X_train: pd.DataFrame, y_train: np.ndarray, history: pd.Series, horizon: int, params: dict
):
    """Fit one XGBoost model on precomputed supervised features and recursively
    forecast ``horizon`` days ahead, feeding each day's own forecast back in as
    a future lag. Shared by the main model comparison (`_fit_predict_models`)
    and the hyperparameter tuning/sensitivity checks so all three use identical
    forecasting logic."""
    from xgboost import XGBRegressor

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    dates = forecast_dates(history, horizon)
    rolling_values = history.to_numpy(dtype=float).tolist()
    predictions = []
    for date in dates:
        row = lag_feature_row(np.asarray(rolling_values), date)
        forecast = float(model.predict(pd.DataFrame([row], columns=X_train.columns))[0])
        predictions.append(max(0.0, forecast))
        rolling_values.append(max(0.0, forecast))
    return np.asarray(predictions), model


def _fit_predict_models(history: pd.Series, horizon: int, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    config = CONFIG["ticket_volume_forecasting"]
    if len(history) < max(config["initial_training_days"], 365):
        raise ValueError("Forecast training history must cover at least 365 calendar days.")
    dates = forecast_dates(history, horizon)
    predictions: dict[str, np.ndarray] = {
        "Seasonal naive (same weekday last week)": np.resize(
            history.iloc[-7:].to_numpy(dtype=float), horizon
        )
    }
    ets = ExponentialSmoothing(
        history.astype(float).asfreq("D"),
        trend=config["ets"]["trend"],
        seasonal=config["ets"]["seasonal"],
        seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True)
    predictions["ETS (weekly additive)"] = np.asarray(ets.forecast(horizon), dtype=float)

    X_train, y_train = supervised_features(history)
    xgb_predictions, last_xgb = _xgboost_autoregressive_forecast(
        X_train, y_train, history, horizon, config["xgboost"]
    )
    predictions["XGBoost autoregression (28-day lags)"] = xgb_predictions


    selector = SelectKBest(
        score_func=lambda features, target: mutual_info_regression(
            features, target, random_state=seed
        ),
        k=min(config["selected_feature_count"], X_train.shape[1]),
    )
    selector.fit(X_train, y_train)
    selected_train = selector.transform(X_train)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(selected_train)
    pca = PCA(n_components=config["pca_explained_variance"], svd_solver="full")
    reduced_train = pca.fit_transform(scaled_train)
    ridge = Ridge(alpha=10.0)
    ridge.fit(reduced_train, y_train)
    rolling_values = history.to_numpy(dtype=float).tolist()
    ridge_predictions = []
    for date in dates:
        row = pd.DataFrame(
            [lag_feature_row(np.asarray(rolling_values), date)],
            columns=X_train.columns,
        )
        transformed = pca.transform(scaler.transform(selector.transform(row)))
        forecast = float(ridge.predict(transformed)[0])
        ridge_predictions.append(max(0.0, forecast))
        rolling_values.append(max(0.0, forecast))
    predictions["Ridge (MI-selected features + PCA)"] = np.asarray(ridge_predictions)
    return predictions, last_xgb.feature_importances_, X_train.columns.tolist(), {
        "selected_features": X_train.columns[selector.get_support()].tolist(),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def _fit_final_model(series: pd.Series, model_name: str, seed: int) -> VolumeForecastArtifact:
    """Refit the single named model on the full available history for persistence."""
    config = CONFIG["ticket_volume_forecasting"]
    if model_name == "Seasonal naive (same weekday last week)":
        return VolumeForecastArtifact(model_name)
    if model_name == "ETS (weekly additive)":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        ets = ExponentialSmoothing(
            series.astype(float).asfreq("D"),
            trend=config["ets"]["trend"],
            seasonal=config["ets"]["seasonal"],
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True)
        return VolumeForecastArtifact(model_name, ets=ets)
    if model_name == "XGBoost autoregression (28-day lags)":
        from xgboost import XGBRegressor

        X_train, y_train = supervised_features(series)
        xgb = XGBRegressor(**config["xgboost"])
        xgb.fit(X_train, y_train, verbose=False)
        return VolumeForecastArtifact(model_name, feature_columns=X_train.columns.tolist(), xgb=xgb)
    if model_name == "Ridge (MI-selected features + PCA)":
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X_train, y_train = supervised_features(series)
        selector = SelectKBest(
            score_func=lambda features, target: mutual_info_regression(
                features, target, random_state=seed
            ),
            k=min(config["selected_feature_count"], X_train.shape[1]),
        )
        selector.fit(X_train, y_train)
        selected_train = selector.transform(X_train)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(selected_train)
        pca = PCA(n_components=config["pca_explained_variance"], svd_solver="full")
        reduced_train = pca.fit_transform(scaled_train)
        ridge = Ridge(alpha=10.0)
        ridge.fit(reduced_train, y_train)
        return VolumeForecastArtifact(
            model_name,
            feature_columns=X_train.columns.tolist(),
            selector=selector,
            scaler=scaler,
            pca=pca,
            ridge=ridge,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def persist_final_model(series: pd.Series, selection: dict, cfg: dict, selection_rule: str) -> dict:
    """Refit the validation-selected model for the primary horizon on all available
    history and save it as the single reproducible model artifact for this project,
    mirroring a standard capstone ``models/`` deliverable.

    This is still a historical-backtest artifact: the source series ends in March
    2023, so its forecasts describe that dataset's history, not live current
    operations.
    """
    import joblib

    primary_horizon = int(cfg["intended_planning_horizon_days"])
    model_name = selection["selected_model"]
    seed = int(cfg["xgboost"]["random_state"])
    artifact = _fit_final_model(series, model_name, seed)
    model_path = MODELS_DIR / "final_volume_forecast_model.joblib"
    joblib.dump(artifact, model_path)
    metadata = {
        "project_name": CONFIG["project_name"],
        "model_name": model_name,
        "selection_rule": selection_rule,
        "forecast_calendar_policy": FORECAST_CALENDAR_POLICY,
        "primary_horizon_days": primary_horizon,
        "validation_mae": selection["validation_mae"],
        "trained_on_rows": int(len(series)),
        "trained_on_date_start": series.index.min().date().isoformat(),
        "trained_on_date_end": series.index.max().date().isoformat(),
        "feature_columns": artifact.feature_columns,
        "artifact_path": str(model_path.resolve().relative_to(ROOT)),
        "usage": (
            "joblib.load(this artifact).predict(history_series, horizon_days) "
            "where history_series is a daily-indexed pandas Series of ticket counts "
            "ending immediately before the forecast start date."
        ),
        "limitations": [
            "Refit on the full historical series ending March 2023; not validated against current arrivals.",
            "Selected by minimum validation MAE for the primary horizon only; other horizons may prefer a different model.",
            "Same zero-arrival-day completeness assumption and non-verified SLA/date caveats as the rest of this project apply.",
        ],
    }
    (MODELS_DIR / "final_volume_forecast_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    absolute = np.abs(errors)
    denom = np.abs(actual).sum()
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(
        2 * absolute,
        smape_denominator,
        out=np.zeros_like(absolute),
        where=smape_denominator != 0,
    )
    return {
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "WAPE": float(absolute.sum() / denom) if denom else float("nan"),
        "sMAPE": float(smape.mean()),
        "signed_bias_actual_minus_forecast": float(errors.mean()),
    }


def run_volume_forecast() -> dict:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
        dates = pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True))
        series = pd.Series(daily["tickets"].to_numpy(dtype=float), index=dates, name="tickets")
        cfg = CONFIG["ticket_volume_forecasting"]
        test_start = len(series) - int(cfg["final_test_days"])
        validation_start = test_start - int(cfg["validation_days"])
        records, importances, pca_notes = [], [], []
        horizons = sorted(
            {int(value) for value in cfg.get(
                "evaluation_horizons_days", [cfg["intended_planning_horizon_days"]]
            )}
        )
        for horizon in horizons:
            origins = range(
                int(cfg["initial_training_days"]),
                len(series) - horizon + 1,
                int(cfg["rolling_origin_stride_days"]),
            )
            for origin in origins:
                target_dates = series.index[origin : origin + horizon]
                if len(target_dates) != horizon:
                    continue
                if origin >= test_start:
                    split = "test"
                elif origin >= validation_start and origin + horizon <= test_start:
                    split = "validation"
                else:
                    continue
                history = series.iloc[:origin]
                forecasts, importance, feature_names, ridge_info = _fit_predict_models(
                    history, horizon, int(cfg["xgboost"]["random_state"])
                )
                if split == "validation":
                    importances.extend(
                        {
                            "forecast_window_days": horizon,
                            "origin_date": history.index[-1],
                            "feature": feature,
                            "importance": float(value),
                        }
                        for feature, value in zip(feature_names, importance)
                    )
                    pca_notes.append(ridge_info)
                for model_name, predicted in forecasts.items():
                    for step, (target_date, actual, forecast) in enumerate(
                        zip(target_dates, series.iloc[origin : origin + horizon], predicted), start=1
                    ):
                        records.append(
                            {
                                "origin_date": history.index[-1],
                                "forecast_date": target_date,
                                "forecast_window_days": horizon,
                                "horizon_days": step,
                                "split": split,
                                "model": model_name,
                                "actual_tickets": float(actual),
                                "forecast_tickets_raw": float(forecast),
                                "forecast_tickets": max(0.0, float(forecast)),
                            }
                        )
    predictions = pd.DataFrame(records)
    if predictions.empty or set(predictions["split"]) != {"validation", "test"}:
        raise ValueError("Rolling forecast did not produce both full validation and test windows.")
    validation = predictions[predictions["split"].eq("validation")]
    calibration = (
        validation.assign(
            absolute_error=lambda frame: (
                frame["actual_tickets"] - frame["forecast_tickets"]
            ).abs()
        )
        .groupby(["forecast_window_days", "model"])["absolute_error"]
        .quantile(0.9)
        .to_dict()
    )
    predictions["validation_abs_error_p90"] = [
        calibration[(int(horizon), model)]
        for horizon, model in zip(
            predictions["forecast_window_days"], predictions["model"]
        )
    ]
    test_mask = predictions["split"].eq("test")
    predictions["forecast_lower_90"] = np.nan
    predictions["forecast_upper_90"] = np.nan
    predictions.loc[test_mask, "forecast_lower_90"] = np.maximum(
        0.0,
        predictions.loc[test_mask, "forecast_tickets"]
        - predictions.loc[test_mask, "validation_abs_error_p90"],
    )
    predictions.loc[test_mask, "forecast_upper_90"] = (
        predictions.loc[test_mask, "forecast_tickets"]
        + predictions.loc[test_mask, "validation_abs_error_p90"]
    )
    predictions.to_csv(REPORTS / "volume_forecast_rolling_predictions.csv", index=False)

    metric_rows = []
    for (horizon, split, model), frame in predictions.groupby(
        ["forecast_window_days", "split", "model"], sort=False
    ):
        metric_rows.append(
            {
                "split": split,
                "model": model,
                "forecast_window_days": int(horizon),
                "forecast_days": int(frame["forecast_date"].nunique()),
                "forecast_origins": int(frame["origin_date"].nunique()),
                **_metrics(frame["actual_tickets"], frame["forecast_tickets"]),
                "validation_abs_error_p90": float(calibration[(horizon, model)]),
                "coverage_90_interval": (
                    float(
                        frame.loc[frame["forecast_lower_90"].notna()].pipe(
                            lambda x: x["actual_tickets"].between(
                                x["forecast_lower_90"], x["forecast_upper_90"]
                            ).mean()
                        )
                    )
                    if split == "test"
                    else np.nan
                ),
                "interpretation": "historical backtest; transfer to current operations is unverified",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORTS / "volume_forecast_rolling_metrics.csv", index=False)
    selected_models = {}
    for horizon in horizons:
        horizon_validation = metrics.loc[
            metrics["split"].eq("validation")
            & metrics["forecast_window_days"].eq(horizon)
        ].sort_values("MAE")
        selected_models[str(horizon)] = {
            "selected_model": horizon_validation.iloc[0]["model"],
            "validation_mae": float(horizon_validation.iloc[0]["MAE"]),
            # Coverage is evaluated on test targets, not the calibration split.
            "validation_metrics": horizon_validation.assign(
                coverage_90_interval=None
            ).to_dict(orient="records"),
        }
    primary_horizon = int(cfg["intended_planning_horizon_days"])
    primary_selection = selected_models[str(primary_horizon)]
    selection = {
        "selection_rule": "minimum validation MAE only; test not used for selection",
        "forecast_calendar_policy": FORECAST_CALENDAR_POLICY,
        "primary_horizon_days": primary_horizon,
        "selected_model": primary_selection["selected_model"],
        "selected_models_by_horizon": selected_models,
        "validation_mae": primary_selection["validation_mae"],
    }
    (REPORTS / "volume_forecast_model_selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8"
    )
    model_metadata = persist_final_model(series, primary_selection, cfg, selection["selection_rule"])
    if importances:
        importance = pd.DataFrame(importances).groupby(
            ["forecast_window_days", "feature"], as_index=False
        ).agg(
            mean_validation_importance=("importance", "mean"),
            validation_origins=("origin_date", "nunique"),
        ).sort_values(
            ["forecast_window_days", "mean_validation_importance"],
            ascending=[True, False],
        )
        importance.to_csv(REPORTS / "xgboost_validation_feature_importance.csv", index=False)
    metadata = {
        "project_name": CONFIG["project_name"],
        "forecast_calendar_policy": FORECAST_CALENDAR_POLICY,
        "source": str(SOURCE.resolve().relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_rows": int(len(read_source())),
        "scoped_ticket_rows": int(daily["tickets"].sum()),
        "scope": "issue_type exactly Ticket; issue_created on/after 2016-01-01 UTC",
        "date_start": series.index.min().date().isoformat(),
        "date_end": series.index.max().date().isoformat(),
        "observed_calendar_days": int(len(series)),
        "zero_ticket_days": int((series == 0).sum()),
        "zero_ticket_day_pct": float((series == 0).mean() * 100),
        "coverage_assumption": CONFIG["analysis_scope"]["coverage_assumption"],
        "complete_daily_coverage_independently_verified": False,
        "forecast_horizon_days": primary_horizon,
        "evaluation_horizons_days": horizons,
        "validation_calendar_days": int(cfg["validation_days"]),
        "test_calendar_days": int(cfg["final_test_days"]),
        "rolling_origin_stride_days": int(cfg["rolling_origin_stride_days"]),
        "validation_forecast_origins_by_horizon": {
            str(horizon): int(
                validation.loc[
                    validation["forecast_window_days"].eq(horizon), "origin_date"
                ].nunique()
            )
            for horizon in horizons
        },
        "test_forecast_origins_by_horizon": {
            str(horizon): int(
                predictions.loc[
                    predictions["split"].eq("test")
                    & predictions["forecast_window_days"].eq(horizon),
                    "origin_date",
                ].nunique()
            )
            for horizon in horizons
        },
        "validation_forecast_origins": selected_models[str(primary_horizon)]["validation_metrics"][0]["forecast_origins"],
        "test_forecast_origins": int(
            metrics.loc[
                metrics["split"].eq("test")
                & metrics["forecast_window_days"].eq(primary_horizon),
                "forecast_origins",
            ].max()
        ),
        "selected_model": primary_selection["selected_model"],
        "selected_models_by_horizon": selected_models,
        "model_selection": selection["selection_rule"],
        "persisted_model_artifact": model_metadata["artifact_path"],
        "persisted_model_metadata": str(
            (MODELS_DIR / "final_volume_forecast_model_metadata.json").resolve().relative_to(ROOT)
        ),
        "interval_method": "symmetric 90% empirical absolute-error quantile calibrated only from validation residuals",
        "interval_caveat": "Test coverage measures this historical test only; temporal drift means future coverage is not assured.",
        "models": sorted(predictions["model"].unique().tolist()),
        "operational_status": CONFIG["ticket_volume_forecasting"]["deployment_status"],
        "limitations": [
            "Source documentation and observed pre-2016 dates disagree; the documented 2016 start is used for the primary analysis.",
            "Zero-arrival days are assumed to be fully covered; source completeness is not independently verified.",
            "Historical data ends in March 2023 and does not establish performance on current operations.",
            "The test is a historical holdout from this dataset, not external or prospective validation.",
        ],
    }
    (REPORTS / "volume_forecast_experiment.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def run_explainability() -> dict:
    metrics = pd.read_csv(REPORTS / "volume_forecast_rolling_metrics.csv")
    importance_path = REPORTS / "xgboost_validation_feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.is_file() else pd.DataFrame()
    selection = json.loads((REPORTS / "volume_forecast_model_selection.json").read_text(encoding="utf-8"))
    horizon = int(CONFIG["ticket_volume_forecasting"]["intended_planning_horizon_days"])
    metrics = metrics.loc[metrics["forecast_window_days"].eq(horizon)]
    if "forecast_window_days" in importance:
        importance = importance.loc[importance["forecast_window_days"].eq(horizon)]
    daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
    series = pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        name="tickets",
    )
    X, y = supervised_features(series)
    feature_dates = series.index[28:]
    validation_start = series.index.max() - pd.Timedelta(
        days=CONFIG["ticket_volume_forecasting"]["validation_days"]
        + CONFIG["ticket_volume_forecasting"]["final_test_days"]
        - 1
    )
    test_start = series.index[-CONFIG["ticket_volume_forecasting"]["final_test_days"]]
    training_mask = feature_dates < validation_start
    validation_mask = (feature_dates >= validation_start) & (feature_dates < test_start)
    from xgboost import XGBRegressor
    import shap

    model = XGBRegressor(**CONFIG["ticket_volume_forecasting"]["xgboost"])
    model.fit(X.loc[training_mask], y[training_mask], verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X.loc[validation_mask]))
    shap_summary = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_absolute_shap_value": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_value", ascending=False)
    shap_summary.to_csv(REPORTS / "xgboost_validation_shap_summary.csv", index=False)

    # Partial dependence (PDP): marginal effect of each top SHAP feature on the
    # XGBoost challenger's predicted daily ticket count, holding other features
    # at their observed validation-period distribution (sklearn's PDP average).
    from sklearn.inspection import partial_dependence

    pdp_features = shap_summary["feature"].head(3).tolist()
    pdp_rows = []
    for feature in pdp_features:
        pdp_result = partial_dependence(
            model, X.loc[validation_mask], features=[feature], kind="average"
        )
        grid_values = np.asarray(pdp_result["grid_values"][0])
        averages = np.asarray(pdp_result["average"][0])
        pdp_rows.extend(
            {"feature": feature, "grid_value": float(g), "avg_predicted_tickets": float(a)}
            for g, a in zip(grid_values, averages)
        )
    pdp_summary = pd.DataFrame(pdp_rows)
    pdp_summary.to_csv(REPORTS / "xgboost_validation_pdp_summary.csv", index=False)

    # ICE (Individual Conditional Expectation): one curve per validation-period
    # day for the single top-SHAP feature, showing how that day's own prediction
    # would change as the feature varies, without averaging across days the way
    # PDP does. This exposes heterogeneity (e.g. crossing/divergent curves) that
    # PDP's average line can mask.
    ice_feature = pdp_features[0]
    ice_result = partial_dependence(
        model, X.loc[validation_mask], features=[ice_feature], kind="individual"
    )
    ice_grid = np.asarray(ice_result["grid_values"][0])
    ice_curves = np.asarray(ice_result["individual"][0])
    ice_rows = [
        {
            "feature": ice_feature,
            "validation_row": int(row_index),
            "grid_value": float(g),
            "predicted_tickets": float(ice_curves[row_index, col_index]),
        }
        for row_index in range(ice_curves.shape[0])
        for col_index, g in enumerate(ice_grid)
    ]
    ice_summary = pd.DataFrame(ice_rows)
    ice_summary.to_csv(REPORTS / "xgboost_validation_ice_summary.csv", index=False)

    # LIME: local explanation for one representative validation-period forecast,
    # showing which feature values pushed that single day's prediction up or down.
    from lime.lime_tabular import LimeTabularExplainer

    X_train_values = X.loc[training_mask]
    X_validation = X.loc[validation_mask]
    validation_dates = feature_dates[np.asarray(validation_mask)]
    lime_explainer = LimeTabularExplainer(
        X_train_values.to_numpy(dtype=float),
        feature_names=X_train_values.columns.tolist(),
        mode="regression",
        random_state=int(CONFIG["ticket_volume_forecasting"]["xgboost"]["random_state"]),
    )
    lime_row_position = len(X_validation) // 2
    lime_instance_date = validation_dates[lime_row_position]
    lime_instance = X_validation.iloc[lime_row_position].to_numpy(dtype=float)
    lime_explanation = lime_explainer.explain_instance(
        lime_instance, model.predict, num_features=10
    )
    lime_summary = pd.DataFrame(
        lime_explanation.as_list(), columns=["feature_condition", "local_weight"]
    ).sort_values("local_weight", key=np.abs, ascending=False)
    lime_summary.insert(0, "explained_date", str(pd.Timestamp(lime_instance_date).date()))
    lime_summary.insert(1, "predicted_tickets", float(model.predict(lime_instance.reshape(1, -1))[0]))
    lime_summary.to_csv(REPORTS / "xgboost_validation_lime_summary.csv", index=False)

    return {
        "validation_selected_model": selection["selected_models_by_horizon"][str(horizon)]["selected_model"],
        "forecast_horizon_days": horizon,
        "validation_metrics": metrics.loc[metrics["split"].eq("validation")],
        "xgboost_importance": importance.head(15),
        "xgboost_shap": shap_summary.head(15),
        "xgboost_pdp": pdp_summary,
        "xgboost_ice": ice_summary,
        "xgboost_lime": lime_summary,
        "interpretation": "XGBoost importance describes split utility within validation-fitted models, not causal ticket drivers. If seasonal naive wins, its forecast is directly traceable to the prior week's same weekday. PDP shows the average marginal effect of each top feature across the validation period; ICE unpacks that same average into one line per validation day for the single top-SHAP feature, so divergent or crossing individual lines reveal day-to-day heterogeneity PDP's average would hide; LIME shows a single local explanation for one representative validation-day forecast. All four (SHAP, PDP, ICE, LIME) explain the XGBoost challenger only, not the model actually selected for production if it differs.",
    }


def run_hyperparameter_sensitivity() -> dict:
    """Quantify how sensitive the XGBoost challenger's validation-period accuracy
    is to its hyperparameters, evaluated only on the same training/validation
    split used elsewhere (no test-period access, so this cannot leak into the
    reported test metrics). This is a bounded sensitivity/tuning check, not a
    replacement for the full expanding-window rolling-origin backtest: it fits
    once per hyperparameter variant on the pre-validation history and scores
    one-step-ahead validation-day predictions, which is far cheaper than
    re-running the rolling-origin loop per variant while still surfacing
    whether the fixed configuration in ``configs/project_config.yaml`` is a
    reasonable, non-fragile choice.
    """
    from xgboost import XGBRegressor

    base_config = dict(CONFIG["ticket_volume_forecasting"]["xgboost"])
    daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
    series = pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        name="tickets",
    )
    X, y = supervised_features(series)
    feature_dates = series.index[28:]
    cfg = CONFIG["ticket_volume_forecasting"]
    validation_start = series.index.max() - pd.Timedelta(
        days=cfg["validation_days"] + cfg["final_test_days"] - 1
    )
    test_start = series.index[-cfg["final_test_days"]]
    training_mask = feature_dates < validation_start
    validation_mask = (feature_dates >= validation_start) & (feature_dates < test_start)
    X_train, y_train = X.loc[training_mask], y[training_mask]
    X_validation, y_validation = X.loc[validation_mask], y[validation_mask]

    variants = [
        {"label": "configured (baseline)", "overrides": {}},
        {"label": "shallower_trees", "overrides": {"max_depth": max(2, base_config["max_depth"] - 1)}},
        {"label": "deeper_trees", "overrides": {"max_depth": base_config["max_depth"] + 2}},
        {"label": "lower_learning_rate", "overrides": {"learning_rate": base_config["learning_rate"] / 2}},
        {"label": "higher_learning_rate", "overrides": {"learning_rate": base_config["learning_rate"] * 3}},
        {"label": "fewer_trees", "overrides": {"n_estimators": max(10, base_config["n_estimators"] // 2)}},
        {"label": "more_trees", "overrides": {"n_estimators": base_config["n_estimators"] * 3}},
        {"label": "no_regularization", "overrides": {"reg_lambda": 0.0}},
    ]
    rows = []
    for variant in variants:
        params = {**base_config, **variant["overrides"]}
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        predicted = np.maximum(0.0, model.predict(X_validation))
        actual = y_validation
        absolute_error = np.abs(actual - predicted)
        rows.append(
            {
                "variant": variant["label"],
                "changed_params": json.dumps(variant["overrides"]) if variant["overrides"] else "(none)",
                "validation_mae": float(absolute_error.mean()),
                "validation_rmse": float(np.sqrt((absolute_error ** 2).mean())),
                "validation_days_scored": int(len(actual)),
            }
        )
    sensitivity = pd.DataFrame(rows).sort_values("validation_mae")
    baseline_mae = sensitivity.loc[sensitivity["variant"].eq("configured (baseline)"), "validation_mae"].iloc[0]
    sensitivity["mae_change_vs_baseline_pct"] = (
        (sensitivity["validation_mae"] - baseline_mae) / baseline_mae * 100.0
    )
    sensitivity.to_csv(REPORTS / "xgboost_hyperparameter_sensitivity.csv", index=False)
    max_swing_pct = float(sensitivity["mae_change_vs_baseline_pct"].abs().max())
    return {
        "baseline_params": base_config,
        "sensitivity_table": sensitivity,
        "baseline_validation_mae": float(baseline_mae),
        "max_mae_swing_vs_baseline_pct": max_swing_pct,
        "interpretation": (
            "Single-step one-day-ahead validation-period refits used to bound hyperparameter "
            "sensitivity cheaply; the configured baseline is not always the single best variant "
            "here, but this one-step proxy is a sensitivity check, not the model-selection metric "
            "(that remains the multi-horizon rolling-origin validation MAE in "
            "volume_forecast_rolling_metrics.csv). A large swing across variants would indicate a "
            "fragile, poorly-chosen configuration; a small swing indicates the forecast is robust "
            "to reasonable hyperparameter choices."
        ),
    }


def run_hyperparameter_tuning() -> dict:
    """Two-stage hyperparameter tuning for the XGBoost forecasting challenger.

    Stage 1 (cheap search): score a compact grid of hyperparameter combinations
    on the same one-step-ahead validation split used by
    ``run_hyperparameter_sensitivity``, and keep the lowest-MAE candidate.

    Stage 2 (honest validation): re-fit only the winning candidate through the
    project's actual model-selection criterion -- the expanding-window
    rolling-origin backtest -- restricted to the validation-period origins
    (never the held-out test period) for both evaluated horizons (7 and 30
    days), and compare its rolling-origin validation MAE against the
    already-computed baseline XGBoost numbers in
    ``volume_forecast_rolling_metrics.csv``. A hyperparameter set from Stage 1
    is only treated as an improvement if it also wins under this more
    expensive, more faithful evaluation; the cheap one-step proxy alone is not
    trusted to make the final call.

    This function does NOT modify ``configs/project_config.yaml`` or re-run
    the full backtest automatically: adopting a new configuration is left as a
    deliberate, reviewed step (see the returned ``recommendation``), because
    changing it would silently change every downstream reported metric,
    persisted model and MLflow run in this project.
    """
    baseline_manifest = json.loads(
        (REPORTS / "volume_forecast_experiment.json").read_text(encoding="utf-8")
    )
    if baseline_manifest.get("forecast_calendar_policy") != FORECAST_CALENDAR_POLICY:
        raise ValueError(
            "Forecast baseline predates the calendar-alignment correction. "
            "Re-run Notebook 03 before comparing tuning candidates."
        )

    import itertools

    from xgboost import XGBRegressor

    cfg = CONFIG["ticket_volume_forecasting"]
    base_config = dict(cfg["xgboost"])
    daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
    series = pd.Series(
        daily["tickets"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        name="tickets",
    )
    X, y = supervised_features(series)
    feature_dates = series.index[28:]
    validation_start_date = series.index.max() - pd.Timedelta(
        days=cfg["validation_days"] + cfg["final_test_days"] - 1
    )
    test_start_date = series.index[-cfg["final_test_days"]]
    training_mask = feature_dates < validation_start_date
    validation_mask = (feature_dates >= validation_start_date) & (feature_dates < test_start_date)
    X_train, y_train = X.loc[training_mask], y[training_mask]
    X_validation, y_validation = X.loc[validation_mask], y[validation_mask]

    # --- Stage 1: cheap one-step-ahead grid search ---
    grid = {
        "max_depth": sorted(
            {max(2, base_config["max_depth"] - 1), base_config["max_depth"], base_config["max_depth"] + 2}
        ),
        "learning_rate": sorted(
            {
                round(base_config["learning_rate"] / 2, 4),
                base_config["learning_rate"],
                round(base_config["learning_rate"] * 2, 4),
            }
        ),
        "n_estimators": sorted(
            {max(10, base_config["n_estimators"] // 2), base_config["n_estimators"], base_config["n_estimators"] * 2}
        ),
    }
    search_rows = []
    for max_depth, learning_rate, n_estimators in itertools.product(
        grid["max_depth"], grid["learning_rate"], grid["n_estimators"]
    ):
        params = {
            **base_config,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        predicted = np.maximum(0.0, model.predict(X_validation))
        absolute_error = np.abs(y_validation - predicted)
        search_rows.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "is_configured_baseline": params == base_config,
                "one_step_validation_mae": float(absolute_error.mean()),
            }
        )
    search = pd.DataFrame(search_rows).sort_values("one_step_validation_mae").reset_index(drop=True)
    search.to_csv(REPORTS / "xgboost_tuning_search.csv", index=False)
    best_row = search.iloc[0]
    candidate_params = {
        **base_config,
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
    }
    candidate_matches_baseline = candidate_params == base_config

    # --- Stage 2: honest validation on the rolling-origin criterion ---
    baseline_metrics = pd.read_csv(REPORTS / "volume_forecast_rolling_metrics.csv")
    baseline_validation_mae = {}
    for horizon in (7, 30):
        row = baseline_metrics.loc[
            baseline_metrics["forecast_window_days"].eq(horizon)
            & baseline_metrics["split"].eq("validation")
            & baseline_metrics["model"].eq("XGBoost autoregression (28-day lags)")
        ]
        baseline_validation_mae[horizon] = float(row["MAE"].iloc[0])

    test_start = len(series) - int(cfg["final_test_days"])
    validation_start = test_start - int(cfg["validation_days"])
    rolling_rows = []
    if not candidate_matches_baseline:
        for horizon in (7, 30):
            origins = range(
                int(cfg["initial_training_days"]),
                len(series) - horizon + 1,
                int(cfg["rolling_origin_stride_days"]),
            )
            for origin in origins:
                target_dates = series.index[origin : origin + horizon]
                if len(target_dates) != horizon:
                    continue
                if not (origin >= validation_start and origin + horizon <= test_start):
                    continue  # keep this stage restricted to validation-period origins only
                history = series.iloc[:origin]
                target = series.iloc[origin : origin + horizon].to_numpy(dtype=float)
                X_history, y_history = supervised_features(history)
                forecast, _ = _xgboost_autoregressive_forecast(
                    X_history, y_history, history, horizon, candidate_params
                )
                absolute_error = np.abs(target - forecast)
                rolling_rows.append(
                    {
                        "forecast_window_days": horizon,
                        "origin_date": history.index[-1],
                        "mae": float(absolute_error.mean()),
                    }
                )
    candidate_rolling = pd.DataFrame(rolling_rows)
    candidate_validation_mae = (
        candidate_rolling.groupby("forecast_window_days")["mae"].mean().to_dict()
        if not candidate_rolling.empty
        else {}
    )

    comparison_rows = []
    for horizon in (7, 30):
        baseline_val = baseline_validation_mae[horizon]
        candidate_val = (
            baseline_val
            if candidate_matches_baseline
            else candidate_validation_mae.get(horizon, np.nan)
        )
        comparison_rows.append(
            {
                "forecast_window_days": horizon,
                "baseline_validation_mae": baseline_val,
                "candidate_validation_mae": candidate_val,
                "candidate_wins": bool(candidate_val < baseline_val) if pd.notna(candidate_val) else False,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORTS / "xgboost_tuning_rolling_validation.csv", index=False)

    if candidate_matches_baseline:
        decision = "keep_baseline"
        recommendation = (
            "Stage-1 search confirmed the currently configured hyperparameters are already the lowest "
            "one-step validation-MAE combination in the searched grid; no Stage-2 rolling-origin "
            "re-validation was necessary. No change to configs/project_config.yaml is recommended."
        )
    elif comparison["candidate_wins"].all():
        decision = "candidate_wins_both_horizons"
        recommendation = (
            f"Candidate {candidate_params} beat the current baseline on rolling-origin validation MAE at "
            "both the 7- and 30-day horizons. Adopting it would require updating "
            "configs/project_config.yaml and re-running the full pipeline (run_volume_forecast, "
            "run_explainability, model persistence and MLflow logging) so every downstream reported "
            "number reflects the new configuration consistently; this is left as a deliberate follow-up "
            "step, not applied automatically by this function."
        )
    elif comparison["candidate_wins"].any():
        decision = "mixed_result"
        recommendation = (
            f"Candidate {candidate_params} improved rolling-origin validation MAE at only one of the two "
            "evaluated horizons; the current baseline is kept since it must serve both horizons well, and "
            "this two-stage search did not find a configuration that wins at both."
        )
    else:
        decision = "keep_baseline"
        recommendation = (
            f"Candidate {candidate_params} looked best on the cheap one-step proxy but did not beat the "
            "baseline once validated on the actual rolling-origin criterion at either horizon. This is "
            "exactly the reason Stage 2 exists: the current configuration in configs/project_config.yaml "
            "is kept unchanged."
        )

    return {
        "search_grid": grid,
        "search_results": search,
        "candidate_params": candidate_params,
        "candidate_matches_baseline": candidate_matches_baseline,
        "rolling_validation_comparison": comparison,
        "decision": decision,
        "recommendation": recommendation,
    }


def run_fairness_audit() -> dict:
    _, tickets = source_and_ticket_data()
    cohort = _sla_cohort(tickets).copy()
    cohort["met"] = cohort["sla_reference_met"].astype(int)
    by_priority = cohort.groupby("mapped_priority", observed=True).agg(
        assessed=("id", "size"),
        reference_met=("met", "sum"),
        reference_met_rate=("met", "mean"),
        median_elapsed_hours=("elapsed_wall_clock_hours", "median"),
    ).reset_index()
    by_priority = _add_wilson_interval(by_priority)
    by_priority.to_csv(PROJECT_REPORTS / "sla_group_audit_by_priority.csv", index=False)
    by_year = cohort.groupby("created_year").agg(
        assessed=("id", "size"),
        reference_met=("met", "sum"),
        reference_met_rate=("met", "mean"),
        median_elapsed_hours=("elapsed_wall_clock_hours", "median"),
    ).reset_index()
    by_year = _add_wilson_interval(by_year)
    by_year.to_csv(PROJECT_REPORTS / "sla_group_audit_by_year.csv", index=False)
    by_project = cohort.groupby("issue_proj", dropna=False).agg(
        assessed=("id", "size"),
        reference_met_rate=("met", "mean"),
    ).reset_index()
    by_project["group_stability"] = np.where(
        by_project["assessed"] >= 30, "at least 30 records", "small group: unstable estimate"
    )
    by_project.to_csv(PROJECT_REPORTS / "sla_group_audit_by_project.csv", index=False)
    return {
        "priority_groups": by_priority,
        "year_groups": by_year,
        "project_group_count": len(by_project),
        "protected_attribute_status": "not present/approved; protected-group fairness cannot be assessed",
    }


def _add_wilson_interval(frame: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.proportion import proportion_confint

    result = frame.copy()
    intervals = [
        proportion_confint(int(row.reference_met), int(row.assessed), method="wilson")
        for row in result.itertuples()
    ]
    result["reference_met_wilson95_low"] = [low for low, _ in intervals]
    result["reference_met_wilson95_high"] = [high for _, high in intervals]
    return result


def run_reporting() -> dict:
    sla = pd.read_csv(PROJECT_REPORTS / "sla_attainment_by_priority.csv")
    forecast = pd.read_csv(REPORTS / "volume_forecast_rolling_metrics.csv")
    manifest = json.loads((REPORTS / "volume_forecast_experiment.json").read_text(encoding="utf-8"))
    contract = {
        "project_name": CONFIG["project_name"],
        "primary_analysis": "retrospective priority reference attainment and daily Ticket volume forecasting",
        "source": str(SOURCE.resolve().relative_to(ROOT)),
        "sla_reference": {
            "priority_mapping": CONFIG["priority_sla_assessment"]["priority_mapping"],
            "targets_hours": CONFIG["priority_sla_assessment"]["reference_targets_hours"],
            "duration": CONFIG["priority_sla_assessment"]["duration_definition"],
            "policy_status": CONFIG["priority_sla_assessment"]["applicability_status"],
        },
        "daily_volume_forecast": {
            "target": CONFIG["ticket_volume_forecasting"]["target"],
            "horizon_days": CONFIG["ticket_volume_forecasting"]["intended_planning_horizon_days"],
            "evaluated_horizons_days": manifest["evaluation_horizons_days"],
            "selected_models_by_horizon": manifest["selected_models_by_horizon"],
            "split_metrics_path": str((REPORTS / "volume_forecast_rolling_metrics.csv").relative_to(ROOT)),
            "zero_day_assumption": manifest["coverage_assumption"],
            "production_ready": False,
        },
        "sensitive_attributes_present": False,
        "roi_claim": "not calculated; intervention and cost inputs are unavailable",
    }
    (PROJECT_REPORTS / "dashboard_data_contract.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8"
    )
    sla.to_csv(PROJECT_REPORTS / "dashboard_sla_summary.csv", index=False)
    forecast.to_csv(PROJECT_REPORTS / "dashboard_forecast_metrics.csv", index=False)
    return {"contract": contract, "sla": sla, "forecast": forecast}


def run_powerbi_export() -> dict:
    """Build a Power BI ready export folder covering both the after the fact SLA
    report and the volume forecast, re-derived from source data every run.

    Every table here is recomputed from ``source_and_ticket_data()`` and the
    persisted rolling forecast tables rather than copied from fixed numbers, so
    the export automatically covers however many rows the source CSV contains
    (more tickets, more dates, more priorities all flow through unchanged); no
    row counts or date ranges are hardcoded.
    """
    powerbi_dir = PROJECT_REPORTS / "powerbi"
    powerbi_dir.mkdir(parents=True, exist_ok=True)

    _, tickets = source_and_ticket_data()
    cohort = _sla_cohort(tickets).copy()

    # Report side: one row per eligible resolved Ticket, ready for a Power BI fact table.
    fact_tickets_sla = cohort[
        [
            "id",
            "issue_proj",
            "mapped_priority",
            "sla_reference_hours",
            "created_at",
            "resolved_at",
            "created_year",
            "created_month",
            "created_weekday",
            "elapsed_wall_clock_hours",
            "sla_reference_met",
        ]
    ].rename(
        columns={
            "issue_proj": "project",
            "mapped_priority": "priority",
        }
    )
    fact_tickets_sla["reference_breach"] = ~fact_tickets_sla["sla_reference_met"]
    fact_tickets_sla.to_csv(powerbi_dir / "fact_tickets_sla.csv", index=False)

    priority_mapping = CONFIG["priority_sla_assessment"]["priority_mapping"]
    reference_targets = CONFIG["priority_sla_assessment"]["reference_targets_hours"]
    dim_priority = pd.DataFrame(
        {
            "priority": list(reference_targets.keys()),
            "sla_reference_hours": list(reference_targets.values()),
        }
    )
    dim_priority["source_priority_values"] = dim_priority["priority"].map(
        lambda mapped: ", ".join(sorted(k for k, v in priority_mapping.items() if v == mapped))
    )
    dim_priority.to_csv(powerbi_dir / "dim_priority.csv", index=False)

    sla_summary = pd.read_csv(PROJECT_REPORTS / "dashboard_sla_summary.csv")
    sla_summary.to_csv(powerbi_dir / "sla_summary_by_priority.csv", index=False)

    # Forecast side: full daily actuals plus the rolling origin forecasts and their metrics.
    daily_counts = daily_ticket_counts(tickets)
    daily_counts.to_csv(powerbi_dir / "fact_daily_ticket_volume.csv", index=False)

    rolling_predictions_path = REPORTS / "volume_forecast_rolling_predictions.csv"
    if rolling_predictions_path.exists():
        pd.read_csv(rolling_predictions_path).to_csv(
            powerbi_dir / "fact_volume_forecast.csv", index=False
        )

    rolling_metrics_path = REPORTS / "volume_forecast_rolling_metrics.csv"
    if rolling_metrics_path.exists():
        pd.read_csv(rolling_metrics_path).to_csv(
            powerbi_dir / "model_comparison_metrics.csv", index=False
        )

    contract_path = PROJECT_REPORTS / "dashboard_data_contract.json"
    if contract_path.exists():
        (powerbi_dir / "dashboard_data_contract.json").write_text(
            contract_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    spec = {
        "pages": [
            "SLA Attainment (After the Fact Report)",
            "Daily Volume and Forecast",
            "Model Comparison",
            "Data Contract and Caveats",
        ],
        "report_fact": "fact_tickets_sla.csv",
        "report_dimensions": ["dim_priority.csv"],
        "report_summary": "sla_summary_by_priority.csv",
        "forecast_facts": ["fact_daily_ticket_volume.csv", "fact_volume_forecast.csv"],
        "forecast_summary": "model_comparison_metrics.csv",
        "relationships": (
            "Join fact_tickets_sla.priority to dim_priority.priority (many-to-one). "
            "Join fact_daily_ticket_volume.date_created to fact_volume_forecast.forecast_date "
            "on a shared date dimension when actual-versus-forecast overlays are needed."
        ),
        "scaling_note": (
            "All tables are rebuilt from data/raw/issues.csv and the persisted rolling forecast "
            "tables on every run of run_powerbi_export(); row counts and date ranges are not "
            "hardcoded, so a larger or refreshed source file is picked up automatically."
        ),
        "refresh": (
            "Run the notebooks 01-08 pipeline (or python -c \"from src.issues_capstone import "
            "run_powerbi_export; run_powerbi_export()\") after replacing data/raw/issues.csv, "
            "then refresh the Power BI model against this output/issue_helpdesk/powerbi folder."
        ),
        "caveats": [
            "Historical backtest; SLA mapping is not policy validated.",
            "Zero arrival day filling is an unverified completeness assumption.",
            "No protected attributes are present; this is not a fairness dashboard.",
        ],
        "screenshots": {
            "folder": "screenshots",
            "note": (
                "Power BI Desktop is not installed in this environment, so these are rendered "
                "screenshot samples of each report page above, built from this same folder's "
                "tables, not native .pbix captures."
            ),
            "files": {
                "SLA Attainment (After the Fact Report)": "screenshots/01_sla_attainment_report.png",
                "Daily Volume and Forecast": "screenshots/02_daily_volume_and_forecast.png",
                "Model Comparison": "screenshots/03_model_comparison.png",
                "Data Contract and Caveats": "screenshots/04_data_contract_and_caveats.png",
            },
        },
    }
    (powerbi_dir / "power_bi_dashboard_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )

    try:
        from src.generate_powerbi_screenshots import build_powerbi_screenshots
    except ImportError:
        from generate_powerbi_screenshots import build_powerbi_screenshots

    screenshots = build_powerbi_screenshots()

    return {
        "powerbi_dir": str(powerbi_dir.relative_to(ROOT)),
        "fact_tickets_sla_rows": len(fact_tickets_sla),
        "fact_daily_ticket_volume_rows": len(daily_counts),
        "spec": spec,
        "screenshots": screenshots,
    }


def log_experiment_to_mlflow(tracking_uri: str | None = None) -> dict:
    """Log the notebook 03 model-selection outcome and the persisted `models/`
    artifact as one tracked MLflow run, so this capstone's single reproducible
    model has real experiment-tracking history rather than only static JSON files.

    Reads the already-generated selection, metadata and metrics files (it does not
    retrain anything), so it can be re-run any time after `run_volume_forecast()`
    to (re)record the current artifact's lineage. Tracking metadata is stored in a
    local SQLite database (`mlflow.db`, gitignored, this MLflow version's supported
    local backend) with run artifacts under `mlruns/` (also gitignored) — no remote
    tracking server is configured for this historical-backtest project.
    """
    import os

    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    import mlflow

    tracking_uri = tracking_uri or f"sqlite:///{(ROOT / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "helpdesk_volume_forecast"
    artifact_location = f"file:{(ROOT / 'mlruns').resolve()}"
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    selection = json.loads((REPORTS / "volume_forecast_model_selection.json").read_text(encoding="utf-8"))
    model_metadata = json.loads(
        (MODELS_DIR / "final_volume_forecast_model_metadata.json").read_text(encoding="utf-8")
    )
    experiment_metadata = json.loads((REPORTS / "volume_forecast_experiment.json").read_text(encoding="utf-8"))
    metrics_table = pd.read_csv(REPORTS / "volume_forecast_rolling_metrics.csv")

    primary_horizon = model_metadata["primary_horizon_days"]
    test_row = metrics_table.loc[
        metrics_table["split"].eq("test")
        & metrics_table["forecast_window_days"].eq(primary_horizon)
        & metrics_table["model"].eq(model_metadata["model_name"])
    ]

    params = {
        "model_name": model_metadata["model_name"],
        "selection_rule": selection["selection_rule"],
        "primary_horizon_days": primary_horizon,
        "trained_on_rows": model_metadata["trained_on_rows"],
        "trained_on_date_start": model_metadata["trained_on_date_start"],
        "trained_on_date_end": model_metadata["trained_on_date_end"],
        "validation_calendar_days": experiment_metadata["validation_calendar_days"],
        "test_calendar_days": experiment_metadata["test_calendar_days"],
    }
    metrics = {"validation_mae": float(model_metadata["validation_mae"])}
    if not test_row.empty:
        row = test_row.iloc[0]
        for metric in ("MAE", "RMSE", "WAPE", "sMAPE", "coverage_90_interval"):
            if metric in row and pd.notna(row[metric]):
                metrics[f"test_{metric.lower()}"] = float(row[metric])

    with mlflow.start_run(
        run_name=f"{model_metadata['model_name']} (horizon={primary_horizon}d)"
    ) as run:
        mlflow.set_tags(
            {
                "project": CONFIG["project_name"],
                "workstream": "ticket_volume_forecasting",
                "operational_status": CONFIG["ticket_volume_forecasting"]["deployment_status"],
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODELS_DIR / "final_volume_forecast_model.joblib"))
        mlflow.log_artifact(str(MODELS_DIR / "final_volume_forecast_model_metadata.json"))
        mlflow.log_artifact(str(REPORTS / "volume_forecast_model_selection.json"))
        run_id = run.info.run_id

    result = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "logged_params": params,
        "logged_metrics": metrics,
        "view_with": f"mlflow ui --backend-store-uri \"{tracking_uri}\"",
        "caveat": (
            "Tracks the historical-backtest selection and artifact only; no production "
            "training job, current data feed, or deployed serving endpoint is logged."
        ),
    }
    (PROJECT_REPORTS / "mlflow_run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_monitoring_plan() -> dict:
    predictions = pd.read_csv(REPORTS / "volume_forecast_rolling_predictions.csv")
    selection = json.loads((REPORTS / "volume_forecast_model_selection.json").read_text(encoding="utf-8"))
    horizon = int(CONFIG["ticket_volume_forecasting"]["intended_planning_horizon_days"])
    selected = selection["selected_models_by_horizon"][str(horizon)]["selected_model"]
    selected_rows = predictions.loc[
        predictions["model"].eq(selected)
        & predictions["forecast_window_days"].eq(horizon)
    ].copy()
    selected_rows["forecast_date"] = pd.to_datetime(selected_rows["forecast_date"], utc=True)
    selected_rows["signed_error_actual_minus_forecast"] = (
        selected_rows["actual_tickets"] - selected_rows["forecast_tickets"]
    )
    by_month = selected_rows.groupby(
        [selected_rows["split"], selected_rows["forecast_date"].dt.strftime("%Y-%m")]
    ).agg(
        days=("actual_tickets", "size"),
        actual_total=("actual_tickets", "sum"),
        forecast_total=("forecast_tickets", "sum"),
        mean_signed_error=("signed_error_actual_minus_forecast", "mean"),
        mae=("signed_error_actual_minus_forecast", lambda values: values.abs().mean()),
    ).reset_index(names=["split", "forecast_month"])
    by_month.to_csv(PROJECT_REPORTS / "forecast_historical_monitoring.csv", index=False)
    plan = {
        "status": "offline historical monitoring only; no live feed or scheduled job",
        "monitor": [
            "Daily source row counts and date completeness",
            "Weekly and monthly actual-versus-forecast signed bias and MAE",
            "Validation-calibrated interval coverage on future data",
            "Priority-reference mapping and resolution-date quality",
            "Data/schema drift and sustained ticket mix changes",
        ],
        "response": "Pause operational use on source gaps, sustained under-forecasting, interval undercoverage, or failed owner-approved tolerance; investigate before retraining.",
        "human_review": "Capacity planner approves actions and records overrides; forecast does not allocate staffing automatically.",
    }
    (PROJECT_REPORTS / "monitoring_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    return {"historical_monitoring": by_month, "plan": plan}


def run_business_report() -> dict:
    assumptions_path = PROJECT_REPORTS / "roi_assumptions_template.csv"
    assumptions = pd.DataFrame(
        [
            ("Annual ticket volume", "", "tickets/year", "Owner-confirmed representative volume"),
            ("Staffing cost per hour", "", "currency/hour", "Finance-approved fully loaded cost"),
            ("Forecast-guided intervention cost", "", "currency/year", "Measure in a controlled pilot"),
            ("Service outcome value", "", "currency or approved KPI", "Do not infer from forecast error"),
            ("Intervention effectiveness", "", "fraction", "Estimate from credible comparison"),
        ],
        columns=["input", "value", "unit", "required_evidence"],
    )
    assumptions.to_csv(assumptions_path, index=False)
    summary = {
        "business_question": "Can historical daily Ticket arrivals support useful 7- and 30-day capacity-planning forecasts, and does planner-reviewed use improve outcomes?",
        "current_evidence": "Historical backtest only; no current operations or intervention outcome data.",
        "roi_status": "not estimated",
        "decision_gate": [
            "Confirm data provenance, full daily coverage, and zero-arrival interpretation.",
            "Agree acceptable error and service KPIs with capacity owners before future-period evaluation.",
            "Require stable rolling backtest performance against the same-weekday baseline.",
            "Run a human-reviewed pilot and measure service and cost outcomes before an ROI claim.",
        ],
        "assumption_template": str(assumptions_path.relative_to(ROOT)),
    }
    (PROJECT_REPORTS / "business_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
