"""Build the eight ordered notebooks for the issues.csv capstone workflow.

NOTE (drift warning): notebooks 03, 04, 05 and 07 have since received manual,
targeted markdown/code additions directly in their `.ipynb` JSON (added
explanatory markdown cells in 03/05; the persisted-model-artifact section in
03; SHAP+PDP+LIME and the hyperparameter-sensitivity check in 04) that are
NOT mirrored in the ``BOOKS`` list below. Re-running ``main()`` will
overwrite those notebooks with this file's older content and silently lose
that work. Before running this script again, port any notebook content you
want to keep back into the corresponding ``BOOKS`` entry first, or edit the
notebook JSON directly instead of running this generator.
"""

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
BOOTSTRAP = """from pathlib import Path
import sys
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""


def markdown(source):
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:10],
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:10],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INTRO = """# Help Desk Ticket SLA and Volume Forecasting

This capstone uses `data/raw/issues.csv` for two integrated workstreams: **(1) retrospective comparison of elapsed resolution time with project-configured SLA references by recorded priority; and (2) separate 7- and 30-day daily Ticket-volume forecasting experiments for capacity-planning research.** The 30-day forecast is a daily path, not one monthly total. The primary analysis includes `issue_type == Ticket` and creation dates on/after 2016-01-01 UTC, matching the period described in the accompanying source documentation.

**Important source caveat:** the CSV contains creation dates before 2016 despite the documentation's stated range. Those records are profiled but excluded from the primary scope. Dates without scoped Ticket rows are treated as zero arrivals in the forecasting series under an explicit completeness assumption that has not been independently verified. The source ends in March 2023 and cannot establish current operational performance.

The configured SLA references are project assumptions: Blocker/Highest → Critical (4h), High (8h), Medium (24h), Low/Lowest → Low (24h); `unknown` is excluded. The measured duration is calendar wall-clock time between creation and recorded resolution—not a verified business-hours SLA clock. Results are retrospective, not contractual compliance or a live warning model.

**Shared retrospective result across this notebook sequence:** among 16,735 eligible completed Tickets, 3,333 (19.9%) met the configured reference. By mapped priority: Critical 524/2,124 (24.7%); High 576/3,146 (18.3%); Medium 2,161/11,161 (19.4%); Low 72/304 (23.7%). Eligibility requires exact Ticket type, documented 2016+ scope, `Done`, known mapped priority, and valid nonnegative elapsed duration. Notebook 02 shows the detailed comparison; these same counts and caveats are the project-wide retrospective baseline.
"""

BOOKS = [
    (
        "01_data_understanding.ipynb",
        "# 01 | Data understanding and retrospective SLA assessment\n\nProfile source completeness, document the date-scope discrepancy, define the ticket cohort and priority mapping, and establish the retrospective reference-attainment denominator. Daily volume aggregation is also created here; the model evaluation is in Notebook 03.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_data_understanding\nresult = run_data_understanding()\nprint(result)\n"),
            code(BOOTSTRAP + """import pandas as pd
from src.issues_capstone import PROJECT_REPORTS, REPORTS
profile = pd.read_csv(PROJECT_REPORTS / "raw_column_profile.csv")
quality = pd.read_csv(PROJECT_REPORTS / "source_quality_summary.csv")
sla = pd.read_csv(PROJECT_REPORTS / "sla_attainment_by_priority.csv")
coverage = pd.read_csv(PROJECT_REPORTS / "daily_coverage_audit.csv")
display(quality)
display(coverage)
display(sla)
display(profile)
"""),
            markdown("## Interpretation guardrails\n\n- Report `Done` outcomes with a mapped known priority and nonnegative elapsed duration; unknown priority is not counted as SLA met or breached.\n- The policy references and priority mapping are not independently verified. Validate them with a service owner before compliance claims.\n- `wf_total_time` is a separate workflow-derived measure and is only compared as a data diagnostic.\n- The source's missing calendar dates are not independently proven to be true no-ticket days."),
        ],
    ),
    (
        "02_eda_feature_engineering.ipynb",
        "# 02 | Ticket EDA, feature engineering, selection and PCA\n\nThis project predicts a numeric daily ticket count, so it is a time-series regression task—not a classification task. Accuracy, precision, recall and sensitivity require class labels and are not appropriate forecast metrics here; Notebook 03 evaluates MAE, RMSE, WAPE, sMAPE and interval coverage instead. The SLA attainment labels are retrospective descriptive outcomes, not predictions from a trained classifier.\n\nThis notebook examines category and numeric distributions, weekly/monthly patterns, resolution-time relationships and numeric associations. Forecast features are engineered from prior daily counts and calendar cycles. Mutual-information feature selection and PCA are fitted on training data only; model-based importance and SHAP for the XGBoost challenger are in Notebook 04. Clustering tendency and t-SNE/UMAP are not applied because this project has no unsupervised segmentation objective; PCA is used for the supervised Ridge comparison.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_eda\nresult = run_eda()\nprint({key: value for key, value in result.items() if key != 'issue_types'})\ndisplay(result['issue_types'])\n"),
            code(BOOTSTRAP + """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.issues_capstone import PROJECT_REPORTS, REPORTS
from src.issues_capstone import source_and_ticket_data
_, ticket_data = source_and_ticket_data()
monthly = pd.read_csv(PROJECT_REPORTS / "monthly_ticket_volume.csv")
monthly.plot(x="month", y="tickets", figsize=(14, 4), legend=False, title="Monthly created Ticket records")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(daily["tickets"], bins=25, edgecolor="white")
axes[0].set(title="Distribution of daily Ticket arrivals", xlabel="Tickets/day", ylabel="Calendar days")
axes[1].hist(np.log1p(ticket_data["elapsed_wall_clock_hours"].dropna()), bins=30, edgecolor="white")
axes[1].set(title="Resolution duration distribution (log1p scale)", xlabel="log(1 + elapsed hours)", ylabel="Tickets")
plt.tight_layout()
plt.show()
print("Daily arrival distribution:")
display(daily["tickets"].describe(percentiles=[.25, .5, .75, .9, .95]).to_frame().T)
print("Recorded priorities:")
display(ticket_data["issue_priority"].value_counts(dropna=False).rename_axis("priority").to_frame("tickets"))
weekday = pd.read_csv(PROJECT_REPORTS / "tickets_by_weekday.csv")
display(weekday)
weekday.plot(x="created_weekday", y="tickets", kind="bar", legend=False, figsize=(8, 4),
             title="Ticket arrivals by weekday")
plt.ylabel("Tickets")
plt.tight_layout()
plt.show()
sla = pd.read_csv(PROJECT_REPORTS / "sla_attainment_by_priority.csv")
sla = sla.loc[~sla["priority"].eq("Overall")].copy()
priority_order = ["Critical", "High", "Medium", "Low"]
sla["priority"] = pd.Categorical(sla["priority"], categories=priority_order, ordered=True)
sla = sla.sort_values("priority")
sla["reference_breached"] = sla["assessed_tickets"] - sla["reference_met"]
sla_table = sla[[
    "priority", "reference_hours", "assessed_tickets", "reference_met",
    "reference_breached", "reference_met_pct", "median_elapsed_hours", "p90_elapsed_hours",
]].rename(columns={
    "priority": "Mapped priority",
    "reference_hours": "Configured reference (hours)",
    "assessed_tickets": "Eligible completed tickets",
    "reference_met": "Met reference",
    "reference_breached": "Exceeded reference",
    "reference_met_pct": "Met (%)",
    "median_elapsed_hours": "Median elapsed hours",
    "p90_elapsed_hours": "90th percentile elapsed hours",
})
print("Retrospective SLA-reference attainment by mapped priority")
display(sla_table.round(2))
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(
    sla["priority"].astype(str),
    sla["reference_met_pct"],
    color="#4472C4",
)
ax.set_xlim(0, 100)
ax.set_xlabel("Eligible tickets meeting configured reference (%)")
ax.set_title("Historical SLA-reference attainment by priority")
for bar, row in zip(bars, sla.itertuples()):
    ax.text(
        min(row.reference_met_pct + 1, 84),
        bar.get_y() + bar.get_height() / 2,
        f"{row.reference_met_pct:.1f}% (reference {row.reference_hours:.0f}h; n={row.assessed_tickets:,})",
        va="center",
        fontsize=9,
    )
plt.tight_layout()
plt.show()
duration = pd.read_csv(PROJECT_REPORTS / "resolution_duration_by_priority.csv")
display(duration)
correlation = pd.read_csv(PROJECT_REPORTS / "ticket_numeric_spearman_correlation.csv", index_col=0)
display(correlation.round(2))
fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=60, ha="right")
ax.set_yticks(range(len(correlation.index)), correlation.index)
for row in range(len(correlation.index)):
    for col in range(len(correlation.columns)):
        ax.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(image, ax=ax, label="Spearman correlation")
ax.set_title("Numeric-feature relationships (pairwise monotonic association)")
plt.tight_layout()
plt.show()
resolved = ticket_data.loc[
    ticket_data["issue_resolution"].eq("Done")
    & ticket_data["mapped_priority"].notna()
    & ticket_data["elapsed_wall_clock_hours"].notna()
]
resolved.boxplot(column="elapsed_wall_clock_hours", by="mapped_priority", showfliers=False, figsize=(9, 4))
plt.ylim(0, 48)
plt.title("Elapsed wall-clock resolution time by mapped priority (outliers hidden for view)")
plt.suptitle("")
plt.ylabel("Elapsed hours")
plt.tight_layout()
plt.show()
print("The boxplot is limited to 0–48 hours for readability; the duration distribution above includes the full observed range.")
display(pd.read_csv(PROJECT_REPORTS / "forecast_training_feature_selection.csv").head(15))
display(pd.read_csv(PROJECT_REPORTS / "forecast_pca_diagnostics.csv"))
"""),
            markdown("## Retrospective SLA-reference measurement by priority\n\nThe table and chart above compare observed elapsed resolution time with the configured reference separately for each mapped priority. The cohort is limited to exact `Ticket` records created in the documented 2016+ scope that are marked `Done`, have a known mapped priority, and have a valid nonnegative creation-to-resolution duration. `Met (%)` is the count within that priority whose elapsed UTC wall-clock duration is at or below its configured reference, divided by all eligible tickets in that priority; the remainder exceeded the reference. `n` is the eligible denominator. Median and 90th-percentile elapsed times show the full duration distribution alongside the binary reference comparison. These are retrospective reference rates—not model precision/recall and not confirmed contractual SLA compliance, because the configured thresholds and business-hours/pause rules have not been verified."),
            markdown("## Feature-engineering, selection and dimensionality reduction\n\nCreation timestamps are parsed as UTC, the documented period is enforced, and the primary cohort is restricted to exact `Ticket` records. Missing calendar dates are zero-filled only under the unverified coverage assumption. Each supervised row predicts today's count using only information available before that date: 28 daily lags, 7/28-day rolling means, sine/cosine weekday encodings and sine/cosine annual-cycle encodings. Rolling means summarize recent level; cyclical encodings represent recurring calendar patterns without an artificial boundary between Sunday/Monday or year-end/year-start.\n\n**Feature selection:** mutual-information `SelectKBest` ranks candidate predictors by their estimated dependence with next-day volume; the configured top 10 features are retained for the Ridge comparison. This is a filter method, not a causal ranking. **Dimensionality reduction:** standardized training features are transformed by PCA, retaining enough components for 95% explained variance (26 of 34 components in this run); the MI-selection/PCA steps are refitted within each rolling origin for Ridge evaluation. PCA is for the supervised forecast comparison, not a visual cluster map. Clustering tendency and t-SNE/UMAP are intentionally omitted because there is no unsupervised clustering question or validated cluster use case. See Notebook 04 for XGBoost model-based importance and validation-only SHAP; those explain the challenger model, not the selected ETS forecast."),
        ],
    ),
    (
        "03_model_training_comparison.ipynb",
        "# 03 | Seven-day and month-ahead ticket-volume forecasts\n\nThe retrospective SLA-reference comparison is a separate descriptive workstream: 3,333/16,735 eligible Tickets (19.9%) met the configured thresholds, with rates reported by priority in Notebook 02. It is not the model target. This experiment predicts one daily ticket count for each day in the next 7 or 30 days; the 30-day output is a daily forecast path, not a single monthly total. Compare a same-weekday-last-week baseline (weekly pattern repeated across the horizon), weekly ETS, XGBoost autoregression and a training-only mutual-information/PCA Ridge model. This is numeric time-series regression: classification accuracy, precision, recall and sensitivity do not apply unless a separate classification target and threshold are justified. Each horizon is backtested separately using expanding-window origins across a full validation year and a later final-test year. Select a model for each horizon by validation MAE only; the test does not choose models.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_volume_forecast\nmetadata = run_volume_forecast()\nprint(metadata)\n"),
            code(BOOTSTRAP + """import pandas as pd
import matplotlib.pyplot as plt
from src.issues_capstone import REPORTS
metrics = pd.read_csv(REPORTS / "volume_forecast_rolling_metrics.csv")
selection = pd.read_json(REPORTS / "volume_forecast_model_selection.json", typ="series")
display(metrics)
display(selection)
predictions = pd.read_csv(REPORTS / "volume_forecast_rolling_predictions.csv")
predictions["absolute_error"] = (predictions["actual_tickets"] - predictions["forecast_tickets"]).abs()
predictions["lead_band"] = pd.cut(
    predictions["horizon_days"],
    bins=[0, 7, 14, 21, 30],
    labels=["Days 1-7", "Days 8-14", "Days 15-21", "Days 22-30"],
)
lead_errors = predictions.loc[predictions["forecast_window_days"].eq(30)].groupby(
    ["split", "model", "lead_band"], observed=True
).agg(
    scored_daily_forecasts=("absolute_error", "size"),
    MAE=("absolute_error", "mean"),
).reset_index()
display(lead_errors)
lead_errors.loc[lead_errors["split"].eq("test")].pivot(
    index="lead_band", columns="model", values="MAE"
).plot(
    marker="o", figsize=(10, 5), title="30-day final-test MAE by lead-time band"
)
plt.ylabel("MAE (tickets/day)")
plt.tight_layout()
plt.show()
"""),
            markdown("## Reading the horizon comparison\n\nEach row in the main comparison evaluates the complete forecast path for one horizon; the 30-day metrics score daily forecasts across all 30 lead days and overlapping historical origins. The lead-band table and chart reveal whether errors tend to rise later in the month. The seasonal-naive model repeats the latest observed seven-day pattern as its baseline for days 8–30; ETS and ML forecasts are recursively evaluated for all lead days. MAE is the average daily miss in tickets; RMSE penalizes large misses more; WAPE and sMAPE summarize relative error. The validation-calibrated 90% interval coverage is historical and does not guarantee future calibration. Because the data ends in March 2023, even a measured 30-day historical backtest does not validate next-month operations today."),
        ],
    ),
    (
        "04_explainability_shap_lime_pdp.ipynb",
        "# 04 | Forecast explainability and model interpretation\n\nThe validation-selected forecast may be a statistical model rather than a tree model. Interpret the method actually selected first. As a challenger-model explanation, SHAP values are computed for a one-day-ahead XGBoost model fitted only on pre-validation history and evaluated on validation dates; no final-test outcomes inform the explanation.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_explainability\nresult = run_explainability()\nprint(result['interpretation'])\nprint('Validation-selected model:', result['validation_selected_model'])\ndisplay(result['validation_metrics'])\ndisplay(result['xgboost_importance'])\ndisplay(result['xgboost_shap'])\n"),
            code(BOOTSTRAP + """import matplotlib.pyplot as plt
importance = result["xgboost_importance"].sort_values("mean_validation_importance")
if not importance.empty:
    importance.plot.barh(x="feature", y="mean_validation_importance", figsize=(9, 5), legend=False,
                         title="XGBoost mean feature importance across validation folds")
    plt.tight_layout()
    plt.show()
shap_summary = result["xgboost_shap"].sort_values("mean_absolute_shap_value")
shap_summary.plot.barh(x="feature", y="mean_absolute_shap_value", figsize=(9, 5), legend=False,
                       title="Validation-only XGBoost mean absolute SHAP")
plt.tight_layout()
plt.show()
"""),
            markdown("## Explainability summary\n\nRead the horizon-specific validation selections in Notebook 03; the 7- and 30-day winners need not be the same. ETS is interpreted through its estimated level, trend and weekly seasonal pattern, not XGBoost feature importance or SHAP.\n\nThe explanations here describe a separate **one-day-ahead** XGBoost model fitted before the validation year. They are not directly interchangeable with each rolling-origin refit, even when XGBoost is selected for a horizon. Read the generated SHAP table for the current feature ranking. Mean absolute SHAP describes how much a feature tends to move predictions, not whether it raises or lowers a particular forecast. Correlated lags can share attribution; these are predictive associations, not causal ticket drivers, and they do not explain an ETS artifact or current operational demand."),
            markdown("## Explainability limits\n\nFor seasonal naive, the forecast is directly traceable to the previous week's same weekday. ETS represents level, trend and weekly seasonality. XGBoost SHAP values explain a challenger fitted before the validation year; correlated lags can share contribution, and SHAP is not causal. It does not explain the validation-selected ETS forecast or demonstrate drivers of real-world demand."),
        ],
    ),
    (
        "05_bias_audit_fairness_evaluation.ipynb",
        "# 05 | Ethical interpretation and subgroup audit\n\nThe retrospective reference baseline is 3,333/16,735 eligible Tickets met (19.9%); the notebook below reports variation by priority, year and masked project group. These are operational/descriptive slices, not protected-group fairness measurements. The single aggregate forecast series has no queue/site/shift segments, so forecast-allocation fairness cannot be evaluated from this file.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_fairness_audit\nresult = run_fairness_audit()\nprint(result['protected_attribute_status'])\nprint('Project groups:', result['project_group_count'])\ndisplay(result['priority_groups'])\ndisplay(result['year_groups'])\n"),
            code(BOOTSTRAP + """import pandas as pd
from src.issues_capstone import PROJECT_REPORTS
groups = pd.read_csv(PROJECT_REPORTS / "sla_group_audit_by_project.csv")
display(groups.sort_values("assessed", ascending=False).head(20))
display(groups["group_stability"].value_counts())
"""),
            markdown("## Safeguards and limitations\n\nNo approved demographic attributes are present, so demographic parity, equalized odds and protected-group fairness cannot be assessed. Priority-specific reference rules make differences in attainment partly dependent on target construction. Small project groups are unstable and should not be ranked. Before operational use, validate coverage and policy, report signed forecast errors by approved service segments, keep human review, and investigate persistent under-forecasting. The current aggregate history cannot support that segment-level forecast audit."),
        ],
    ),
    (
        "06_genai_dashboard_mlops.ipynb",
        "# 06 | Reporting contract and data integration\n\nCreate a dashboard-ready split between retrospective SLA-reference results (3,333/16,735 eligible Tickets met; 19.9%) and the 7-/30-day arrival forecast experiments. Preserve field meaning, source scope, assumptions and operational status in the contract. This notebook does not call an LLM or present any endpoint as production-ready.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_reporting\nresult = run_reporting()\nprint(result['contract'])\ndisplay(result['sla'])\ndisplay(result['forecast'])\n"),
            markdown("## Integration contract\n\nThe forecast grain is one calendar date in UTC and the count includes only exact `Ticket` records in the 2016+ scope. The SLA assessment is a separate issue-level retrospective aggregate. Never mix a forecast count with an SLA rate or represent an SLA reference label as a live breach alert. Input data coverage, target mapping and the stated historical end date must travel with downstream extracts."),
            code(BOOTSTRAP + """import pandas as pd
import matplotlib.pyplot as plt
from src.issues_capstone import PROJECT_REPORTS, REPORTS

sla = pd.read_csv(PROJECT_REPORTS / "sla_attainment_by_priority.csv")
sla = sla.loc[~sla["priority"].eq("Overall")].sort_values("reference_met_pct")
forecast_metrics = pd.read_csv(PROJECT_REPORTS / "dashboard_forecast_metrics.csv")
contract = result["contract"]["daily_volume_forecast"]
selected_models = contract["selected_models_by_horizon"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].barh(sla["priority"], sla["reference_met_pct"], color="#4472C4")
axes[0].set_xlabel("Tickets meeting configured reference (%)")
axes[0].set_title("Retrospective SLA-reference attainment")
axes[0].set_xlim(0, max(30, sla["reference_met_pct"].max() * 1.15))
for y, value in enumerate(sla["reference_met_pct"]):
    axes[0].text(value + 0.3, y, f"{value:.1f}%", va="center", fontsize=9)

test_metrics = forecast_metrics.loc[forecast_metrics["split"].eq("test")].copy()
pivot = test_metrics.pivot(index="model", columns="forecast_window_days", values="MAE")
pivot = pivot[[7, 30]]
pivot.plot(kind="bar", ax=axes[1], color=["#70AD47", "#ED7D31"])
axes[1].set_ylabel("Test MAE (tickets/day)")
axes[1].set_xlabel("")
axes[1].set_title("Historical forecast error by horizon and model")
axes[1].legend(title="Forecast days")
axes[1].tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.show()
"""),
            code(BOOTSTRAP + """import pandas as pd
from IPython.display import display
from src.issues_capstone import PROJECT_REPORTS, REPORTS

contract = result["contract"]["daily_volume_forecast"]
selected_models = contract["selected_models_by_horizon"]
metrics = pd.read_csv(PROJECT_REPORTS / "dashboard_forecast_metrics.csv")
selected_test = []
for horizon in [7, 30]:
    chosen = selected_models[str(horizon)]["selected_model"]
    row = metrics.loc[
        metrics["split"].eq("test")
        & metrics["forecast_window_days"].eq(horizon)
        & metrics["model"].eq(chosen)
    ].iloc[0]
    selected_test.append({
        "horizon_days": horizon,
        "selected_model": chosen,
        "test_MAE_tickets_per_day": row["MAE"],
        "test_RMSE_tickets_per_day": row["RMSE"],
        "test_WAPE": row["WAPE"],
        "test_interval_coverage": row["coverage_90_interval"],
    })
print("Selected model performance on the historical test period:")
display(pd.DataFrame(selected_test).round(3))

predictions = pd.read_csv(REPORTS / "volume_forecast_rolling_predictions.csv")
example_rows = []
for horizon in [7, 30]:
    chosen = selected_models[str(horizon)]["selected_model"]
    selected = predictions.loc[
        predictions["split"].eq("test")
        & predictions["forecast_window_days"].eq(horizon)
        & predictions["model"].eq(chosen)
    ].copy()
    selected["origin_date"] = pd.to_datetime(selected["origin_date"], utc=True)
    latest_origin = selected["origin_date"].max()
    example_rows.append(selected.loc[selected["origin_date"].eq(latest_origin)].head(5))
sample = pd.concat(example_rows, ignore_index=True)
sample["origin_date"] = pd.to_datetime(sample["origin_date"], utc=True).dt.strftime("%Y-%m-%d")
sample["forecast_date"] = pd.to_datetime(sample["forecast_date"], utc=True).dt.strftime("%Y-%m-%d")
display(sample[[
    "forecast_window_days", "model", "origin_date", "forecast_date", "horizon_days",
    "actual_tickets", "forecast_tickets", "forecast_lower_90", "forecast_upper_90",
]].round(1))
"""),
            markdown("## How to read the graph and sample output\n\nThe left panel summarizes historical completed-ticket attainment against the configured priority references; these are retrospective comparisons, not confirmed contractual SLA compliance. The right panel compares held-out historical daily forecast MAE for the 7- and 30-day paths; lower is better, and both horizons select models independently using validation MAE. A 30-day path is 30 daily estimates, not one monthly total.\n\nThe tables show horizon-specific held-out metrics and five example daily predictions from the latest available historical test origin for each selected model. `actual_tickets` is included only because this is a backtest; in live forecasting it would be unknown. Interval bounds are empirical validation-calibrated bounds and are shown for test rows only. The sample dates come from data ending March 2023, so this output demonstrates the reporting shape and historical test performance—not a current forecast. Data coverage, zero-arrival days, and SLA policy remain unverified."),
        ],
    ),
    (
        "07_deployment_monitoring_mlflow.ipynb",
        "# 07 | Monitoring and deployment readiness\n\nInspect historical forecast error over time and define monitoring responsibilities. Keep this predictive workstream distinct from the retrospective SLA baseline (3,333/16,735 eligible Tickets met the configured reference; 19.9%). This capstone remains an offline backtest; there is no scheduled feed, current source data, production endpoint or staffing integration.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_monitoring_plan\nresult = run_monitoring_plan()\nprint(result['plan'])\ndisplay(result['historical_monitoring'])\n"),
            markdown("## Release gates\n\nBefore a pilot: verify source completeness and the zero-day interpretation; confirm the priority mapping, SLA clock and applicability with owners; establish current data access; agree tolerances in advance; evaluate on a genuinely future period; measure interval coverage and subgroup errors where appropriate; and require a planner to review and log decisions. Stop or investigate on source gaps, sustained under-forecasting, interval undercoverage or excessive drift. No such deployment approval is established by these historical results."),
        ],
    ),
    (
        "08_business_analytics_reporting_roi.ipynb",
        "# 08 | Business interpretation, decision criteria and ROI limits\n\nThe retrospective baseline is 3,333 of 16,735 eligible completed Tickets (19.9%) meeting the configured priority references. Translate this descriptive SLA finding and the separate historical daily-volume forecast into a decision framework for a possible capacity-planning pilot. Keep the project-configured SLA assessment descriptive and the daily volume experiment predictive but historical. Do not infer staffing savings or avoided SLA events from forecast accuracy.",
        [
            code(BOOTSTRAP + "from IPython.display import display\nfrom src.issues_capstone import run_business_report\nsummary = run_business_report()\nprint(summary)\ndisplay(__import__('pandas').read_csv(ROOT / summary['assumption_template']))\n"),
            markdown("## Business conclusion\n\nThe history supports separate one-week and one-month daily forecast backtests, and each horizon can be compared with seasonal baselines. A longer horizon is useful for earlier planning but should be expected to have greater uncertainty; the lead-time error results are reported separately. The source remains old, extracted historical data with unverified zero-day coverage—not evidence of present demand. A credible ROI claim needs owner-approved costs, an intervention that changes outcomes, and a comparison design. Keep ROI inputs blank until those are available."),
        ],
    ),
]


def main():
    for name, section, cells in BOOKS:
        document = notebook([markdown(INTRO), markdown(section), *cells])
        (NOTEBOOKS / name).write_text(
            json.dumps(document, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Rebuilt {NOTEBOOKS / name}")


if __name__ == "__main__":
    main()
