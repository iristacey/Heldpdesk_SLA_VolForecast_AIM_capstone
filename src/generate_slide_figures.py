"""Generate small illustration figures for the technical deck from existing
output/issue_helpdesk artifacts (no re-training needed; reads the same CSVs the
notebooks already wrote). Keeps the notebooks as the source of analysis and
this script purely responsible for deck-ready PNGs.

Run: .venv/Scripts/python.exe src/generate_slide_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "issue_helpdesk"
VOLUME = REPORTS / "volume_forecast"

MONTHLY_VOLUME_PNG = REPORTS / "notebook02_monthly_volume.png"
LEAD_TIME_MAE_PNG = REPORTS / "notebook03_lead_time_mae.png"
EXPLAINABILITY_PNG = REPORTS / "notebook04_explainability.png"


def build_monthly_volume_figure() -> Path:
    """Monthly created Ticket-record counts (notebook 02, EDA)."""
    monthly = pd.read_csv(REPORTS / "monthly_ticket_volume.csv")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(monthly["month"], monthly["tickets"], color="#3470a8", linewidth=1.6)
    ax.set_title("Monthly created Ticket records", fontsize=11, color="#1f3752")
    ax.set_ylabel("tickets", fontsize=9)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(MONTHLY_VOLUME_PNG, dpi=170)
    plt.close(fig)
    return MONTHLY_VOLUME_PNG


def build_lead_time_mae_figure() -> Path:
    """30-day final-test MAE by lead-time band (notebook 03, model comparison)."""
    predictions = pd.read_csv(VOLUME / "volume_forecast_rolling_predictions.csv")
    predictions["absolute_error"] = (
        predictions["actual_tickets"] - predictions["forecast_tickets"]
    ).abs()
    predictions["lead_band"] = pd.cut(
        predictions["horizon_days"],
        bins=[0, 7, 14, 21, 30],
        labels=["Days 1-7", "Days 8-14", "Days 15-21", "Days 22-30"],
    )
    lead_errors = (
        predictions.loc[predictions["forecast_window_days"].eq(30) & predictions["split"].eq("test")]
        .groupby(["model", "lead_band"], observed=True)
        .agg(MAE=("absolute_error", "mean"))
        .reset_index()
    )
    pivot = lead_errors.pivot(index="lead_band", columns="model", values="MAE")
    fig, ax = plt.subplots(figsize=(9, 2.8))
    pivot.plot(marker="o", ax=ax, linewidth=1.4, markersize=4)
    ax.set_title("30-day final-test MAE by lead-time band", fontsize=11, color="#1f3752")
    ax.set_ylabel("MAE (tickets/day)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(LEAD_TIME_MAE_PNG, dpi=170)
    plt.close(fig)
    return LEAD_TIME_MAE_PNG


def build_explainability_figure() -> Path:
    """Composite explainability panel: XGBoost feature importance, mean absolute
    SHAP, partial dependence and a single LIME local explanation (notebook 04).
    """
    importance = pd.read_csv(VOLUME / "xgboost_validation_feature_importance.csv")
    if "forecast_window_days" in importance:
        importance = importance.loc[importance["forecast_window_days"].eq(importance["forecast_window_days"].max())]
    importance = importance.sort_values("mean_validation_importance")
    shap_values = pd.read_csv(VOLUME / "xgboost_validation_shap_summary.csv").sort_values(
        "mean_absolute_shap_value"
    )
    pdp = pd.read_csv(VOLUME / "xgboost_validation_pdp_summary.csv")
    lime_summary = pd.read_csv(VOLUME / "xgboost_validation_lime_summary.csv")

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2))
    ax_importance, ax_shap, ax_pdp, ax_lime = axes.flatten()

    ax_importance.barh(importance["feature"], importance["mean_validation_importance"], color="#3470a8")
    ax_importance.set_title("XGBoost feature\nimportance", fontsize=9, color="#1f3752")
    ax_importance.tick_params(labelsize=6.5)

    ax_shap.barh(shap_values["feature"], shap_values["mean_absolute_shap_value"], color="#1f3752")
    ax_shap.set_title("Mean absolute\nSHAP (validation)", fontsize=9, color="#1f3752")
    ax_shap.tick_params(labelsize=6.5)

    for feature, group in pdp.groupby("feature", sort=False):
        ax_pdp.plot(group["grid_value"], group["avg_predicted_tickets"], marker=".", markersize=3, label=feature)
    ax_pdp.set_title("Partial dependence:\ntop SHAP features", fontsize=9, color="#1f3752")
    ax_pdp.set_xlabel("feature value", fontsize=6.5)
    ax_pdp.set_ylabel("avg predicted tickets/day", fontsize=6.5)
    ax_pdp.tick_params(labelsize=6)
    ax_pdp.legend(fontsize=5.5, frameon=False)

    lime_summary = lime_summary.sort_values("local_weight")
    colors = ["#c0392b" if value < 0 else "#3470a8" for value in lime_summary["local_weight"]]
    ax_lime.barh(lime_summary["feature_condition"], lime_summary["local_weight"], color=colors)
    ax_lime.set_title(
        f"LIME local example\n{lime_summary['explained_date'].iloc[0]}"
        f" ({lime_summary['predicted_tickets'].iloc[0]:.1f} tickets)",
        fontsize=9,
        color="#1f3752",
    )
    ax_lime.tick_params(labelsize=6)

    for ax in axes.flatten():
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Explainability: what drives the XGBoost forecast challenger", fontsize=11, color="#1f3752")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(EXPLAINABILITY_PNG, dpi=170)
    plt.close(fig)
    return EXPLAINABILITY_PNG


def build_all() -> None:
    build_monthly_volume_figure()
    build_lead_time_mae_figure()
    build_explainability_figure()


if __name__ == "__main__":
    build_all()
    print("Wrote:")
    for path in (MONTHLY_VOLUME_PNG, LEAD_TIME_MAE_PNG, EXPLAINABILITY_PNG):
        print(" -", path)
