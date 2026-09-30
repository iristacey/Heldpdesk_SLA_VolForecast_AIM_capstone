"""Render a BI dashboard style proof image from the current project's real
dashboard data contract.

Power BI Desktop is not installed in this environment, so a native .pbix
screenshot cannot be produced. This script renders the same underlying data
(output/issue_helpdesk/dashboard_sla_summary.csv and
output/issue_helpdesk/dashboard_forecast_metrics.csv) as a dashboard style
image using matplotlib, styled to resemble a BI report page (KPI cards plus
charts), so there is a visual artifact backed by the current project's
figures rather than the legacy project's data.

Output: output/issue_helpdesk/dashboard_proof_visual.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "issue_helpdesk"
SLA_PATH = OUT_DIR / "dashboard_sla_summary.csv"
FORECAST_PATH = OUT_DIR / "dashboard_forecast_metrics.csv"
IMG_PATH = OUT_DIR / "dashboard_proof_visual.png"

BG = "#F3F4F6"
CARD = "#FFFFFF"
NAVY = "#1F2A44"
ACCENT = "#2E7D8C"
WARN = "#C0453D"
GREY = "#6B7280"


def load_data():
    sla = pd.read_csv(SLA_PATH)
    forecast = pd.read_csv(FORECAST_PATH)
    return sla, forecast


def kpi_card(ax, label, value, sub, color=NAVY):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22, fontweight="bold", color=color)
    ax.text(0.5, 0.32, label, ha="center", va="center", fontsize=10.5, color=NAVY)
    ax.text(0.5, 0.12, sub, ha="center", va="center", fontsize=8.5, color=GREY)


def build_dashboard():
    sla, forecast = load_data()
    overall = sla[sla["priority"] == "Overall"].iloc[0]
    by_priority = sla[sla["priority"] != "Overall"].copy()
    priority_order = ["Critical", "High", "Medium", "Low"]
    by_priority["priority"] = pd.Categorical(by_priority["priority"], priority_order, ordered=True)
    by_priority = by_priority.sort_values("priority")

    horizon7 = forecast[(forecast["forecast_window_days"] == 7) & (forecast["split"] == "validation")].copy()
    horizon7 = horizon7.sort_values("MAE")

    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    fig.suptitle(
        "Help Desk Ticket SLA and Volume Forecasting, Reporting View",
        fontsize=16, fontweight="bold", color=NAVY, x=0.03, ha="left", y=0.985,
    )
    fig.text(
        0.03, 0.955,
        "Rendered from output/issue_helpdesk/dashboard_sla_summary.csv and dashboard_forecast_metrics.csv "
        "(this project's current data). Power BI Desktop is not installed in this environment, so this image "
        "stands in for a native dashboard screenshot using the same figures.",
        fontsize=8.3, color=GREY, ha="left",
    )

    grid = fig.add_gridspec(3, 4, top=0.90, bottom=0.07, left=0.04, right=0.97, hspace=0.55, wspace=0.35)

    kpi_specs = [
        ("Assessed Tickets", f"{int(overall['assessed_tickets']):,}", "Tickets with resolution timestamps"),
        ("Reference Met", f"{overall['reference_met_pct']:.1f}%", "Share meeting priority reference hours"),
        ("Reference Breaches", f"{int(overall['reference_breach']):,}", "Tickets outside reference hours"),
        ("Median Resolution", f"{overall['median_elapsed_hours']:.0f}h", "Median elapsed hours, all priorities"),
    ]
    for i, (label, value, sub) in enumerate(kpi_specs):
        ax = fig.add_subplot(grid[0, i])
        kpi_card(ax, label, value, sub, color=WARN if "Breach" in label else NAVY)

    ax_sla = fig.add_subplot(grid[1:, 0:2])
    ax_sla.set_facecolor(CARD)
    bars = ax_sla.bar(by_priority["priority"].astype(str), by_priority["reference_met_pct"], color=ACCENT, width=0.55)
    for b, pct, n in zip(bars, by_priority["reference_met_pct"], by_priority["assessed_tickets"]):
        ax_sla.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{pct:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=8.5, color=NAVY)
    ax_sla.set_title("Reference Attainment by Priority", fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax_sla.set_ylabel("Percent meeting priority reference hours")
    ax_sla.set_ylim(0, max(by_priority["reference_met_pct"]) + 12)
    for spine in ["top", "right"]:
        ax_sla.spines[spine].set_visible(False)
    ax_sla.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax_sla.set_axisbelow(True)

    ax_fc = fig.add_subplot(grid[1:, 2:4])
    ax_fc.set_facecolor(CARD)
    colors = [ACCENT if m == "ETS (weekly additive)" else "#A9B4C0" for m in horizon7["model"]]
    bars2 = ax_fc.barh(horizon7["model"], horizon7["MAE"], color=colors)
    for b, mae in zip(bars2, horizon7["MAE"]):
        ax_fc.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2, f"{mae:.2f}", va="center", fontsize=8.5, color=NAVY)
    ax_fc.set_title("7 Day Volume Forecast, Validation MAE by Model", fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax_fc.set_xlabel("Mean absolute error, tickets per day (lower is better)")
    ax_fc.invert_yaxis()
    for spine in ["top", "right"]:
        ax_fc.spines[spine].set_visible(False)
    ax_fc.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax_fc.set_axisbelow(True)

    fig.text(
        0.03, 0.015,
        "Historical backtest on data\\raw\\issues.csv through March 2023. Not a live production dashboard. "
        "SLA reference mapping is not policy validated. ROI is not calculated.",
        fontsize=7.5, color=GREY, ha="left",
    )

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_PATH, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {IMG_PATH}")


if __name__ == "__main__":
    build_dashboard()
