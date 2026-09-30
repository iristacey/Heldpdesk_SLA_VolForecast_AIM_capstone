"""Render sample screenshots of the Power BI report pages described in
``output/issue_helpdesk/powerbi/power_bi_dashboard_spec.json``.

Power BI Desktop is not installed in this environment, so native ``.pbix``
screenshots cannot be captured here. These images are rendered with
matplotlib directly from the tables already written to
``output/issue_helpdesk/powerbi/`` (``fact_tickets_sla.csv``,
``dim_priority.csv``, ``sla_summary_by_priority.csv``,
``fact_daily_ticket_volume.csv``, ``fact_volume_forecast.csv``,
``model_comparison_metrics.csv``), so every screenshot is backed by this
project's current data and automatically reflects however many priorities,
models or dates that data contains; nothing is hardcoded to today's row
counts.

Output: output/issue_helpdesk/powerbi/screenshots/*.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POWERBI_DIR = ROOT / "output" / "issue_helpdesk" / "powerbi"
SCREENSHOT_DIR = POWERBI_DIR / "screenshots"

BG = "#F3F4F6"
CARD = "#FFFFFF"
NAVY = "#1F2A44"
ACCENT = "#2E7D8C"
ACCENT2 = "#ED7D31"
WARN = "#C0453D"
GREY = "#6B7280"
GRID_COLOR = "#DDDDDD"

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]


def _page_frame(fig, title, note):
    fig.suptitle(title, fontsize=16, fontweight="bold", color=NAVY, x=0.03, ha="left", y=0.97)
    fig.text(0.03, 0.925, note, fontsize=8.3, color=GREY, ha="left", wrap=True)
    fig.text(
        0.03, 0.02,
        "Rendered screenshot sample, not a native Power BI Desktop capture "
        "(Power BI Desktop is not installed in this environment). Historical backtest; "
        "SLA mapping is not policy validated.",
        fontsize=7.3, color=GREY, ha="left",
    )


def _kpi_card(ax, label, value, sub, color=NAVY):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=20, fontweight="bold", color=color)
    ax.text(0.5, 0.30, label, ha="center", va="center", fontsize=10, color=NAVY)
    ax.text(0.5, 0.10, sub, ha="center", va="center", fontsize=8, color=GREY)


def render_sla_attainment_page():
    summary = pd.read_csv(POWERBI_DIR / "sla_summary_by_priority.csv")
    overall = summary[summary["priority"] == "Overall"].iloc[0]
    by_priority = summary[summary["priority"] != "Overall"].copy()
    present_order = [p for p in PRIORITY_ORDER if p in set(by_priority["priority"])]
    extra = [p for p in by_priority["priority"] if p not in present_order]
    by_priority["priority"] = pd.Categorical(by_priority["priority"], present_order + extra, ordered=True)
    by_priority = by_priority.sort_values("priority")

    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    _page_frame(
        fig,
        "SLA Attainment (After the Fact Report)",
        "Source: output/issue_helpdesk/powerbi/sla_summary_by_priority.csv, fact_tickets_sla.csv, dim_priority.csv",
    )
    grid = fig.add_gridspec(3, 4, top=0.86, bottom=0.10, left=0.05, right=0.97, hspace=0.6, wspace=0.35)

    kpis = [
        ("Assessed Tickets", f"{int(overall['assessed_tickets']):,}", "Eligible resolved tickets"),
        ("Reference Met", f"{overall['reference_met_pct']:.1f}%", "Share meeting priority reference hours"),
        ("Reference Breaches", f"{int(overall['reference_breach']):,}", "Tickets outside reference hours"),
        ("Median Resolution", f"{overall['median_elapsed_hours']:.0f}h", "Median elapsed hours, all priorities"),
    ]
    for i, (label, value, sub) in enumerate(kpis):
        ax = fig.add_subplot(grid[0, i])
        _kpi_card(ax, label, value, sub, color=WARN if "Breach" in label else NAVY)

    ax = fig.add_subplot(grid[1:, :])
    ax.set_facecolor(CARD)
    bars = ax.bar(by_priority["priority"].astype(str), by_priority["reference_met_pct"], color=ACCENT, width=0.5)
    for b, pct, n in zip(bars, by_priority["reference_met_pct"], by_priority["assessed_tickets"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{pct:.1f}%\n(n={int(n):,})",
                ha="center", va="bottom", fontsize=9, color=NAVY)
    ax.set_title("Reference Attainment by Priority", fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.set_ylabel("Percent meeting priority reference hours")
    ax.set_ylim(0, max(30.0, by_priority["reference_met_pct"].max() + 12))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)

    path = SCREENSHOT_DIR / "01_sla_attainment_report.png"
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    return path


def render_daily_volume_and_forecast_page():
    daily = pd.read_csv(POWERBI_DIR / "fact_daily_ticket_volume.csv", parse_dates=["date_created"])
    forecasts = pd.read_csv(POWERBI_DIR / "fact_volume_forecast.csv", parse_dates=["forecast_date"])

    test_7d = forecasts[
        (forecasts["forecast_window_days"] == 7) & (forecasts["split"] == "test")
    ].copy()
    if test_7d.empty:
        test_7d = forecasts[forecasts["split"].eq("test")].copy()
    best_model = test_7d.groupby("model")["forecast_tickets"].count().idxmax() if not test_7d.empty else None
    if best_model is not None:
        test_7d = test_7d[test_7d["model"] == best_model]
    latest_origin = test_7d["origin_date"].max() if not test_7d.empty else None
    window = test_7d[test_7d["origin_date"] == latest_origin].sort_values("forecast_date") if latest_origin is not None else test_7d

    window_start = window["forecast_date"].min() if not window.empty else daily["date_created"].max()
    cutoff = window_start - pd.Timedelta(days=60)
    plot_daily = daily[daily["date_created"] >= cutoff]

    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    _page_frame(
        fig,
        "Daily Volume and Forecast",
        "Source: output/issue_helpdesk/powerbi/fact_daily_ticket_volume.csv, fact_volume_forecast.csv",
    )
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)
    ax.plot(plot_daily["date_created"], plot_daily["tickets"], color=NAVY, linewidth=1.2, label="Actual daily tickets")
    if not window.empty:
        ax.plot(window["forecast_date"], window["forecast_tickets"], color=ACCENT2, marker="o",
                linewidth=1.6, label=f"Forecast ({best_model})" if best_model else "Forecast")
        if "forecast_lower_90" in window and window["forecast_lower_90"].notna().any():
            ax.fill_between(window["forecast_date"], window["forecast_lower_90"], window["forecast_upper_90"],
                             color=ACCENT2, alpha=0.2, label="90% interval")
    ax.set_ylabel("Tickets per day")
    ax.set_title("Recent actual volume with latest historical forecast window", fontsize=12,
                 fontweight="bold", color=NAVY, loc="left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.08, right=0.97)

    path = SCREENSHOT_DIR / "02_daily_volume_and_forecast.png"
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    return path


def render_model_comparison_page():
    metrics = pd.read_csv(POWERBI_DIR / "model_comparison_metrics.csv")
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    if test_metrics.empty:
        test_metrics = metrics.copy()
    horizons = sorted(test_metrics["forecast_window_days"].unique())

    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    _page_frame(
        fig,
        "Model Comparison",
        "Source: output/issue_helpdesk/powerbi/model_comparison_metrics.csv",
    )
    n = max(1, len(horizons))
    grid = fig.add_gridspec(1, n, top=0.80, bottom=0.20, left=0.06, right=0.97, wspace=0.4)
    for i, horizon in enumerate(horizons):
        subset = test_metrics[test_metrics["forecast_window_days"] == horizon].sort_values("MAE")
        ax = fig.add_subplot(grid[0, i])
        ax.set_facecolor(CARD)
        colors = [ACCENT if v == subset["MAE"].min() else "#A9B4C0" for v in subset["MAE"]]
        bars = ax.barh(subset["model"], subset["MAE"], color=colors)
        for b, mae in zip(bars, subset["MAE"]):
            ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2, f"{mae:.2f}",
                    va="center", fontsize=8.5, color=NAVY)
        ax.set_title(f"{horizon} day horizon, test MAE", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        ax.set_xlabel("Tickets/day (lower is better)")
        ax.invert_yaxis()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)

    path = SCREENSHOT_DIR / "03_model_comparison.png"
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    return path


def render_data_contract_page():
    contract_path = POWERBI_DIR / "dashboard_data_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
    forecast_block = contract.get("daily_volume_forecast", {})
    sla_block = contract.get("sla_reference", {})

    lines = [
        f"Project: {contract.get('project_name', 'n/a')}",
        f"Primary analysis: {contract.get('primary_analysis', 'n/a')}",
        f"Source file: {contract.get('source', 'n/a')}",
        "",
        "SLA reference",
        f"  Duration definition: {sla_block.get('duration', 'n/a')}",
        f"  Policy status: {sla_block.get('policy_status', 'n/a')}",
        "",
        "Daily volume forecast",
        f"  Target: {forecast_block.get('target', 'n/a')}",
        f"  Evaluated horizons (days): {forecast_block.get('evaluated_horizons_days', 'n/a')}",
        f"  Zero day assumption: {forecast_block.get('zero_day_assumption', 'n/a')}",
        f"  Production ready: {forecast_block.get('production_ready', 'n/a')}",
        "",
        f"Sensitive attributes present: {contract.get('sensitive_attributes_present', 'n/a')}",
        f"ROI claim: {contract.get('roi_claim', 'n/a')}",
    ]

    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    _page_frame(
        fig,
        "Data Contract and Caveats",
        "Source: output/issue_helpdesk/powerbi/dashboard_data_contract.json",
    )
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    wrapped = []
    for line in lines:
        if len(line) > 100:
            wrapped.extend([line[i:i + 100] for i in range(0, len(line), 100)])
        else:
            wrapped.append(line)
    ax.text(0.02, 0.97, "\n".join(wrapped), ha="left", va="top", fontsize=9.5, color=NAVY, family="monospace")
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.05, right=0.97)

    path = SCREENSHOT_DIR / "04_data_contract_and_caveats.png"
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)
    return path


def build_powerbi_screenshots() -> dict:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "sla_attainment_report": render_sla_attainment_page(),
        "daily_volume_and_forecast": render_daily_volume_and_forecast_page(),
        "model_comparison": render_model_comparison_page(),
        "data_contract_and_caveats": render_data_contract_page(),
    }
    return {name: str(path.relative_to(ROOT)) for name, path in paths.items()}


if __name__ == "__main__":
    result = build_powerbi_screenshots()
    for name, path in result.items():
        print(f"Wrote {name}: {path}")
