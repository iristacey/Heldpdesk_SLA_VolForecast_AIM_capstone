"""Render the capstone as a reference-inspired, single-page infographic.

Reads canonical analysis artifacts without retraining. Exports a 300-dpi PNG
and a vector PDF at 12 x 18 inches.

Run: .venv\\Scripts\\python.exe src\\generate_poster.py
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.issues_capstone import FORECAST_CALENDAR_POLICY

REPORTS = ROOT / "output" / "issue_helpdesk"
VOLUME = REPORTS / "volume_forecast"
PRESENTATION = ROOT / "presentations"
POSTER_PNG = PRESENTATION / "capstone_poster.png"
POSTER_PDF = PRESENTATION / "capstone_poster.pdf"

NAVY = "#07334D"
TEAL = "#227E85"
GOLD = "#DEB85E"
INK = "#263845"
MUTED = "#596D76"
WHITE = "#FFFFFF"
PAPER = "#FAFCFC"
PALE_TEAL = "#DDEFF0"
PALE_GOLD = "#FBEDC9"
BORDER = "#DCE4E6"
WIDTH, HEIGHT = 864, 1296


def _load_numbers():
    manifest = json.loads(
        (VOLUME / "volume_forecast_experiment.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(VOLUME / "volume_forecast_rolling_metrics.csv")
    sla = pd.read_csv(REPORTS / "sla_attainment_by_priority.csv")
    shap_values = pd.read_csv(VOLUME / "xgboost_validation_shap_summary.csv")
    horizon = int(manifest["forecast_horizon_days"])
    validation = metrics.loc[
        metrics["split"].eq("validation")
        & metrics["forecast_window_days"].eq(horizon)
    ].sort_values("MAE")
    winner = validation.iloc[0]
    winner_test = metrics.loc[
        metrics["split"].eq("test")
        & metrics["forecast_window_days"].eq(horizon)
        & metrics["model"].eq(winner["model"])
    ].iloc[0]
    return {
        "manifest": manifest,
        "horizon": horizon,
        "winner": winner,
        "winner_test": winner_test,
        "overall_sla": sla.loc[sla["priority"].eq("Overall")].iloc[0],
        "top_shap": ", ".join(
            shap_values.sort_values("mean_absolute_shap_value", ascending=False)
            .head(3)["feature"].tolist()
        ),
    }


def _box(ax, x, y, width, height, *, fill=WHITE, edge=BORDER, radius=10):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=0.8,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, text, *, size=11.5, color=INK, bold=False,
          serif=False, ha="left", va="top"):
    return ax.text(
        x, y, text, fontsize=size, color=color,
        fontweight="bold" if bold else "normal",
        fontfamily="DejaVu Serif" if serif else "DejaVu Sans",
        ha=ha, va=va, linespacing=1.35,
    )


def _paragraph(ax, x, y, text, width, *, size=11.5, max_lines=2):
    """Wrap by rendered width in page points; reject overflow, never shrink."""
    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(family="DejaVu Sans", size=size)
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        pixels = renderer.get_text_width_height_descent(candidate, font, False)[0]
        if pixels * 72 / ax.figure.dpi > width:
            if not current:
                raise ValueError(f"Poster word is too wide: {word!r}")
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        raise ValueError(f"Poster copy exceeds {max_lines} lines: {text!r}")
    return _text(ax, x, y, "\n".join(lines), size=size)


def _icon(ax, kind, x, y):
    """Small original line icons, drawn in equal-aspect page coordinates."""
    def line(points, color=NAVY, lw=2.3):
        ax.plot(
            [x + p[0] for p in points], [y + p[1] for p in points],
            color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round",
        )

    def circle(cx, cy, radius):
        ax.add_patch(Circle(
            (x + cx, y + cy), radius, fill=False, edgecolor=NAVY, linewidth=2.3,
        ))

    if kind == "document":
        line([(4, 0), (32, 0), (44, 12), (44, 52), (4, 52), (4, 0)])
        line([(32, 0), (32, 13), (44, 13)])
        for yy, end in [(23, 34), (33, 34), (43, 26)]:
            line([(13, yy), (end, yy)])
    elif kind == "database":
        for yy in (9, 25, 41):
            ax.add_patch(Ellipse(
                (x + 24, y + yy), 43, 17,
                facecolor=WHITE, edgecolor=NAVY, linewidth=2.3,
                zorder=4 - yy / 100,
            ))
        line([(2.5, 9), (2.5, 41)])
        line([(45.5, 9), (45.5, 41)])
    elif kind == "features":
        for xx, top in [(2, 33), (19, 18), (36, 2)]:
            line([(xx, 52), (xx, top), (xx + 10, top), (xx + 10, 52), (xx, 52)])
    elif kind == "models":
        for cx, cy in [(6, 9), (42, 9), (24, 44)]:
            circle(cx, cy, 8)
        line([(14, 9), (34, 9)])
        line([(10, 17), (20, 36)])
        line([(38, 17), (28, 36)])
    elif kind == "forecast":
        line([(1, 1), (1, 50), (49, 50)])
        line([(8, 37), (19, 23), (29, 30), (45, 9)], color=TEAL, lw=3)
        line([(34, 9), (45, 9), (45, 20)], color=TEAL)
    elif kind == "clock":
        circle(24, 26, 23)
        line([(24, 10), (24, 26), (37, 33)])
    elif kind == "search":
        circle(20, 21, 19)
        line([(34, 35), (49, 51)], lw=3)
    else:
        raise ValueError(f"Unknown poster icon: {kind}")
    line([(3, 65), (46, 65)], color=GOLD, lw=2)


def _section(ax, number, title, tag, icon, body, takeaway, *, highlight=False):
    top = 198 + (number - 1) * 122
    _box(ax, 20, top, 824, 112, fill="#F0F8F8" if highlight else WHITE)
    ax.add_patch(Circle((45, top + 27), 19, facecolor=NAVY, edgecolor="none"))
    _text(ax, 45, top + 27, str(number), size=20, bold=True,
          serif=True, color=WHITE, ha="center", va="center")
    _icon(ax, icon, 82, top + 24)
    ax.plot([146, 146], [top + 15, top + 96], color=TEAL, lw=1)
    _text(ax, 164, top + 12, title, size=17, serif=True, bold=True, color=NAVY)
    _box(ax, 746, top + 12, 84, 24, fill=PALE_GOLD, edge=PALE_GOLD, radius=7)
    _text(ax, 788, top + 24, tag, size=9, bold=True, color=NAVY,
          ha="center", va="center")
    _paragraph(ax, 164, top + 40, body, 658)
    _box(ax, 160, top + 81, 670, 23, fill=PALE_TEAL, edge=PALE_TEAL, radius=7)
    _text(ax, 172, top + 92.5, "TAKEAWAY", size=9.5, bold=True, color=TEAL, va="center")
    _text(ax, 241, top + 92.5, takeaway, size=10.3, color=NAVY, va="center")


def _bottom_panel(ax, x, title, items, *, dark=False):
    top = 1056
    _box(ax, x, top, 405, 156)
    color = NAVY if dark else TEAL
    _box(ax, x, top, 405, 31, fill=color, edge=color, radius=9)
    ax.add_patch(Polygon(
        [(x, top + 16), (x + 405, top + 16),
         (x + 405, top + 31), (x, top + 31)],
        facecolor=color, edgecolor="none",
    ))
    _text(ax, x + 15, top + 15.5, title, size=12.5, bold=True,
          color=WHITE, va="center")
    for i, (label, detail) in enumerate(items):
        yy = top + 44 + i * 35
        ax.add_patch(Circle((x + 19, yy + 6), 3, facecolor=TEAL, edgecolor="none"))
        _text(ax, x + 31, yy, label, size=10.7, bold=True, color=NAVY)
        _text(ax, x + 31, yy + 15, detail, size=9.8, color=MUTED)


def _validate_layout(fig):
    """Catch clipped text and text collisions before publishing either file."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        text for ax in fig.axes for text in ax.texts
        if text.get_visible() and text.get_text()
    ]
    bounds = [text.get_window_extent(renderer) for text in texts]
    for text, box in zip(texts, bounds):
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            raise ValueError(f"Text outside poster: {text.get_text()!r}")
    for i, box in enumerate(bounds):
        for j in range(i + 1, len(bounds)):
            if box.overlaps(bounds[j]):
                raise ValueError(
                    f"Overlapping poster text: {texts[i].get_text()!r} "
                    f"and {texts[j].get_text()!r}"
                )


def _create_poster(numbers):
    manifest = numbers["manifest"]
    winner = numbers["winner"]
    result = numbers["winner_test"]
    sla = numbers["overall_sla"]
    horizon = numbers["horizon"]
    fig = plt.figure(figsize=(12, 18), dpi=144, facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect="equal")
    ax.axis("off")

    _text(ax, 432, 16, "HELP DESK TICKET SLA", size=31, bold=True, serif=True,
          color=NAVY, ha="center")
    _text(ax, 432, 60, "& VOLUME FORECASTING", size=31, bold=True, serif=True,
          color=NAVY, ha="center")
    comparison_current = manifest.get("forecast_calendar_policy") == FORECAST_CALENDAR_POLICY
    subtitle = (
        "AI / ML CAPSTONE  |  Historical evidence. Reproducible research."
        if comparison_current else
        "RE-RUN REQUIRED  |  Scores predate the calendar-feature correction."
    )
    _text(ax, 432, 103, subtitle,
          size=12, color=TEAL, ha="center", bold=True)
    ax.plot([20, 844], [126, 126], color=NAVY, lw=1)

    stats = [
        (f"{manifest['scoped_ticket_rows']:,}", "SCOPED TICKET RECORDS"),
        (f"{result['MAE']:.2f}", f"TICKETS / DAY MAE  |  {horizon}-DAY TEST"),
        (f"{sla['reference_met_pct']:.2f}%", "SLA REFERENCE MET  |  HISTORICAL"),
    ]
    for i, (value, label) in enumerate(stats):
        center = 157 + i * 275
        _text(ax, center, 137, value, size=26, bold=True,
              color=TEAL if i == 1 else NAVY, ha="center")
        _text(ax, center, 173, label, size=8.3, bold=True, color=MUTED, ha="center")
        if i < 2:
            ax.plot([294 + i * 275] * 2, [140, 182], color=BORDER, lw=1)

    _section(
        ax, 1, "FRAME THE PROBLEM", "PURPOSE", "document",
        "Two linked questions: how often did resolved tickets meet priority reference times, "
        "and how well can past arrivals predict daily help-desk demand?",
        "Retrospective service analysis + 7- and 30-day volume forecasts.",
    )
    _section(
        ax, 2, "UNDERSTAND THE DATA", "DATA", "database",
        f"{manifest['source_rows']:,} source rows; {manifest['scoped_ticket_rows']:,} exact Ticket records. "
        f"Scope: {manifest['date_start']} to {manifest['date_end']} (UTC). "
        "Earlier timestamps are excluded and flagged.",
        f"{manifest['observed_calendar_days']:,} calendar days; source identity verified, completeness unverified.",
    )
    _section(
        ax, 3, "CLEAN, EXPLORE & ENGINEER", "PROCESS", "features",
        "Build daily counts with past-only lags and calendar features. Compare training-only "
        "mutual-information selection and PCA for the Ridge challenger.",
        f"{manifest['zero_ticket_days']:,} empty dates are zero-filled under an unverified completeness assumption.",
    )
    _section(
        ax, 4, "BUILD & COMPARE MODELS", "METHOD", "models",
        "Compare seasonal naive, weekly ETS, XGBoost and MI/PCA Ridge using expanding-window "
        "backtests. Select on validation MAE; keep a later test year separate.",
        f"Selected model: {winner['model']}; {horizon}-day validation MAE {winner['MAE']:.2f}.",
        highlight=True,
    )
    _section(
        ax, 5, "READ THE FORECAST RESULTS", "RESULT", "forecast",
        f"{horizon}-day horizon: test MAE {result['MAE']:.2f} tickets/day; "
        f"RMSE {result['RMSE']:.2f}; WAPE {result['WAPE']:.2%}. "
        "This forecasts a daily path, not a single monthly total.",
        f"90% interval coverage: {result['coverage_90_interval']:.1%} on historical tests; calibration is imperfect.",
        highlight=True,
    )
    _section(
        ax, 6, "ASSESS SLA REFERENCE TIMES", "SERVICE", "clock",
        f"{int(sla['reference_met']):,} of {int(sla['assessed_tickets']):,} eligible resolved tickets "
        f"({sla['reference_met_pct']:.2f}%) met the reference. "
        "Critical: 4h; High: 8h; Medium / Low: 24h, using elapsed UTC time.",
        "Project-defined thresholds, not approved policy or contractual SLA compliance.",
    )
    _section(
        ax, 7, "EXPLAIN, AUDIT & REFLECT", "AUDIT", "search",
        f"XGBoost SHAP: {numbers['top_shap']}. "
        "Separate ETS state/residual diagnostics are in the report. No protected attributes: "
        "demographic fairness cannot be assessed.",
        "Feature attribution shows model association, not causal demand drivers.",
    )

    _bottom_panel(ax, 20, "PACKAGE & DEMONSTRATE", [
        ("Reproducible analysis", "Eight notebooks; shared outputs; report and two decks."),
        ("Model artifact + MLflow", "Saved model, metadata and tracked experiment results."),
        ("FastAPI + Power BI-ready exports", "Deployment proof of concept, not a live service."),
    ])
    _bottom_panel(ax, 439, "BEFORE ANY OPERATIONAL USE", [
        ("Resolve rights and coverage", "Retain CC BY 4.0; resolve mirror label and zero-day gaps."),
        ("Approve the service definitions", "Validate priority mapping, SLA clock and pause rules."),
        ("Re-test on current data", "Audit drift, interval coverage and operational segments."),
    ], dark=True)

    _box(ax, 20, 1226, 824, 40, fill=PALE_GOLD, edge=PALE_GOLD, radius=9)
    _text(ax, 432, 1236, "RESEARCH PROTOTYPE  |  Historical results, not current-operations evidence.",
          size=12, color=NAVY, bold=True, ha="center")
    _text(ax, 432, 1275,
          "Source listing: Kaggle / Mendeley Data  |  Full narrative: final_capstone_report.md",
          size=9, color=MUTED, ha="center")
    return fig


def build_poster() -> Path:
    with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig = _create_poster(_load_numbers())
        try:
            _validate_layout(fig)
            PRESENTATION.mkdir(parents=True, exist_ok=True)
            fig.savefig(POSTER_PNG, dpi=300, facecolor=PAPER)
            fig.savefig(POSTER_PDF, facecolor=PAPER, metadata={
                "Title": "Help Desk Ticket SLA and Volume Forecasting",
                "Subject": "Historical AI/ML capstone research poster",
            })
        finally:
            plt.close(fig)
    return POSTER_PNG


if __name__ == "__main__":
    path = build_poster()
    print("Wrote:", path)
    print("Wrote:", POSTER_PDF)
