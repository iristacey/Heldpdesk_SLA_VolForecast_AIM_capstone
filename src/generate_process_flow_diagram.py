"""Render a visual, numbered stage diagram of this project's actual ML and
data forecasting pipeline (source data to notebooks to reports/decks/app),
in the style of a typical machine learning process flow graphic: numbered
boxes with a short title and description, connected by arrows in sequence.

This mirrors the exact step order documented in docs/process_flow.md and the
notebook chain in src/generate_issue_notebooks.py; it is not a stock or
decorative image, it is a rendering of this project's real pipeline so it
stays correct if the pipeline changes.

Output: output/issue_helpdesk/process_flow_diagram.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "issue_helpdesk"
IMG_PATH = OUT_DIR / "process_flow_diagram.png"

NAVY = "#1F3752"
BLUE = "#3470A8"
LIGHT_BLUE = "#E8F0F7"
GREY = "#4B545D"
WHITE = "#FFFFFF"

STAGES = [
    ("1", "Data collection", "data/raw/issues.csv\n66,691 source rows, 58 fields"),
    ("2", "Data understanding", "Notebook 01: scope, profile,\nSLA reference cohort"),
    ("3", "EDA & feature engineering", "Notebook 02: lags, calendar\nfeatures, MI selection, PCA"),
    ("4", "Model training & comparison", "Notebook 03: rolling backtest,\n4 candidates, model selection"),
    ("5", "Explainability", "Notebook 04: validation only\nSHAP / feature importance"),
    ("6", "Fairness audit", "Notebook 05: operational\nsubgroup SLA review"),
    ("7", "Reporting & MLOps", "Notebooks 06 to 07: dashboard\ncontract, MLflow, monitoring plan"),
    ("8", "Business report & deployment", "Notebook 08, decks and\nFastAPI forecast proof of concept"),
]


def _stage_box(ax, cx, cy, number, title, desc, width=2.55, height=1.55):
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=BLUE, facecolor=WHITE,
    )
    ax.add_patch(box)
    badge = Circle((cx - width / 2 + 0.28, cy + height / 2 - 0.28), 0.22, facecolor=NAVY, edgecolor="none", zorder=3)
    ax.add_patch(badge)
    ax.text(cx - width / 2 + 0.28, cy + height / 2 - 0.28, number, ha="center", va="center",
            fontsize=12, fontweight="bold", color=WHITE, zorder=4)
    ax.text(cx, cy + 0.28, title, ha="center", va="center", fontsize=10.3, fontweight="bold", color=NAVY, wrap=True)
    ax.text(cx, cy - 0.32, desc, ha="center", va="center", fontsize=8.3, color=GREY)


def _arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=BLUE, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def build_diagram():
    fig, ax = plt.subplots(figsize=(14.5, 6.6), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    fig.suptitle(
        "Help Desk Ticket SLA and Volume Forecasting: ML and data forecasting process flow",
        fontsize=14.5, fontweight="bold", color=NAVY, x=0.03, ha="left", y=0.97,
    )
    fig.text(
        0.03, 0.905,
        "Every stage reads and writes real project artifacts under output/issue_helpdesk/ (see docs/process_flow.md "
        "for the exact function and file names); nothing here is illustrative only.",
        fontsize=8.6, color=GREY, ha="left",
    )

    row1_y, row2_y = 4.55, 1.55
    xs = [1.75, 4.9, 8.05, 11.2]

    for i, (number, title, desc) in enumerate(STAGES[:4]):
        _stage_box(ax, xs[i], row1_y, number, title, desc)
    row2_stages = list(reversed(STAGES[4:]))
    for i, (number, title, desc) in enumerate(row2_stages):
        _stage_box(ax, xs[i], row2_y, number, title, desc)

    for i in range(3):
        _arrow(ax, (xs[i] + 1.28, row1_y), (xs[i + 1] - 1.28, row1_y))
    _arrow(ax, (xs[3], row1_y - 0.78), (xs[3], row2_y + 0.78))
    for i in range(3):
        _arrow(ax, (xs[3 - i] - 1.28, row2_y), (xs[3 - i - 1] + 1.28, row2_y))

    fig.text(
        0.03, 0.03,
        "Output: technical_capstone.pptx / business_capstone.pptx (decks), final_capstone_report.md/.docx/.pdf "
        "(report), a Power BI ready export folder, and a FastAPI forecast proof of concept (src/app.py).",
        fontsize=8.3, color=GREY, ha="left",
    )

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_PATH, dpi=160, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {IMG_PATH}")


if __name__ == "__main__":
    build_diagram()
