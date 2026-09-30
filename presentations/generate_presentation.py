"""Refresh the dedicated technical deck and its daily-arrival chart.

Decks, charts, the poster, generator scripts and authored narrative share the
project-root `presentations/` folder. Analysis inputs remain under `output/`.
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_capstone_decks import OUTPUT, build_all
from src.issues_capstone import REPORTS


def main():
    build_all()
    output_dir = OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    source_deck = output_dir / "technical_capstone.pptx"
    shutil.copyfile(source_deck, output_dir / "technical_presentation.pptx")
    shutil.copyfile(
        REPORTS / "volume_forecast_experiment.json",
        output_dir / "volume_forecast_experiment.json",
    )

    daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
    daily.to_csv(output_dir / "daily_ticket_counts.csv", index=False)
    monthly = daily.assign(
        month=lambda frame: pd.to_datetime(frame["date_created"], utc=True).dt.to_period("M").astype(str)
    ).groupby("month", as_index=False)["tickets"].sum()
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(monthly["month"], monthly["tickets"], color="#3470a8", linewidth=1.4)
    axis.set_title("Monthly created Ticket records (historical; not a forecast)")
    axis.set_ylabel("Ticket count")
    axis.set_xlabel("UTC month")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "daily_ticket_counts.png", dpi=160)
    plt.close(figure)
    print(f"Refreshed {output_dir / 'technical_presentation.pptx'}")


if __name__ == "__main__":
    main()