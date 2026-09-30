"""Execute the 7- and 30-day rolling-origin forecasts for scoped Ticket issues."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.issues_capstone import run_data_understanding, run_volume_forecast


def main():
    source_results = run_data_understanding()
    print(
        f"Source rows: {source_results['source_rows']:,}; "
        f"scoped Ticket rows: {source_results['ticket_rows']:,}; "
        f"SLA cohort: {source_results['sla_assessed_rows']:,}"
    )
    metadata = run_volume_forecast()
    print(
        f"Forecast dates: {metadata['date_start']} to {metadata['date_end']}; "
        f"horizons: {metadata['evaluation_horizons_days']}; "
        f"selected models: {metadata['selected_models_by_horizon']}"
    )


if __name__ == "__main__":
    main()
