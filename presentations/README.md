# Capstone presentations


**In summary:** this is the single active folder for the capstone poster, technical and business decks, presentation charts, authored narrative, and presentation generators. Read `technical_presentation.md` for the full explanation in both technical and plain language.

This package presents the issues.csv capstone: retrospective priority-SLA reference analysis and separate 7- and 30-day daily Ticket-volume forecasting. The month-ahead output is a daily path, not one monthly total.

**Intended audience:** service managers and workforce/capacity planners, supported by technical reviewers who need traceable data preparation, model comparisons and limitations. Forecasting helps in identifying staffing requirements to address volume and demand, and retrospective review of performance surfaces opportunities to improve service based on SLA attainment. The project demonstrates an analytical workflow; it does not demonstrate that using its forecasts improves staffing costs or service outcomes.

> **Validation status:** all eight notebooks executed using the actual `.venv` kernel after the calendar correction. Earlier checks passed **39/39 tests in a clean LOCAL submission copy** using the existing `.venv`, without raw CSV, `mlflow.db`, or editor/cache/archive folders. That copy was not a Git clone or Docker/cloud CI run. The expanded local suite subsequently passed **52 tests without skips**. Corrected validation selects **XGBoost at 7 days** (test MAE 5.049222) and **ETS at 30 days** (test MAE 4.672305); lower ETS 7-day test error does not override validation-only selection. The project is [published on GitHub](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone).

All three manifests record `forecast_calendar_policy='day_after_history_end_v1'`; stale artifact/tuning checks remain enforced. The earlier SciPy DLL block no longer reproduces without any policy bypass. Pandoc is installed; browser-based publication does not require a local Git installation. Corrected deck/poster exports have been regenerated without stale warnings and technical-slide clipping is fixed. Final report DOCX/PDF have been regenerated from the corrected source; instructor review remains separate.

**Verified cloud CI:** [Heldpdesk_SLA_VolForecast_AIM_capstone run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719) passed at commit `744b0770a5426dbee5be86a657aed44127057d75`, before this synchronized update. Lightweight CI includes skipped tests; it does not mean all 52 tests executed and passed in each job. Docker ran on GitHub's runner, not locally. Full raw-data analysis reproduction in a fresh dependency environment, the full Docker `analysis` target and containerized API serving remain unverified. Confirm the new repository's own CI run after this update is uploaded.

**Serving scope:** the API serves **one persisted primary 30-day ETS artifact for all accepted horizons**. `horizon_days=7` does not load the validation-selected 7-day XGBoost: **no separate 7-day model is persisted**. Distinguish horizon-specific experimental selection from the single-model serving demonstration.

**Verified delivery checks:** the canonical technical/business decks contain **12/10 slides**, with no off-slide shapes or stale pre-correction warnings. An actual localhost HTTP demonstration also passed: health OK, seven dated ETS points starting 2023-03-15, and HTTP 422 for an invalid horizon. The temporary server was stopped; it is not a hosted or production deployment. Final report DOCX/PDF have been regenerated from the corrected source using Pandoc and an isolated Edge profile; instructor approval is not claimed.

Dataset listing: [Help Desk Tickets (Mendeley Data), Kaggle](https://www.kaggle.com/datasets/janebitor/help-desk-tickets-mendeley-data). Original citation: **Mohammad Abdellatif (2025), Help Desk Tickets, version 2**, published **30 May 2025**, DOI **10.17632/btm76zndnt.2**, **CC BY 4.0**, [Mendeley Data](https://data.mendeley.com/datasets/btm76zndnt/2). Local `issues.csv` now matches the downloaded Kaggle v1 bytes and official Mendeley v2 digest/size; see [verification evidence](../data/source_verification.json). Completeness and the date-range discrepancy remain unresolved. Kaggle's CC0 label conflicts with original CC BY 4.0; retain original attribution and resolve the mirror label. Matching bytes does not certify privacy safety.

Supplemental material: [missing-date sensitivity](../output/issue_helpdesk/volume_forecast/coverage_sensitivity/summary.json) and [direct ETS diagnostics](../output/issue_helpdesk/volume_forecast/ets_diagnostics/ets_diagnostics_summary.json). Present the finding that excluding zero-record targets changes the 7-day validation ranking, while the canonical selection remains unchanged. Distinguish ETS fitted residuals from held-out errors.

Missing demographics, measured ROI or production deployment are not automatically academic failures: transparent scope limits and supported audits deserve consideration for credit. Explicit rubric requirements and the evidence needed for project claims still govern; disclosure is not a waiver. See the [full assessment explanation](../README.md#academic-scope).

- `technical_capstone.pptx` / `business_capstone.pptx` — generated technical and business decks.
- `technical_presentation.pptx` — packaged copy of the technical deck.
- `technical_presentation.md` — narrative, definitions, metrics and caveats.
- `technical_slides.md` — condensed, slide-formatted derivative of
  `technical_presentation.md` (12 slides), used only as the source for the
  reveal.js/Beamer export below; it does not replace or alter the
  `.pptx` decks or the narrative file.
- `technical_reveal_deck.html` / `technical_reveal_deck_beamer.tex` —
  lightweight peer-deck export in the rubric-suggested "Jupyter slides /
  LaTeX Beamer" formats, generated from `technical_slides.md` by
  `src/generate_technical_reveal_deck.py`. The `.html` is a self-contained
  reveal.js deck (open directly in a browser; loads the reveal.js
  library from its CDN at view time). The `.tex` is valid Beamer source
  provided as-is — no LaTeX engine is installed in this environment to
  compile it to PDF.
- `capstone_poster.png` / `capstone_poster.pdf` — single-page, 12 x 18 inch infographic with seven numbered sections, original line icons, navy/teal/gold styling, and dedicated deployment and responsible-use panels. The PNG is 300 dpi; the PDF preserves vector text and graphics for printing. Regenerated by `src/generate_poster.py` from the same canonical `output/issue_helpdesk` artifacts, without retraining.
- `generate_presentation.py` — regenerates both PowerPoint decks and the daily-volume chart.
- `daily_ticket_counts.csv` / `daily_ticket_counts.png` — scoped daily series and monthly view.
- `run_volume_forecast.py` — executes configured expanding-window 7- and 30-day forecast experiments.
- Canonical forecasts, split metrics, model selection and experiment metadata live under `output/issue_helpdesk/volume_forecast/`; the presentation generator writes its packaged outputs here, including `volume_forecast_experiment.json`.

The superseded archived deck and obsolete presentation directories have been removed. The files listed above under root `presentations` are the active package.

From the project root:

```powershell
python presentations\run_volume_forecast.py
python -m src.generate_coverage_sensitivity
python -m src.generate_ets_diagnostics
python presentations\generate_presentation.py
python src\generate_poster.py
```

The full end-to-end process-flow diagram is in [`docs/process_flow.md`](../docs/process_flow.md).

Run notebooks 01–08 in order to reproduce all analysis and reporting stages. Setup instructions and the source/coverage limitations are in the project-root [README](../README.md).

**Refresh order:** with verified Notebook 01/02 inputs, run **03**, then dependent **04–08**, regenerate supplemental diagnostics, review authored metrics and regenerate exports. For a clean reproduction or changed inputs, start at 01. Earlier notices were removed after the successful 39-test clean-copy verification; supplemental source, sensitivity and ETS tests were added afterward. Notebook 03/04 explanations and templates reflect corrected selection/tuning. Future stale-result checks remain enforced; do not manually relabel old results.

The target is daily counts of exact `issue_type == Ticket` records created on/after the documented 2016-01-01 UTC scope start. The file includes earlier records, which are excluded and flagged. The history has 2,628 days through 2023-03-14; 344 dates have no Ticket rows and are treated as zero arrivals under an unverified completeness assumption. Separate 7- and 30-day rolling forecasts use 365-day validation and test periods. Current historical results are not a forecast for today's operations.

The SLA thresholds and priority mapping are project-configured analytical assumptions, not approved policy. Resolution duration is UTC elapsed wall-clock time, not a verified service SLA clock. The technical and business presentations describe these limits and do not claim contractual compliance, protected-group fairness, staffing impact or ROI.

**Research criterion:** the corrected run meets the historical baseline-advantage criterion at both horizons (selected-model test MAE 5.05 vs 6.41 at 7 days; 4.67 vs 6.49 at 30 days). **Proposed business criterion, not measured or owner-approved:** a human-reviewed pilot targets at least **10% fewer overtime hours per 100 tickets** versus a matched baseline without worsening an owner-approved service-attainment rate, contingent on approved definitions, costs, comparison design and current data. Forecast baseline advantage does not establish that business KPI.

The shared retrospective baseline is 3,333 of 16,735 eligible completed Tickets (19.9%) meeting the configured reference: Critical 24.7%, High 18.3%, Medium 19.4%, and Low 23.7%. Eligibility and counts are defined in the technical narrative and Notebook 02.
