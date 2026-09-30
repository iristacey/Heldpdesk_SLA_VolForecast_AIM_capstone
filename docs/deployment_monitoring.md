# Forecast integration and monitoring readiness

**In summary:** this document is a checklist for what would need to happen before this forecasting work could be trusted to run for real, in daily operations, rather than as a historical research exercise. In short: confirm the data is complete and current, get business sign off on the rules being used, and test it on a genuinely future period before relying on it.

## Current status

The project is an offline historical analysis of `data/raw/issues.csv` with a local FastAPI POC in [`src/app.py`](../src/app.py), exposing `/health`, `/model/metadata` and `/forecast`. Both `TestClient` and an actual localhost HTTP demonstration passed: health OK, seven dated ETS points starting 2023-03-15, and HTTP 422 for an invalid horizon. The temporary server was stopped; no running endpoint or hosted service is claimed. It does not include a scheduled feed, current arrivals, automated staffing decisions, or a validated SLA policy. Separate 7- and 30-day rolling-origin experiments use exact `issue_type == Ticket` rows in the documented 2016–2023 range. The 30-day horizon predicts daily counts for each lead day, not one monthly total. The source contains pre-2016 creation dates contrary to its documentation and 344 scoped dates with no Ticket rows; zeros are assumed, not independently verified.

**Local validation complete:** all eight notebooks executed with the actual `.venv` kernel. Final checks passed 39/39 tests in a clean LOCAL submission copy using the existing `.venv`, without raw CSV, `mlflow.db`, or editor/cache/archive folders. This was not a Git clone, new environment or Docker/cloud CI run. Corrected validation selects XGBoost at 7 days (validation/test MAE 4.181157/5.049222) and ETS at 30 days (4.393815/4.672305). ETS's lower 7-day test error does not override selection. The earlier DLL block no longer reproduces without policy bypass; GitHub publication is not complete.

All three manifests use `forecast_calendar_policy='day_after_history_end_v1'`; artifact and tuning checks reject incompatible baselines. Corrected tuning ties the configured baseline at both horizons (`keep_baseline`), not an improvement or general optimum. The API serves **one persisted primary 30-day ETS artifact for every accepted horizon**. Requesting `horizon_days=7` does not load the validation-selected 7-day XGBoost: **no separate 7-day model is persisted**. Pandoc is installed. Git installation was cancelled at the admin prompt; Git readiness is not established.

The original source metadata is confirmed: **Mohammad Abdellatif, Help Desk Tickets, version 2, 30 May 2025, DOI 10.17632/btm76zndnt.2, CC BY 4.0**, [Mendeley Data](https://data.mendeley.com/datasets/btm76zndnt/2). Local `issues.csv` identity and coverage are unresolved. Public availability alone neither proves a content match nor authorizes uploading the local file without applicable-rights, attribution and privacy review.

The priority/SLA analysis is retrospective only. Its thresholds and mapping are carried forward from the prior project and must be approved by service owners. Its duration is UTC elapsed wall-clock hours, not a validated business-hours clock.

## Forecast data contract

- **Input:** complete issue records with `issue_type == Ticket` and a valid `issue_created` timestamp.
- **Grain:** one UTC calendar day, including confirmed zero-arrival dates only when extract completeness has been established.
- **Target:** count of created Ticket records, not individual SLA risk.
- **Horizon:** 7 and 30 days; rolling origins are spaced weekly. Select and evaluate a separate model for each horizon.
- **Outputs:** actual, forecast, origin, split, horizon, model, validation-calibrated interval and source/config metadata.
- **Candidate methods:** same-weekday-last-week baseline, weekly ETS, XGBoost autoregression, and feature-selected/PCA Ridge.
- **Selection:** validation MAE only. Report test MAE, RMSE, WAPE, sMAPE, signed bias and interval coverage separately.

## Before a pilot

No Docker CLI is available, so the local HTTP check is not container validation. Git installation was cancelled at the administrator prompt; no `.git`, remote or publication was created.

The corrected historical run and local API tests are complete, but a pilot still needs current-data validation and deployment verification in its target environment. Verify actual `/forecast` responses with consecutive daily dates; a metadata-only `/health` response is not proof of successful model loading or inference.

1. Resolve the published-versus-observed source date-range mismatch and verify whether zero-record dates mean true zero arrivals or incomplete extraction.
2. Confirm the dataset's provenance and use rights, UTC/day-boundary semantics, and whether its historical projects represent the target operation.
3. Have service owners approve the priority mapping, reference targets, resolution eligibility and SLA-clock definition.
4. Agree the forecast horizon, business tolerances and useful capacity decisions before assessing a new future-period holdout.
5. Obtain current, representative data; test the selected method against the same-weekday baseline and monitor performance through time.
6. If operational segments become available, assess under/over-forecast bias, error and interval coverage by approved queue/site/shift, with small-group uncertainty.
7. Run a human-reviewed pilot, record planner overrides, and measure service outcomes and effort against a credible comparison before any ROI claim.

**Research criterion:** both validation-selected models have lower corrected historical test MAE than the seasonal baseline (5.05 vs 6.41 at 7 days; 4.67 vs 6.49 at 30 days), with bias/coverage disclosed. **Proposed business target (not measured or owner-approved):** a human-reviewed pilot should deliver **at least 10% fewer overtime hours per 100 tickets** than a matched baseline without worsening an owner-approved service-attainment rate. Approve denominator, overtime and service-clock definitions, costs, matched comparison period, stop conditions and current-data coverage first. Historical baseline advantage does not establish that business KPI.

## Monitoring and stop conditions

Monitor daily input row counts, source gaps, schema changes, weekly/monthly signed bias, MAE/RMSE/WAPE, interval coverage, calendar effects and sustained error by approved operational segment. Revisit evaluation when demand, queue mix, workflow or intake rules change. Pause use if source completeness is uncertain, error exceeds pre-agreed tolerances, under-forecasting persists, or intervals under-cover; investigate before retraining.

The current historical monitoring table is a demonstration of these checks, not a live alerting system. No protected demographic fields are available, so protected-group fairness cannot be certified. No source-derived ROI is calculated.
