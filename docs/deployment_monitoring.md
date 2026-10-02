# Forecast integration and monitoring readiness

> **Repository update, 1 October 2026:** publication links refer to the new
> `Heldpdesk_SLA_VolForecast_AIM_capstone` repository. Run #17 and the tested
> snapshot `96136788fa09e20f6a7ca99e0904dd24f32a0811` below belong to the previous
> repository and must not be attributed to the new one. The new repository's
> [run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719)
> passed before this synchronized update; check the new run after uploading it.

**In summary:** this document is a checklist for what would need to happen before this forecasting work could be trusted to run for real, in daily operations, rather than as a historical research exercise. In short: confirm the data is complete and current, get business sign off on the rules being used, and test it on a genuinely future period before relying on it.

**Intended audience:** service managers and workforce/capacity planners, supported by technical reviewers who need traceable data preparation, model comparisons and limitations. Forecasting helps in identifying staffing requirements to address volume and demand, and retrospective review of performance surfaces opportunities to improve service based on SLA attainment. The project demonstrates an analytical workflow; it does not demonstrate that using its forecasts improves staffing costs or service outcomes.

## Current status

The project is an offline historical analysis of `data/raw/issues.csv` with a local FastAPI POC in [`src/app.py`](../src/app.py), exposing `/health`, `/model/metadata` and `/forecast`. Both `TestClient` and an actual localhost HTTP demonstration passed: health OK, seven dated ETS points starting 2023-03-15, and HTTP 422 for an invalid horizon. The temporary server was stopped; no running endpoint or hosted service is claimed. It does not include a scheduled feed, current arrivals, automated staffing decisions, or a validated SLA policy. Separate 7- and 30-day rolling-origin experiments use exact `issue_type == Ticket` rows in the documented 2016–2023 range. The 30-day horizon predicts daily counts for each lead day, not one monthly total. The source contains pre-2016 creation dates contrary to its documentation and 344 scoped dates with no Ticket rows; zeros are assumed, not independently verified.

**Local validation complete:** all eight notebooks executed with the actual `.venv` kernel. Earlier checks passed 39/39 tests in a clean LOCAL submission copy using the existing `.venv`, without raw CSV, `mlflow.db`, or editor/cache/archive folders. That copy was not a Git clone, new environment or Docker/cloud CI run. The expanded local suite subsequently passed 52 tests without skips. Corrected validation selects XGBoost at 7 days (validation/test MAE 4.181157/5.049222) and ETS at 30 days (4.393815/4.672305). ETS's lower 7-day test error does not override selection. The earlier DLL block no longer reproduces without policy bypass.

**Publication and cloud CI:** the project is [published on GitHub](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone). [Run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719) passed at commit `744b0770a5426dbee5be86a657aed44127057d75`, before this synchronized update. Lightweight CI includes skipped tests; it does not mean all 52 tests executed and passed in each job. The Docker test image ran on GitHub's runner; no local Docker execution is claimed. Confirm the new repository's own CI run after this update is uploaded.

All three manifests use `forecast_calendar_policy='day_after_history_end_v1'`; artifact and tuning checks reject incompatible baselines. Corrected tuning ties the configured baseline at both horizons (`keep_baseline`), not an improvement or general optimum. The API serves **one persisted primary 30-day ETS artifact for every accepted horizon**. Requesting `horizon_days=7` does not load the validation-selected 7-day XGBoost: **no separate 7-day model is persisted**. Pandoc is installed. Browser-based publication does not require a local Git installation.

The original source metadata is confirmed: **Mohammad Abdellatif, Help Desk Tickets, version 2, 30 May 2025, DOI 10.17632/btm76zndnt.2, CC BY 4.0**, [Mendeley Data](https://data.mendeley.com/datasets/btm76zndnt/2). Local `issues.csv` matches the downloaded Kaggle v1 member byte-for-byte and the official Mendeley v2 digest/size; see [source verification](../data/source_verification.json). Direct Mendeley binary retrieval was blocked. Coverage and the conflicting Kaggle CC0 label remain unresolved. Verified identity does not authorize uploading the local file without applicable-rights, attribution and privacy review.

The priority/SLA analysis is retrospective only. Its thresholds and mapping are project-configured references and must be approved by service owners before operational use. Its duration is UTC elapsed wall-clock hours, not a validated business-hours clock.

## Historical backtest data contract

- **Input:** complete issue records with `issue_type == Ticket` and a valid `issue_created` timestamp.
- **Grain:** one UTC calendar day, including confirmed zero-arrival dates only when extract completeness has been established.
- **Target:** count of created Ticket records, not individual SLA risk.
- **Horizon:** 7 and 30 days; rolling origins are spaced weekly. Select and evaluate a separate model for each horizon.
- **Outputs:** actual, forecast, origin, split, horizon, model, validation-calibrated interval and source/config metadata.
- **Candidate methods:** same-weekday-last-week baseline, weekly ETS, XGBoost autoregression, and feature-selected/PCA Ridge.
- **Selection:** validation MAE only. Report test MAE, RMSE, WAPE, sMAPE, signed bias and interval coverage separately.

## Integration-testing handoff

The package supports controlled integration testing against the historical `issues.csv`
artifacts, not a live operational rollout. A downloaded snapshot of GitHub commit
`96136788fa09e20f6a7ca99e0904dd24f32a0811` passed all 52 tests without skips
using the existing project interpreter on 1 October 2026. The temporary copy had
no raw CSV, local virtual environment or local MLflow store. This is package
reproduction with existing dependencies, not a fresh dependency installation or
a rerun of notebook training.

The integration check found non-standard `NaN` values for unmeasured validation
interval coverage in five JSON artifacts. The corrected local artifacts use JSON
`null` for that field, and the generator rejects other non-finite JSON values.
`null` means coverage was not evaluated on the calibration split, not zero coverage.
Test-set coverage and all measured forecast metrics are unchanged. Upload these
corrected artifacts and source before consuming the JSON in strict parsers.

### API contract

The API requires `models/final_volume_forecast_model.joblib`, its adjacent
`final_volume_forecast_model_metadata.json`, and
`output/issue_helpdesk/volume_forecast/daily_ticket_counts.csv`, along with the
source package and its dependencies. It does not accept raw ticket uploads or
new history through its HTTP interface. Load only the trusted project artifact;
restart the process after replacing model/history files because they are cached.

From the project root with the analysis dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_project_contracts -v
.\.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health'
Invoke-RestMethod 'http://127.0.0.1:8000/model/metadata'
Invoke-RestMethod 'http://127.0.0.1:8000/forecast?horizon_days=30'
```

Stop the local server with Ctrl+C after testing. API documentation is available
at `http://127.0.0.1:8000/docs`.

| Surface | Expected contract |
|---|---|
| `/health` | Reports metadata availability; not sufficient proof that inference works. |
| `/model/metadata` | Identifies the persisted primary 30-day ETS model, training dates and limitations. |
| `/forecast` | Defaults to 30; accepts integer `horizon_days` from 1 through 90. The same ETS artifact serves every accepted horizon, including 7 days. |
| Successful forecast | Contains `model_name`, `horizon_days`, `history_ends`, `trained_on_date_end`, `forecast`, `limitations` and `caveat`. Each point contains `date` and numeric `forecast_tickets`. |
| Dates | Exactly one consecutive daily point per requested lead, starting 2023-03-15 for the published history ending 2023-03-14. These are not today's forecasts. |
| Invalid horizon | HTTP 422 for 0, negative values, values above 90, nonnumeric text and fractional values. |
| Unavailable required files | HTTP 503 when the needed model, metadata or history file is missing. Restore the package artifacts before retrying. |

Actual localhost HTTP checks on the downloaded snapshot covered 1, 7, 30 and
90 days, default behavior, metadata, OpenAPI and the invalid values listed above.
All returned the expected status and shape; valid predictions were finite and
nonnegative. The temporary server was stopped. Accepting 90 days is an interface
capability, not evidence of 90-day backtest performance.

API forecasts are **point estimates only**: they do not return actual outcomes,
backtest origins/splits or calibrated intervals. The backtest CSV contract above
and the API response contract are different. Seven-day XGBoost experimental
selection does not imply an XGBoost serving endpoint.

## Before a pilot

Successful lightweight Docker CI does not validate the full analysis or a containerized API. Full raw-data analysis reproduction in a fresh dependency environment, the full Docker `analysis` target and containerized API serving remain unverified. The local HTTP demonstration is separate from the cloud test-container run.

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
