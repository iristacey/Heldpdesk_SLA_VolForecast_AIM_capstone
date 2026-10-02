# Project data

> **Repository update, 1 October 2026:** publication links refer to the new
> `Heldpdesk_SLA_VolForecast_AIM_capstone` repository. Run #17 was historical
> evidence from the previous repository. The new repository's
> [run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719)
> passed before this synchronized update; check the new run after uploading it.

**In summary:** this folder holds the one raw data file the whole project is built on, a list of help desk tickets, and explains what we do and do not know about where it came from. We only analyze the part of the file that matches its documented time period, and we are upfront that a few details, like whether every single day is fully represented, still need confirmation from the data provider.

**Intended audience:** service managers and workforce/capacity planners, supported by technical reviewers who need traceable data preparation, model comparisons and limitations. Forecasting helps in identifying staffing requirements to address volume and demand, and retrospective review of performance surfaces opportunities to improve service based on SLA attainment. The project demonstrates an analytical workflow; it does not demonstrate that using its forecasts improves staffing costs or service outcomes.

## Primary source

The listed dataset is [Help Desk Tickets (Mendeley Data), published on Kaggle](https://www.kaggle.com/datasets/janebitor/help-desk-tickets-mendeley-data). The original [Mendeley Data version-2 page](https://data.mendeley.com/datasets/btm76zndnt/2), consulted on 30 September 2026, identifies **Mohammad Abdellatif (2025), Help Desk Tickets, version 2**, DOI **10.17632/btm76zndnt.2**, published 30 May 2025 under **CC BY 4.0**. See the [main README citation](../README.md#acknowledgment-and-citation).

The working extract is `data/raw/issues.csv` (66,691 rows and 58 fields). On 30 September 2026 it was verified byte-for-byte against the downloaded **Kaggle version-1** member. Its **20,586,538 bytes** and SHA-256 `eab3b8857bc19bd917e0767fb9c30b7714d5468b25c5a1c1cd71a760e9602f02` also match the official **Mendeley version-2** file listing. See [source_verification.json](source_verification.json) for the public endpoints, identifiers, comparisons and limitations. Direct Mendeley binary retrieval returned HTTP 403; that comparison uses the publisher's digest, not a separately downloaded Mendeley CSV.

**Mirror license discrepancy:** Kaggle metadata says **CC0: Public Domain**, whereas the original Mendeley release specifies **CC BY 4.0**. Preserve the original attribution, license link and indication of changes; the mirror label does not establish authority to remove those obligations. Resolve the inconsistent Kaggle label before claiming CC0. Identity does not establish privacy safety or permission for third-party elements.

The supplied source notes describe `issues.csv` as covering January 2016–March 2023 and document workflow-duration and change-history fields. However, `issue_created` in the local file ranges from 2007-04-15 to 2023-03-14. This discrepancy is reported in project outputs; the primary analysis follows the documented 2016-01-01 start and excludes earlier records.

The raw file remains excluded from Git (`data/raw/*`). Its verified identity does not make raw upload automatically safe. Retain attribution and review privacy, third-party rights and row-level disclosure risks. The date-range mismatch is present in the identified public content; neither the public `FEATURES.md` nor `EXAMPLE.md` establishes its cause. Do not assume migration, timestamp shifting or complete extraction without provider evidence.

From the project root, recheck the current file without modifying it:

```powershell
.\.venv\Scripts\python.exe -m src.verify_dataset_source
.\.venv\Scripts\python.exe -m src.verify_dataset_source --download-kaggle
```

The first command compares with the retained official digest/size. The second also downloads the pinned public Kaggle archive into memory and checks direct byte equality. A mismatch raises an error; neither command uploads local data.

## Analysis scope

The primary ticket cohort includes exact `issue_type == "Ticket"` records created in the documented 2016+ range. Broader issue types (such as Story, Service, Subtask and Vacation) are summarized for context but are not counted as arrivals in the core ticket-volume target.

The daily forecast counts `issue_created` timestamps by UTC calendar day. There are 2,628 calendar dates in the scoped series, including 344 (13.1%) dates with no matching Ticket record. They are zero-filled for the experiment under the assumption that this is a complete issue-record extract. The file alone cannot prove those days are fully covered; verify with the data provider before treating zeros as known demand.

The [missing-date sensitivity evidence](../output/issue_helpdesk/volume_forecast/coverage_sensitivity/summary.json) now tests a causal alternative training history and compares errors on common positive-record targets. Selected-model test MAEs change from 5.1241 to 5.1781 (7-day XGBoost) and 4.7510 to 4.7897 (30-day ETS). Excluding zero-record validation targets changes the 7-day ranking to ETS even before imputation, so that selection is population-sensitive. This post-hoc check does not prove completeness, score invented labels, or replace the canonical experiment. See the [README method and limitations](../README.md#missing-date-sensitivity).

## Priority-based retrospective reference assessment

The analysis uses project-configured thresholds with an explicit mapping: `Blocker`/`Highest` → Critical (4h), `High` → High (8h), `Medium` → Medium (24h), `Low`/`Lowest` → Low (24h), and `unknown` excluded. The primary cohort also requires resolution outcome `Done`, a mapped priority, and valid nonnegative duration. Duration is computed as `issue_resolution_date - issue_created` in elapsed UTC wall-clock hours.

This duration is not verified to follow a business-hours calendar, pause rules, or the service's actual SLA clock. The mapping and thresholds are not policy-validated. The `wf_total_time` field is a workflow-derived diagnostic, not a substitute SLA duration. The historical comparison currently includes 16,735 records, with 3,333 meeting the configured references. This is not contractual compliance evidence.

## Data structure

The raw source (`data/raw/issues.csv`) is one flat table, 66,691 rows by 58
columns. Grouping the columns by role:

| Group | Example columns | Role |
|---|---|---|
| Identity/keys | `id`, `issue_num`, `issue_proj` | Unique record and (masked) project/issue references. |
| Actors | `issue_reporter`, `issue_assignee`, `issue_contr_count` | Masked user identifiers; `issue_assignee` is 46.55% missing. |
| Classification | `issue_type`, `issue_priority`, `issue_status`, `issue_resolution` | Ticket type, priority, workflow status, and resolution outcome; the primary cohort filters `issue_type == "Ticket"` and `issue_resolution == "Done"`. |
| Timestamps | `started`, `ended`, `issue_created`, `issue_resolution_date`, `last_change_date` | Parsed as UTC; `issue_created` drives the daily forecast series and `issue_resolution_date` drives the SLA reference duration. 853 rows have no resolution date. |
| Workflow-state durations | `wf_in_review`, `wf_in_progress`, `wf_done`, ... (16 `wf_*` fields) | Elapsed seconds spent in each named workflow state; mostly sparse (many states apply to only a small fraction of tickets), kept only as diagnostics, never substituted for the SLA clock. |
| Workflow-state transition counts | `wfe_in_review`, `wfe_in_progress`, ... (16 `wfe_*` fields) | Count of times an issue passed through each state; always populated. |
| Process summary | `wf_total_time`, `processing_steps`, `issue_comments_count` | Total workflow processing time and step/comment counts; diagnostic only. |

See `output/issue_helpdesk/data_dictionary.csv` for the complete, generated
58-row field-by-field reference (dtype, rows processed, present/missing
counts, unique-value counts and a one-line meaning/caveat for every column).

**Engineered structure used for forecasting** (built only from `issue_created`
day counts, never from the raw table directly, so no leakage of same-day
fields into the target): one row per **calendar day** (2,628 rows total),
with these columns constructed by `src/issues_capstone.py`:

| Feature group | Columns | Notes |
|---|---|---|
| Target | `ticket_count` | Count of scoped Ticket records created that UTC day; 0 for the 344 zero-ticket days. |
| Lag features | `lag_1` ... `lag_28` | The prior 28 days' counts, computed only from history strictly before the target date. |
| Rolling means | `mean_7`, `mean_28` | 7- and 28-day trailing averages, same past-only rule. |
| Calendar encodings | `weekday_sin`, `weekday_cos`, `year_sin`, `year_cos` | Cyclical day-of-week and day-of-year encodings so the model can learn weekly/annual seasonality without a discontinuity at the week or year boundary. |

This engineered table (34 feature columns plus the target) is what feeds the
rolling-origin backtest, the mutual-information feature selection, PCA, and
the persisted final model; it is intentionally separate from, and much
narrower than, the 58-field raw source table above.

## Outputs

The local API was also verified over actual localhost HTTP: health OK, seven dated points from the single persisted ETS artifact starting 2023-03-15, and HTTP 422 for an invalid horizon. The temporary server was stopped. This is not a live data feed or current-demand validation. The project is [published on GitHub](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone); browser-based publication does not require a local Git installation.

**Result status:** all eight notebooks completed using the actual `.venv` kernel after the calendar correction. Earlier checks passed 39/39 tests in a clean LOCAL submission copy using the existing `.venv`, without raw CSV, `mlflow.db`, or editor/cache/archive folders. That copy verified packaged artifacts, not a raw-data rerun, Git clone, new environment or Docker/cloud CI run. The expanded local suite subsequently passed 52 tests without skips. All three manifests use `forecast_calendar_policy='day_after_history_end_v1'`. Validation selects XGBoost at 7 days and ETS at 30 days, never by test rank. These are corrected historical results, not current-service evidence or proof of source completeness.

**Verified cloud CI:** [Heldpdesk_SLA_VolForecast_AIM_capstone run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719) passed at commit `744b0770a5426dbee5be86a657aed44127057d75`, before this synchronized update. Lightweight CI includes skipped tests; it does not mean all 52 tests executed and passed in each job. Docker ran on GitHub's runner, not locally. Full raw-data analysis reproduction in a fresh dependency environment, the full Docker `analysis` target and containerized API serving remain unverified. Confirm the new repository's own CI run after this update is uploaded.

Aggregated, reproducible outputs are written to the canonical `output/issue_helpdesk/` folder (see `output/README.md`). `dataset_manifest.json`, `source_quality_summary.csv`, `daily_coverage_audit.csv`, `data_dictionary.csv`, and `resolution_duration_audit.csv` document scope, assumptions and quality. Forecast predictions and metrics are in `output/issue_helpdesk/volume_forecast/`. No row-level export with raw reporter, assignee or project identifiers is required for the capstone reports.

File identity is verified; real-world generalization still requires complete coverage, approved SLA definitions and current-period performance evidence.
