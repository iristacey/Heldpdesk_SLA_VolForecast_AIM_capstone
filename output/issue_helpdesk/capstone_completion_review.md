# Capstone rubric evidence and remaining gaps

> **Repository update, 1 October 2026:** publication links refer to the new
> `Heldpdesk_SLA_VolForecast_AIM_capstone` repository. Runs #15 and #30 and the
> downloaded-snapshot check below were historical evidence from the previous
> repository and must not be attributed to the new one. The new repository's
> [run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719)
> passed at commit `744b0770a5426dbee5be86a657aed44127057d75` before this synchronized
> update. Its passing older tests do not verify the corrected JSON/contract checks;
> check the synchronized update's own CI run after upload.

**In summary:** this document is a self graded checklist against the course rubric. It lists, area by area, what evidence the project already has and what is still missing or unverified, so a reader can quickly see how complete the work is without re reading every notebook.

**Intended audience:** service managers and workforce/capacity planners, supported by technical reviewers who need traceable data preparation, model comparisons and limitations. Forecasting helps in identifying staffing requirements to address volume and demand, and retrospective review of performance surfaces opportunities to improve service based on SLA attainment. The project demonstrates an analytical workflow; it does not demonstrate that using its forecasts improves staffing costs or service outcomes.

> **Validation status:** all eight notebooks completed with the actual `.venv` kernel after the calendar correction. Earlier checks passed **39/39 tests in a clean LOCAL submission copy** using the existing `.venv`, without raw CSV, `mlflow.db`, or editor/cache/archive folders. That copy was not a Git clone, new environment or Docker/cloud CI run. The expanded local suite subsequently passed **52 tests without skips**. The project is now [published on GitHub](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone), and both lightweight Python and Docker CI jobs passed. Corrected validation selects **XGBoost at 7 days** and **ETS at 30 days**.

**Verified cloud evidence:** [Heldpdesk_SLA_VolForecast_AIM_capstone run #2](https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone/actions/runs/36838533719) passed at commit `744b0770a5426dbee5be86a657aed44127057d75`, before this synchronized update. The workflow includes skipped tests where dependencies or artifacts are absent; success is not evidence that all 52 tests executed and passed in each job. Full raw-data analysis reproduction in a fresh dependency environment, the full Docker `analysis` target and containerized API serving remain unverified. Confirm the new repository's own CI run after this update is uploaded.

An actual localhost HTTP demonstration also passed: health OK, seven dated ETS points starting 2023-03-15, and HTTP 422 for an invalid horizon. The temporary server was stopped; this is not a hosted or production deployment. Corrected technical/business decks have 12/10 slides, no off-slide shapes and no stale pre-correction warnings. The Docker test image ran on GitHub's runner; no local Docker execution is claimed.

All three manifests record `forecast_calendar_policy='day_after_history_end_v1'`; artifact/tuning checks reject incompatible baselines. Pandoc is installed at `%LOCALAPPDATA%\Pandoc\pandoc.exe` (a new terminal/PATH refresh may be needed). Technical-slide clipping is fixed and corrected visual exports have been regenerated without stale warnings. Final report DOCX/PDF have been regenerated from the corrected source using Pandoc and an isolated Edge profile; instructor review remains separate. Publication through the GitHub website does not require a local Git installation.

All active presentation files are under root `presentations`. The analysis, virtual environment, caches and local MLflow records are retained.

This self-review reflects the corrected local `issues.csv` run. It is not an instructor grade or a claim of production readiness.

**Reproduction and final review:** run 01–08 for a clean reproduction; with verified 01/02 inputs, run 03 before dependent 04–08 and exports. The earlier clean-copy run passed 39 tests; supplemental source, missing-date and ETS tests expanded the local suite to 52 passing tests without skips. Notebook 03/04 explanations and templates reflect corrected selection/tuning. The clean-copy tests verify packaged artifacts, not a raw-data rerun without inputs. Current final exports have been regenerated; future stale-result checks remain enforced.

**Supplemental evidence:** [source verification](../../data/source_verification.json) establishes direct byte equality to a fresh Kaggle v1 member and a digest/size match to the official Mendeley v2 listing. Direct Mendeley binary retrieval was blocked. The Kaggle CC0 label conflicts with original CC BY 4.0; original attribution must be retained. [Missing-date sensitivity](volume_forecast/coverage_sensitivity/summary.json) and [direct ETS diagnostics](volume_forecast/ets_diagnostics/ets_diagnostics_summary.json) are now available without changing canonical model selection.

**Source/objective and integration cross-check, 1 October 2026:** the raw
`issues.csv` digest still matches the retained source evidence. Recomputing the
scope and retrospective cohort gives 66,691 source rows, 27,348 scoped Tickets,
16,735 eligible completed Tickets and 3,333 reference-met Tickets. The regenerated
in-memory daily series exactly matches the saved 2,628-day series, including
344 assumed zero-arrival dates. These support the two stated workstreams:
retrospective SLA-reference assessment and daily Ticket-volume forecasting.
No classifier or alternative source dataset is part of either workstream.

[Previous-repository CI run #30](https://github.com/iristacey/SLA_Ticket_Forecast_Capstone_AIM/actions/runs/36827582947)
passed Python and lightweight Docker jobs at commit
`96136788fa09e20f6a7ca99e0904dd24f32a0811`. This is historical evidence from the
prior repository, not the new `Heldpdesk_SLA_VolForecast_AIM_capstone` location.
A downloaded copy of that exact commit
also passed all 52 tests without skips using the existing project interpreter,
without raw CSV, a copied virtual environment or local MLflow stores. Actual
localhost API calls returned correctly dated, finite, nonnegative ETS forecasts
for 1, 7, 30 and 90 days; default 30-day behavior, metadata and OpenAPI worked,
and invalid horizons returned HTTP 422. The temporary server was stopped.
This verifies the historical package's test/serving handoff, not fresh dependency
installation, full raw-data notebook reproduction or production readiness.

The cross-check found eight published presentation/source files still awaiting
the local narrative cleanup. It also found non-standard `NaN` values in five
JSON artifacts. The local JSON outputs now use `null` for intentionally
unevaluated validation interval coverage; measured values and model selection
are unchanged. Generator serialization rejects non-finite JSON, and existing
contract tests now check strict JSON plus the API's date/shape/range boundaries.
The corrected files must be uploaded and their own CI result confirmed before
claiming the published handoff is fully synchronized. See the
[integration contract and commands](../../docs/deployment_monitoring.md#integration-testing-handoff).

| Rubric area | Current evidence | Status / remaining gap |
|---|---|---|
| **1. Problem understanding & framing (10)** | Two explicit workstreams: retrospective priority-reference assessment and separate 7-/30-day daily Ticket-volume forecasting; research and proposed business criteria distinguished below. | **Academic framing evidenced.** Owner approval and measured impact are operational gates, not automatic academic deductions unless an explicit rubric descriptor or project claim requires them. |
| **2. Data collection & understanding (10)** | 66,691 records, 58-field profile/dictionary, quality checks, direct Kaggle v1 byte comparison and official Mendeley v2 digest/size match. | **Identity verified.** Source date-range discrepancy, completeness and conflicting mirror license label remain unresolved. Matching bytes does not certify privacy safety. |
| **3. Preprocessing, EDA & feature engineering (10)** | Scoped cohort, UTC rules, past-only features, training-only MI/PCA and causal missing-date sensitivity. | **Evidence strengthened.** 7-day validation ranking changes when zero-record targets are excluded. Completeness remains unknown; no canonical selection is replaced. |
| **4. Model implementation & comparison (20)** | Corrected comparisons, splits, intervals/tuning, primary ETS, MLflow and API checks; all eight notebooks executed locally, the expanded local suite passed 52 tests without skips, and lightweight Python/Docker cloud CI passed with skips. | **Met as a historical prototype.** XGBoost selected at 7 days; ETS at 30 days. Tuning ties baseline, not an improvement or general optimum. Operational validation and full-analysis reproduction in a fresh dependency environment remain unverified. |
| **5. Critical thinking, ethical AI & bias auditing (20)** | XGBoost explanations (SHAP, PDP, ICE, LIME) plus direct ETS components/residuals, sensitivity findings, Wilson intervals, operational-group audit, a named-and-justified mitigation-toolkit scope statement and human-review safeguards. | **Scope-limited evidence, not fairness certification.** Missing demographics, ROI and production are not automatic academic failures; explicit rubric requirements still apply. Residual dependence, underprediction and interval undercoverage are disclosed. |
| **6. Final presentation & communication (10)** | Corrected report/narrative and regenerated deck/poster exports; canonical technical/business decks have 12/10 slides, no off-slide shapes and no stale warnings. | **DOCX/PDF regenerated from corrected source.** Instructor review remains outstanding; instructor approval is not claimed. |
| **7. GitHub profile & upload (15)** | Public repository with notebooks, configuration, pipeline, tests, Docker/CI workflow, setup guide, POC, reports and presentations; successful lightweight Python/Docker CI run #15. Raw data remains excluded. | **Published; lightweight CI verified.** https://github.com/iristacey/Heldpdesk_SLA_VolForecast_AIM_capstone. Local validation passed 52 tests without skips. Check each new upload's CI result. Publication is not automatic full credit; instructor assessment and mirror-license resolution remain separate. |
| **Bonus (5)** | Corrected historical time-series work, PCA/selection, MLflow, locally tested API and dual-audience decks. | **Possible, not assured.** Depends on final demonstration, presentation quality and evaluator judgment. |

## Academic scope and assessment

**Missing demographic attributes, unmeasured ROI and lack of production deployment are not automatically academic failures.** Clear scope boundaries, supported operational-group audits, deliberately unfilled ROI assumptions and an honestly described local POC provide evidence of critical thinking and responsible AI practice and deserve consideration for credit. No fabricated demographics, business returns or deployment evidence to fill headings.

Deduct marks for absent evidence only where the actual rubric requires it or it is needed to support a project claim. Demographic fairness metrics, measured ROI and production deployment remain unevidenced if explicitly required. An executed lightweight test container is now evidenced, but it does not establish full-analysis or API deployment in a container; the exact descriptor matters. Stating a limitation is not a waiver or a promise of marks; the instructor's rubric governs. GitHub publication is a separately listed deliverable and is now established, with successful lightweight cloud CI.

## Confirmed citation and verified source identity

**Mohammad Abdellatif (2025), Help Desk Tickets, version 2**, published **30 May 2025**, DOI **10.17632/btm76zndnt.2**, **CC BY 4.0**, [Mendeley Data](https://data.mendeley.com/datasets/btm76zndnt/2). Local `issues.csv` now matches the Mendeley v2 published SHA-256 and size and a freshly downloaded [Kaggle](https://www.kaggle.com/datasets/janebitor/help-desk-tickets-mendeley-data) v1 member byte-for-byte. Provider coverage and the pre-2016 discrepancy remain open. Kaggle's CC0 label does not override the original attribution obligations; resolution of that mismatch remains unverified. Retain original attribution and review privacy before publishing any additional material.

## Supplemental findings

On common positive-record test targets, replacing zero-record training days with causal historical medians changes selected-model MAE from **5.1241 to 5.1781** (7-day XGBoost) and **4.7510 to 4.7897** (30-day ETS). Both still beat the seasonal baseline. However, excluding zero-record validation targets alone changes the 7-day ranking to ETS (**4.3398** versus XGBoost **4.4408**). This post-hoc population sensitivity is disclosed; canonical model selection is unchanged and no imputed labels are scored.

The primary ETS has final level **20.0965**, daily trend **0.000713**, and smoothing weights **0.04923 / 0 / 0.04373**. Direct state/forecast reconstruction and separate residual/test-error evidence now support its explanation. Fitted lag-1 residual ACF **0.1346** and exploratory Ljung-Box results indicate remaining dependence. Held-out 30-day signed bias is **+0.7087 tickets/day**, indicating underprediction. In-sample fit quality is not substituted for test performance.

## Corrected historical results

- Source CSV: 66,691 rows; 58 columns; exact Ticket records in the documented 2016+ scope: 27,348.
- Retrospective `Done`/known-priority/valid-duration assessment: 16,735 records; 3,333 (19.92%) met the project-configured references.
- Daily series: 2,628 calendar days from 2016-01-03 to 2023-03-14, with 344 zero-ticket dates (13.1%) treated as zero under an unverified completeness assumption.
- Seven-day backtest: 52 validation and 51 test origins. Validation selects XGBoost at MAE **4.181157**; test MAE **5.049222**, RMSE **7.722058**, WAPE **30.412893%**, sMAPE **49.034548%**, coverage **86.554622%**.
- ETS 7-day comparator: validation MAE **4.227266**, test MAE **4.758324**, RMSE **7.398095**, WAPE **28.660732%**, sMAPE **44.764730%**, coverage **87.394958%**. Lower test error does not override XGBoost's validation selection.
- Thirty-day daily-path backtest: 48 validation and 48 test origins, with overlapping weekly-spaced forecast windows. ETS test MAE 4.67 tickets/day, RMSE 7.20, WAPE 27.51%, sMAPE 41.33%; interval coverage 87.7%. Its test MAE ranged from 4.49 tickets/day for lead days 8–14 to 4.88 for days 22–30.
- XGBoost hyperparameter sensitivity (Notebook 04, current adopted baseline `n_estimators: 200`): one-step validation MAE across eight configuration variants ranges from 4.14 to 4.32 tickets/day, with the configured baseline as the single best-performing variant.
- Other corrected comparisons (validation/test MAE): Ridge 7-day **4.323507 / 5.217716**, Ridge 30-day **4.778375 / 5.454248**, XGBoost 30-day **4.684999 / 5.421653**. Primary ETS 30-day validation MAE is **4.393815**.
- Corrected XGBoost tuning: baseline and candidate validation MAE are equal at 7 days (**4.181157**) and 30 days (**4.684999**), yielding `keep_baseline`. This is a bounded no-improvement result, not proof of a general optimum. Earlier improvement claims from the invalid calendar baseline are superseded.

These corrected historical results do not establish current performance, approved SLA compliance or realized staffing value. The API serves **one persisted primary 30-day ETS artifact for every accepted horizon**. `horizon_days=7` does not load the validation-selected 7-day XGBoost; **no separate 7-day model is persisted**.

### Research and business acceptance criteria

**Research acceptance criterion:** validation-only selected models must have lower held-out MAE than the same-weekday baseline at both horizons, disclosing RMSE, bias and coverage. The corrected historical run meets this criterion (5.05 vs 6.41 at 7 days; 4.67 vs 6.49 at 30 days), not an owner-approved business acceptance.

**Proposed business success criterion (not measured or owner-approved):** a human-reviewed pilot targets **at least 10% fewer overtime hours per 100 tickets** versus a matched baseline without worsening an owner-approved service-attainment rate. Approve ticket/overtime/service-clock definitions, costs, pilot period, matching rules and stop conditions, and obtain current representative data first. Historical forecast baseline advantage does not demonstrate that business KPI.

### What "measure staffing/intervention outcomes and approved costs" means

This project provides corrected historical forecasting evidence; it does not prove the forecast changes anything in the real world. Calculating ROI (return on investment) means showing, with real numbers, that acting on the forecast produced a benefit greater than its cost. Concretely, that would require:

- **A pilot**: using the forecast for real, for a defined period, with a human still reviewing every decision (never fully automated) so mistakes are caught.
- **A baseline to compare against**: what staffing, response times, or missed-SLA counts looked like *before* the forecast was used, versus *after*.
- **Approved costs**: the actual cost of running the pipeline, any added staffing/scheduling changes, and the cost of a forecast error (extra staff scheduled unnecessarily, or a shortfall that delays tickets).
- **Approved benefit measures**: e.g., fewer missed SLA references, reduced overtime, faster resolution, agreed with whoever owns the budget.
