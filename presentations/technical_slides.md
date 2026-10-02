---
title: "Help Desk Ticket SLA and Volume Forecasting"
subtitle: "Technical presentation — peer/reviewer deck"
author: "SLA & Volume Forecast Capstone"
date: "Historical analysis, 2016–2023"
theme: "white"
transition: "slide"
---

# Help Desk Ticket SLA and Volume Forecasting

Technical peer deck — reveal.js / Beamer-style export

**Source of truth:** this deck is a condensed, slide-formatted derivative of
`presentations/technical_presentation.md`. That narrative file (and the
existing `technical_capstone.pptx` deck) are unchanged by this export.

Historical analysis of 27,348 scoped Ticket records, 2016-01-03 to
2023-03-14 UTC. Evidence boundary: the source ends in March 2023 and does
not validate current operations.

---

## 1. Problem statement

- **Retrospective task:** compare elapsed resolution time of completed
  Ticket records with project-configured priority references.
- **Predictive task:** forecast daily counts of exact `Ticket`-type issues,
  separate 7- and 30-day horizons, using only past counts and calendar
  features.
- **ML framing:** univariate time-series regression via model comparison —
  seasonal-naive baseline, weekly ETS, XGBoost autoregression (28-day
  lags), MI/PCA-selected Ridge. Supervised regression, not classification.
- **Research acceptance criterion:** held-out MAE below the same-weekday
  baseline at both horizons — met (5.05 vs 6.41 at 7 days; 4.67 vs 6.49 at
  30 days).
- **Proposed business criterion (not measured/approved):** ≥10% fewer
  overtime hours per 100 tickets in a human-reviewed pilot, without
  worsening an owner-approved service-attainment rate.

---

## 2. Data provenance, licensing & structure

- Source: Help Desk Tickets (Mendeley Data v2, CC BY 4.0), mirrored on
  Kaggle under a conflicting CC0 label — original attribution retained,
  mirror-label conflict flagged for resolution.
- `data/raw/issues.csv`: 66,691 rows × 58 fields. Local file bytes verified
  against both the Kaggle mirror and the official Mendeley digest.
- Documented scope is 2016–March 2023, but `issue_created` contains dates
  back to 2007-04-15; pre-2016 rows are profiled and excluded from the
  primary analysis as a disclosed provenance caveat.
- Primary cohort: exact `issue_type == Ticket`, 2016+ → **27,348 rows**;
  344 of 2,628 calendar days (13.1%) have no Ticket rows, filled as zero
  under an unverified completeness assumption.
- Two structures are kept separate: the wide 58-column raw table (identity,
  actors, classification, timestamps, workflow-state fields) versus a
  narrow, purpose-built one-row-per-day forecasting table (target, past-only
  lags, rolling means, cyclical calendar encodings) — this separation is
  what prevents same-day information leakage into the forecast features.

---

## 3. Pipeline workflow

- Eight notebooks run in sequence, each writing to one canonical
  `output/issue_helpdesk/` folder: data understanding → EDA/feature
  engineering → model training/comparison → explainability (SHAP/PDP/ICE/
  LIME) → bias/fairness audit → GenAI-scope dashboard contract → MLflow
  deployment/monitoring → business ROI template.
- Downstream deliverables (decks, final report, FastAPI demo) all read from
  that same canonical output folder — one reproducible source of truth.
- Dashboard proof of work: a Power BI-ready data contract and folder
  (fact tables, model-comparison table, sample report screenshots) built
  without Power BI Desktop installed locally, substituted with rendered
  dashboard-styled proof images.
- Full function/file-level mapping: `docs/process_flow.md`.

---

## 4. Retrospective priority-SLA reference results

| Mapped priority | Reference | Assessed | Met | Met rate |
|---|---:|---:|---:|---:|
| Critical | 4h | 2,124 | 524 | 24.7% |
| High | 8h | 3,146 | 576 | 18.3% |
| Medium | 24h | 11,161 | 2,161 | 19.4% |
| Low | 24h | 304 | 72 | 23.7% |

- 16,735 eligible completed Tickets overall; **3,333 (19.9%) met reference**
  — consistent at roughly 1-in-5 across every priority, including Critical.
- Duration is elapsed UTC wall-clock time, not confirmed business-hours or
  pause-adjusted SLA compliance — a service owner must validate the clock
  and mapping before any compliance claim.

---

## 5. From issue events to daily volume

1. Parse `issue_created` (UTC, mixed fractional-second formats).
2. Restrict to `issue_type == Ticket`, documented 2016+ scope.
3. Aggregate to one count per calendar day.
4. Fill no-row dates as zero, only under the stated completeness assumption.
5. Engineer past-only lag/rolling-mean/cyclical-calendar features.

- Result: **2,628 daily observations**, averaging 10.4 tickets/day, 344
  assumed zero-arrival days, counts ranging 0–88.

---

## 6. Forecast experiment design

- 7- and 30-day horizons, weekly-spaced **expanding-window** origins.
- 365-day validation period for model selection; separate 365-day final
  test period never used for selection.
- 7-day: 52 validation origins, 51 test origins. 30-day: 48 origins each
  split (each forecast must fit inside its evaluation window).
- Four candidates per horizon: seasonal naive, additive weekly ETS,
  XGBoost autoregression (28-day lags), MI/PCA-selected Ridge — selected
  independently per horizon by validation MAE only.

---

## 7. Validation selection & final test

- **7-day:** XGBoost wins validation (MAE 4.181) over ETS, Ridge, naive.
  Test MAE **5.05** vs seasonal baseline 6.41.
- **30-day:** weekly-additive ETS wins validation (MAE 4.394). Test MAE
  **4.67**, RMSE 7.20, WAPE 27.5%, sMAPE 41.3%, 90% interval coverage 87.7%,
  vs seasonal baseline test MAE 6.49.
- Selection is **never** changed on test rank — ETS's lower 7-day test MAE
  does not override the 7-day XGBoost validation selection.
- Bounded hyperparameter sensitivity/tuning search re-validates any
  candidate on the honest rolling-origin backtest; current baseline
  confirmed as the best searched configuration (`keep_baseline`).

---

## 8. Metrics, uncertainty & explainability

- MAE/RMSE describe absolute daily miss (RMSE penalizes larger misses);
  WAPE/sMAPE express relative error. 90% intervals are validation-residual
  quantiles — measured test coverage (86.6%/87.7%) runs below nominal 90%,
  so calibration is imperfect and not guaranteed to persist.
- Challenger-model explanations (pre-validation XGBoost fit only): **SHAP**
  (leading lags 28/21/14), **PDP** (average marginal effect, top 3
  features), **ICE** (per-validation-day curves exposing heterogeneity PDP
  averages mask), **LIME** (one representative local explanation).
- Primary 30-day ETS is interpreted separately through its own level/trend/
  weekly-seasonal components and residual diagnostics — none of the
  XGBoost explanations above explain the ETS artifact.

---

## 9. Ethics & fairness interpretation

- Source has no approved protected demographic attributes (age, gender,
  ethnicity, etc.) — **protected-group fairness cannot be assessed**, not
  because it was skipped, but because the data to run it does not exist.
- Substituted operational-equity proxy: priority/year/masked-project-group
  breakdowns with **Wilson confidence intervals** for small-group rates.
- Standard mitigation toolkit (reweighting, thresholds, augmentation,
  post-processing) is classification/protected-attribute tooling that does
  not transplant onto a no-protected-attribute regression/forecasting task;
  substituted safeguards are the operational audit above, a recommendation
  to collect governed protected-attribute data before any fairness claim,
  and mandatory human review of forecast-driven decisions.

---

## 10. Integration readiness

- **Locally validated:** all eight notebooks executed; 52/52 tests pass
  without skips; FastAPI demo passed a live localhost health/forecast/
  validation check; MLflow run logs parameters and metrics.
- **Serving scope:** the API serves one persisted primary 30-day ETS
  artifact for all accepted horizons — `horizon_days=7` does **not** load
  the validation-selected 7-day XGBoost model.
- **Not ready for operational use:** unresolved date-range provenance
  mismatch; unverified zero-arrival completeness assumption; dataset ends
  March 2023; SLA mapping/clock unapproved; no current feed; no real-world
  staffing outcomes or costs measured.

---

## 11. Conclusion & next steps

- Validation selects XGBoost at 7 days and ETS at 30 days; both beat the
  seasonal-naive baseline on held-out test MAE. Neither this result nor
  lower historical error establishes operational or business impact.
- **Before operational reliance:** resolve the provenance/coverage
  caveats; obtain current representative arrivals beyond March 2023;
  confirm SLA clock/policy with service owners; agree business tolerances
  in advance; test on a genuinely future period; run a human-reviewed ROI
  pilot with approved cost/outcome data.
- **Bottom line:** a corrected, reproducible historical forecasting
  prototype with a disclosed retrospective SLA baseline — not a contractual
  SLA-compliance system or a proven staffing-cost/ROI tool.

---

## Format note

This file (`presentations/technical_slides.md`) renders two ways via
`src/generate_technical_reveal_deck.py`:

- **reveal.js HTML** (`presentations/technical_reveal_deck.html`) — open
  directly in any browser; no install required beyond an internet
  connection to load the reveal.js CDN assets referenced by the page.
- **Beamer LaTeX source** (`presentations/technical_reveal_deck_beamer.tex`)
  — a valid Beamer `.tex` source file. No LaTeX engine is installed in
  this environment, so it is provided as compileable source, not a
  pre-rendered PDF; compile it with `pdflatex`/`xelatex` where available.

`technical_capstone.pptx` and `technical_presentation.pptx` in this folder
are unchanged by this export.
