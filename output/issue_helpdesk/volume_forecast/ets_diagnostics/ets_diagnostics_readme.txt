PRIMARY 30-DAY ETS: COMPONENT AND RESIDUAL EVIDENCE
Reproduce: .\.venv\Scripts\python.exe -m src.generate_ets_diagnostics

No training is run. The saved final ETS and existing rolling predictions are read only.
Level is the evolving baseline count; trend is the daily change in that baseline; the seven-day seasonal state adds a repeating weekday offset.
One-step fitted value at t = level[t-1] + trend[t-1] + seasonal[t-7]. Post-update states at t already incorporate observation t, so summing those is NOT its fitted value.
Future raw value at lead h = terminal level + h * terminal trend + repeated effective seven-day seasonal cycle used by the saved library forecast. Canonical evaluated forecasts are clipped below at zero. The effective cycle is not assumed equal to the final updated states.
Terminal level: 20.096528; daily trend: 0.000713; seasonal range: -14.923 to 8.379.
Smoothing weights: alpha=0.049226, beta=0.000000, gamma=0.043729. A zero beta means the estimated initial trend is carried forward without trend updates.
Saved final optimizer: {'success': True, 'status': 0, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH', 'nit': 58, 'nfev': 884}.

IN-SAMPLE: n=2628, mean actual-minus-fitted=0.0001, MAE=3.6749, RMSE=5.5517.
Positive signed error means underprediction; negative means overprediction.
Lag 1: ACF=0.13463; unadjusted Ljung-Box cumulative p=4.99916e-12.
Lag 7: ACF=-0.00135; unadjusted Ljung-Box cumulative p=1.55294e-12.
Lag 14: ACF=0.02320; unadjusted Ljung-Box cumulative p=1.57249e-12.
Lag 28: ACF=-0.02071; unadjusted Ljung-Box cumulative p=1.63988e-10.

HELD-OUT: 1440 origin/lead pairs, 48 origins, 359 unique target dates; 2022-03-21 to 2023-03-14.
Test MAE=4.6723, RMSE=7.1970, WAPE=27.51%, sMAPE=41.33%, bias(actual-forecast)=0.7087.

LIMITATIONS
- The persisted model was refit on ALL history, including dates in the historical test. Its fitted residuals and states are in-sample explanation, not held-out performance.
- Held-out evidence comes ONLY from saved expanding-window test forecasts made with history available at each origin. The 30-day model was selected by validation MAE, not test MAE.
- Test statistics weight every origin/lead pair, matching canonical metrics. Overlapping 30-day windows at seven-day strides reuse dates; errors are not independent observations.
- ACF bands +/-1.96/sqrt(n) and Ljung-Box p-values (model_df=0, no fitted-parameter or multiple-testing adjustment) are descriptive screening tools. Neither small mean error nor non-rejection establishes residual whiteness.
- Optimizer success refers only to the saved final fit, not all historical rolling fits; no historical optimizer logs are available here. Boundary parameters can limit adaptation.
- Level and seasonal offsets use the saved raw parameterization; seasonal offsets need not average zero. Their sum is interpretable, but their separate baselines are not unique.
- The final 30-day projection is after the March 2023 data endpoint, not a current forecast or a held-out test. No new uncertainty intervals are inferred from fitted residuals.
- Zero-arrival completeness is assumed, not independently verified. Historical holdout performance is not prospective/external validation and may not transfer under drift.
- These are additive ETS state contributions, not causal feature effects or SHAP explanations.
- The effective forward seasonal cycle is obtained from the saved model's first seven forecasts after subtracting level and trend, then checked against all 30 forecasts. It can differ from the final post-update seasonal states at the library's forecast boundary; the CSV records that difference explicitly. No model behavior is changed.

OUTPUT GUIDE
ets_components.png: recent fitted history, full-history level/trend and terminal weekday cycle.
ets_residual_diagnostics.png: in-sample residuals/ACF versus held-out errors and lead metrics.
ets_fitted_components.csv: full daily states, pre-update prediction contributions and residuals.
ets_forecast_components_30d.csv: exact raw forecast sum and canonical clipped version.
ets_residual_autocorrelation.csv: lags 1-28, ACF, heuristic bands and cumulative Ljung-Box.
ets_test_errors_30d.csv: saved primary test rows plus signed/absolute errors.
ets_test_errors_by_lead.csv: canonical error metrics separately for leads 1-30.
ets_diagnostics_summary.json: parameters, convergence, source hashes, metrics and caveats.
