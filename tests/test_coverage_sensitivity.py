import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import generate_coverage_sensitivity as sensitivity


class CoverageSensitivityTests(unittest.TestCase):
    def test_imputation_is_causal_and_leaves_observations_unchanged(self):
        series = pd.Series(np.arange(1, 71, dtype=float), index=pd.date_range("2020-01-01", periods=70, tz="UTC"))
        series.iloc[[2, 28, 35]] = 0
        original = series.copy()
        result = sensitivity.causal_gap_scenario(series)
        self.assertEqual(result.iloc[2], 1.5)
        self.assertEqual(result.iloc[28], np.median([1, 8, 15, 22]))
        self.assertEqual(result.iloc[35], np.median([1, 8, 15, 22]))
        changed = series.copy()
        changed.iloc[36:] = 900
        pd.testing.assert_series_equal(result.iloc[:36], sensitivity.causal_gap_scenario(changed).iloc[:36])
        pd.testing.assert_series_equal(series, original)
        pd.testing.assert_series_equal(result[series > 0], series[series > 0])

    def test_invalid_counts_calendar_and_unfillable_gap_raise(self):
        for values, dates in [
            ([0, 2], ["2020-01-01", "2020-01-02"]),
            ([1, np.nan], ["2020-01-01", "2020-01-02"]),
            ([1, -1], ["2020-01-01", "2020-01-02"]),
            ([1, 2], ["2020-01-01", "2020-01-03"]),
        ]:
            with self.subTest(values=values, dates=dates), self.assertRaises(ValueError):
                sensitivity.causal_gap_scenario(pd.Series(values, index=pd.to_datetime(dates, utc=True)))

    def test_scenario_refit_uses_origin_history_and_retains_real_targets(self):
        history = pd.Series(np.arange(1, 15, dtype=float), index=pd.date_range("2020-01-01", periods=14, tz="UTC"))
        origin = history.index[9]
        baseline = pd.DataFrame({
            "origin_date": [origin] * 2, "horizon_days": [1, 2], "model": [sensitivity.MODELS[1]] * 2,
            "actual_tickets": [11.0, 12.0], "forecast_tickets": [1.0, 1.0],
        })
        seen = []

        def fit(training, model, seed):
            seen.append(training.copy())
            return SimpleNamespace(predict=lambda h, n: np.array([-1.0, 8.0]))

        with patch.object(sensitivity, "_fit_final_model", side_effect=fit):
            result = sensitivity.scenario_predictions(baseline, history)
        pd.testing.assert_series_equal(seen[0], history.iloc[:10])
        self.assertEqual(result["forecast_tickets"].tolist(), [0.0, 8.0])
        self.assertEqual(result["actual_tickets"].tolist(), [11.0, 12.0])
        self.assertEqual(baseline["forecast_tickets"].tolist(), [1.0, 1.0])

    def test_metrics_use_common_observed_targets_never_imputed_labels(self):
        frame = pd.DataFrame({
            "scenario": ["canonical_zero_fill"] * 2 + ["causal_missing_record_stress"] * 2,
            "split": ["validation"] * 4, "forecast_window_days": [7] * 4,
            "model": ["ETS"] * 4, "actual_tickets": [0, 10, 0, 10],
            "forecast_tickets": [4, 8, 9, 9],
            "forecast_date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2, utc=True),
        })
        metrics = sensitivity.summarize_predictions(frame)
        self.assertEqual(len(metrics), 3)
        positive = metrics[metrics["scored_population"].eq("observed_positive_dates_only")]
        self.assertEqual(set(positive["forecast_target_pairs"]), {1})
        self.assertEqual(set(positive["excluded_zero_record_pairs"]), {1})
        self.assertEqual(positive.set_index("scenario").loc["canonical_zero_fill", "MAE"], 2.0)
        self.assertEqual(positive.set_index("scenario").loc["causal_missing_record_stress", "MAE"], 1.0)

    def test_prediction_validation_rejects_calendar_and_target_mismatch(self):
        series = pd.Series([1., 2., 3., 4.], index=pd.date_range("2020-01-01", periods=4, tz="UTC"))
        frame = pd.DataFrame({
            "origin_date": series.index[[0, 2]], "forecast_date": series.index[[1, 3]],
            "horizon_days": [1, 1], "forecast_window_days": [1, 1],
            "split": ["validation", "test"], "model": ["ETS", "ETS"],
            "actual_tickets": [2., 4.], "forecast_tickets": [1., 3.],
        })
        sensitivity.validate_predictions(frame, series)
        frame.loc[0, "actual_tickets"] = 8.
        with self.assertRaisesRegex(ValueError, "targets differ"):
            sensitivity.validate_predictions(frame, series)
        frame.loc[0, "actual_tickets"] = 2.
        frame.loc[0, "horizon_days"] = 2
        with self.assertRaisesRegex(ValueError, "Forecast dates"):
            sensitivity.validate_predictions(frame, series)


if __name__ == "__main__":
    unittest.main()
