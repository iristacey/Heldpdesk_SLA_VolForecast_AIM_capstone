import importlib.util
import json
import unittest

import numpy as np
import pandas as pd

from src import generate_ets_diagnostics as diagnostics
from src import issues_capstone as core


@unittest.skipUnless(importlib.util.find_spec("statsmodels"), "statsmodels is needed for ETS fixtures")
class ETSDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = pd.Series(
            20 + np.arange(84) * 0.03 + np.resize([0, 2, 3, 1, 0, -4, -2], 84)
            + np.sin(np.arange(84)) * 0.5,
            index=pd.date_range("2023-01-01", periods=84, tz="UTC"),
        )
        cls.fitted = core._fit_final_model(cls.history, diagnostics.MODEL_NAME, 42).fitted["ets"]

    def test_state_reconstruction_and_future_calendar(self):
        components, future = diagnostics.component_evidence(self.history, self.fitted)
        self.assertEqual(components.shape, (84, 10))
        self.assertEqual(future.shape, (30, 9))
        np.testing.assert_allclose(
            components["prediction_previous_level"] + components["prediction_previous_trend"]
            + components["prediction_seasonal_lag7"], self.fitted.fittedvalues,
        )
        np.testing.assert_allclose(
            components["residual_actual_minus_fitted"],
            self.history.to_numpy() - self.fitted.fittedvalues.to_numpy(),
        )
        np.testing.assert_allclose(future["forecast_tickets_raw"], self.fitted.forecast(30))
        np.testing.assert_allclose(
            future["last_updated_seasonal_state"], np.resize(self.fitted.season.iloc[-7:], 30),
        )
        np.testing.assert_allclose(
            future["seasonal_contribution"].iloc[:7],
            future["seasonal_contribution"].iloc[7:14],
        )
        self.assertEqual(future["date"].iloc[0], self.history.index[-1] + pd.Timedelta(days=1))
        self.assertEqual(future["date"].iloc[-1], self.history.index[-1] + pd.Timedelta(days=30))

    def test_history_and_horizon_validation(self):
        for history in (
            self.history.iloc[:10], self.history.iloc[::-1], self.history.drop(self.history.index[20]),
            self.history.reset_index(drop=True), self.history.copy().mask(self.history.index.day == 3),
            self.history * -1, self.history.set_axis([self.history.index[0]] * len(self.history)),
        ):
            with self.subTest(history=type(history)), self.assertRaises(ValueError):
                diagnostics.validate_history(history)
        for horizon in (0, -1, 1.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                diagnostics.component_evidence(self.history, self.fitted, horizon)
        with self.assertRaisesRegex(ValueError, "observations do not match"):
            diagnostics.component_evidence(self.history + 1, self.fitted)
        with self.assertRaisesRegex(ValueError, "dates do not match"):
            diagnostics.component_evidence(
                self.history.set_axis(self.history.index + pd.Timedelta(days=1)), self.fitted,
            )

    def test_residual_metrics_acf_and_degenerate_series(self):
        values = np.array([1.0, -2, 3, -4, 5])
        summary, table = diagnostics.residual_evidence(values, 2)
        self.assertAlmostEqual(summary["mean_actual_minus_fitted"], 0.6)
        self.assertAlmostEqual(summary["MAE"], 3)
        self.assertAlmostEqual(summary["RMSE"], np.sqrt(11))
        centered = values - values.mean()
        expected_acf1 = np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered)
        self.assertAlmostEqual(table["acf"].iloc[0], expected_acf1)
        expected_q = len(values) * (len(values) + 2) * expected_acf1 ** 2 / (len(values) - 1)
        self.assertAlmostEqual(table["ljung_box_statistic"].iloc[0], expected_q)
        self.assertEqual(table.shape, (2, 5))
        with self.assertRaisesRegex(ValueError, "constant residuals"):
            diagnostics.residual_evidence(np.ones(10), 3)
        for values, lag in (([], 1), ([1, np.nan, 2], 1), ([[1, 2, 3]], 1),
                            ([1, 2, 3], 3), ([1, 2, 3], 0), ([1, 2, 3], True)):
            with self.subTest(values=values, lag=lag), self.assertRaises(ValueError):
                diagnostics.residual_evidence(values, lag)

    @staticmethod
    def predictions():
        rows = []
        for origin in pd.to_datetime(["2023-01-01", "2023-01-08"], utc=True):
            for lead in range(1, 31):
                rows.append({
                    "model": diagnostics.MODEL_NAME, "split": "test", "forecast_window_days": 30,
                    "origin_date": origin, "forecast_date": origin + pd.Timedelta(days=lead),
                    "horizon_days": lead, "actual_tickets": 5, "forecast_tickets_raw": 3,
                    "forecast_tickets": 3,
                })
        return pd.DataFrame(rows)

    def test_test_errors_filtering_weighting_and_clipping(self):
        primary = self.predictions()
        unrelated = primary.assign(split="validation", actual_tickets=999)
        errors, summary, by_lead = diagnostics.test_error_evidence(pd.concat([primary, unrelated]))
        self.assertEqual(len(errors), 60)
        self.assertEqual(summary["forecast_origins"], 2)
        self.assertEqual(summary["unique_target_dates"], 37)
        self.assertEqual(summary["MAE"], 2)
        self.assertEqual(summary["signed_bias_actual_minus_forecast"], 2)
        self.assertAlmostEqual(summary["WAPE"], 0.4)
        self.assertEqual(by_lead.shape, (30, 7))
        self.assertTrue(by_lead["origin_lead_pairs"].eq(2).all())
        primary.loc[0, ["forecast_tickets_raw", "forecast_tickets"]] = [-2, 0]
        errors, summary, _ = diagnostics.test_error_evidence(primary)
        self.assertEqual(summary["negative_raw_forecasts_clipped"], 1)
        self.assertEqual(errors["error_actual_minus_forecast"].iloc[0], 5)

    def test_invalid_test_predictions(self):
        frame = self.predictions()
        wrong_date = frame.copy()
        wrong_date.loc[0, "forecast_date"] += pd.Timedelta(days=1)
        wrong_clip = frame.copy()
        wrong_clip.loc[0, "forecast_tickets"] = -1
        for invalid in (
            frame.drop(columns="model"), frame.assign(model="Other"), frame.iloc[1:],
            pd.concat([frame, frame.iloc[:1]]), frame.assign(actual_tickets=np.nan),
            wrong_date, wrong_clip,
        ):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                diagnostics.test_error_evidence(invalid)

    def test_json_uses_null_for_nonfinite_parameters(self):
        result = diagnostics._json_safe({"nan": np.float64(np.nan), "array": np.array([1, np.inf])})
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), {"nan": None, "array": [1, None]})


if __name__ == "__main__":
    unittest.main()
