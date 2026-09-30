import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import issues_capstone as capstone


class ForecastCalendarTests(unittest.TestCase):
    def test_future_dates_cross_year_and_leap_day_boundaries(self):
        for end in ("2023-12-31", "2024-02-28"):
            with self.subTest(end=end):
                history = pd.Series(
                    np.arange(365, dtype=float),
                    index=pd.date_range(end=end, periods=365, tz="UTC"),
                )
                expected = pd.date_range(
                    pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                    periods=30, tz="UTC",
                )
                pd.testing.assert_index_equal(capstone.forecast_dates(history, 30), expected)
        with self.assertRaisesRegex(ValueError, "at least one observed day"):
            capstone.forecast_dates(pd.Series(dtype=float), 7)

    def test_recursive_challengers_and_artifacts_use_future_calendar(self):
        history = pd.Series(
            np.arange(365, dtype=float),
            index=pd.date_range(end="2024-02-28", periods=365, tz="UTC"),
        )
        expected_dates = pd.date_range("2024-02-29", periods=7, tz="UTC")
        observed = []
        original_feature_row = capstone.lag_feature_row

        def capture(values, date):
            observed.append((date, values[-1]))
            return original_feature_row(values, date)

        class IdentityTransform:
            def __init__(self, **kwargs):
                self.n_components_ = 34
                self.explained_variance_ratio_ = np.ones(34) / 34

            def fit(self, features, target=None):
                return self

            def transform(self, features):
                return np.asarray(features)

            def fit_transform(self, features):
                return self.transform(features)

            def get_support(self):
                return np.ones(34, dtype=bool)

        class ConstantRegressor:
            def __init__(self, **kwargs):
                self.feature_importances_ = np.ones(34) / 34

            def fit(self, *args, **kwargs):
                return self

            def predict(self, features):
                return np.full(len(features), 7.0)

        class ConstantETS:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, **kwargs):
                return self

            def forecast(self, horizon):
                return np.full(horizon, 7.0)

        modules = {
            "xgboost": SimpleNamespace(XGBRegressor=ConstantRegressor),
            "sklearn.decomposition": SimpleNamespace(PCA=IdentityTransform),
            "sklearn.feature_selection": SimpleNamespace(
                SelectKBest=IdentityTransform, mutual_info_regression=lambda *a, **k: None,
            ),
            "sklearn.linear_model": SimpleNamespace(Ridge=ConstantRegressor),
            "sklearn.preprocessing": SimpleNamespace(StandardScaler=IdentityTransform),
            "statsmodels.tsa.holtwinters": SimpleNamespace(ExponentialSmoothing=ConstantETS),
        }
        features, target = capstone.supervised_features(history)
        with patch.dict("sys.modules", modules), patch.object(
            capstone, "supervised_features", return_value=(features, target),
        ), patch.object(capstone, "lag_feature_row", side_effect=capture):
            predictions, _, _, _ = capstone._fit_predict_models(history, 7, 42)
            for offset in (0, 7):
                self.assertEqual([date for date, _ in observed[offset:offset + 7]], list(expected_dates))
                self.assertEqual(observed[offset][1], history.iloc[-1])
                self.assertTrue(all(value == 7.0 for _, value in observed[offset + 1:offset + 7]))
            self.assertEqual(len(observed), 14)
            self.assertEqual(len(predictions), 4)
            for name in (
                "XGBoost autoregression (28-day lags)",
                "Ridge (MI-selected features + PCA)",
            ):
                observed.clear()
                artifact = capstone._fit_final_model(history, name, 42)
                np.testing.assert_array_equal(artifact.predict(history, 7), np.full(7, 7.0))
                self.assertEqual([date for date, _ in observed], list(expected_dates))

    def test_tuning_rejects_pre_correction_baseline(self):
        with TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports / "volume_forecast_experiment.json").write_text(
                json.dumps({"selected_model": "ETS (weekly additive)"}), encoding="utf-8",
            )
            with patch.object(capstone, "REPORTS", reports):
                with self.assertRaisesRegex(ValueError, "Re-run Notebook 03"):
                    capstone.run_hyperparameter_tuning()


if __name__ == "__main__":
    unittest.main()
