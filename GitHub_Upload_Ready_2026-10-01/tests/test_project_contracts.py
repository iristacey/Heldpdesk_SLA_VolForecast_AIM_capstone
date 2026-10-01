import json
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.issues_capstone import (
    CONFIG, FORECAST_CALENDAR_POLICY, MODELS_DIR, PROJECT_REPORTS, REPORTS,
    _metrics, lag_feature_row,
)


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_configuration_defines_two_core_workstreams_and_scope(self):
        config = yaml.safe_load(
            (ROOT / "configs" / "project_config.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["dataset"], "data/raw/issues.csv")
        self.assertEqual(
            [item["id"] for item in config["project_workstreams"]],
            ["retrospective_priority_sla_assessment", "daily_ticket_volume_forecasting"],
        )
        self.assertEqual(config["analysis_scope"]["issue_type"], "Ticket")
        self.assertEqual(config["analysis_scope"]["documented_start_date"], "2016-01-01")
        self.assertTrue(config["analysis_scope"]["complete_daily_coverage"])

    def test_active_generated_artifacts_use_canonical_output_root(self):
        output_root = ROOT / "output"
        self.assertTrue(PROJECT_REPORTS.is_relative_to(output_root))
        self.assertTrue(REPORTS.is_relative_to(output_root))

    def test_process_flow_and_container_test_environment_are_documented(self):
        flow = (ROOT / "docs" / "process_flow.md").read_text(encoding="utf-8")
        self.assertIn("```mermaid", flow)
        for notebook in range(1, 9):
            self.assertIn(f"Notebook {notebook:02d}", flow)
        self.assertIn("output/issue_helpdesk/", flow)
        self.assertTrue((ROOT / "Dockerfile").is_file())
        self.assertTrue((ROOT / ".dockerignore").is_file())
        test_requirements = (ROOT / "requirements-test.txt").read_text(encoding="utf-8")
        self.assertIn("numpy==", test_requirements)
        self.assertIn("pandas==", test_requirements)

    def test_sla_reference_mapping_is_explicit_and_unknown_is_not_mapped(self):
        mapping = CONFIG["priority_sla_assessment"]["priority_mapping"]
        self.assertEqual(mapping["Blocker"], "Critical")
        self.assertEqual(mapping["Highest"], "Critical")
        self.assertEqual(mapping["High"], "High")
        self.assertEqual(mapping["Lowest"], "Low")
        self.assertNotIn("unknown", mapping)
        self.assertEqual(
            CONFIG["priority_sla_assessment"]["reference_targets_hours"],
            {"Critical": 4, "High": 8, "Medium": 24, "Low": 24},
        )
        self.assertIn("wall-clock", CONFIG["priority_sla_assessment"]["duration_definition"])

    def test_forecast_contract_compares_week_and_month_horizons_chronologically(self):
        forecast = CONFIG["ticket_volume_forecasting"]
        self.assertEqual(forecast["intended_planning_horizon_days"], 30)
        self.assertEqual(forecast["evaluation_horizons_days"], [7, 30])
        self.assertEqual(forecast["evaluation"]["horizon_days"], 30)
        self.assertEqual(forecast["initial_training_days"], 365)
        self.assertEqual(forecast["validation_days"], 365)
        self.assertEqual(forecast["final_test_days"], 365)
        self.assertIn("final test kept separate", forecast["evaluation"]["model_selection"])

    def test_daily_lag_features_use_only_previous_history(self):
        feature = lag_feature_row(
            np.arange(1, 29, dtype=float), pd.Timestamp("2024-01-01", tz="UTC")
        )
        self.assertEqual(feature["lag_1"], 28)
        self.assertEqual(feature["lag_28"], 1)
        self.assertEqual(feature["mean_7"], 25)
        self.assertEqual(feature["mean_28"], 14.5)

    def test_metrics_are_defined_when_actuals_include_zeros(self):
        result = _metrics([0, 10], [0, 12])
        self.assertEqual(result["MAE"], 1)
        self.assertEqual(result["RMSE"], 2**0.5)
        self.assertEqual(result["WAPE"], 0.2)
        self.assertAlmostEqual(result["sMAPE"], 1 / 11)
        self.assertEqual(result["signed_bias_actual_minus_forecast"], -1)

    def test_all_eight_notebooks_are_current_and_valid_json(self):
        notebooks = sorted((ROOT / "notebooks").glob("0*.ipynb"))
        self.assertEqual(len(notebooks), 8)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(notebook["cells"])
            intro = "".join(notebook["cells"][0]["source"])
            self.assertIn("data/raw/issues.csv", intro)
            self.assertTrue(all(cell.get("id") for cell in notebook["cells"]))
            errors = [
                output
                for cell in notebook["cells"]
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            ]
            self.assertFalse(errors, f"{path.name} contains an execution error")

    def test_generated_source_profile_covers_all_fields(self):
        dictionary_path = PROJECT_REPORTS / "data_dictionary.csv"
        if not dictionary_path.is_file():
            self.skipTest("Run Notebook 01 to generate source profile artifacts.")
        dictionary = pd.read_csv(dictionary_path)
        expected_columns = {
            "column",
            "dtype",
            "rows_processed",
            "present_rows",
            "missing_rows",
            "meaning_or_caveat",
        }
        self.assertEqual(len(dictionary), 58)
        self.assertTrue(expected_columns.issubset(dictionary.columns))
        self.assertEqual(set(dictionary["rows_processed"]), {66_691})
        self.assertTrue(dictionary["meaning_or_caveat"].str.len().gt(0).all())

    def test_forecast_artifacts_have_distinct_full_validation_and_test_periods(self):
        metrics_path = REPORTS / "volume_forecast_rolling_metrics.csv"
        prediction_path = REPORTS / "volume_forecast_rolling_predictions.csv"
        metadata_path = REPORTS / "volume_forecast_experiment.json"
        for path in [metrics_path, prediction_path, metadata_path]:
            if not path.is_file():
                self.skipTest("Run Notebook 03 to generate forecast artifacts.")
        metrics = pd.read_csv(metrics_path)
        predictions = pd.read_csv(prediction_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(set(metrics["split"]), {"validation", "test"})
        self.assertEqual(set(predictions["split"]), {"validation", "test"})
        self.assertEqual(set(predictions["forecast_window_days"]), {7, 30})
        self.assertEqual(
            set(predictions.loc[predictions["forecast_window_days"].eq(7), "horizon_days"]),
            set(range(1, 8)),
        )
        self.assertEqual(
            set(predictions.loc[predictions["forecast_window_days"].eq(30), "horizon_days"]),
            set(range(1, 31)),
        )
        self.assertEqual(metadata["forecast_horizon_days"], 30)
        self.assertEqual(metadata["evaluation_horizons_days"], [7, 30])
        self.assertEqual(metadata["validation_forecast_origins_by_horizon"]["7"], 52)
        self.assertEqual(metadata["test_forecast_origins_by_horizon"]["7"], 51)
        for horizon in [7, 30]:
            validation = metrics.loc[
                metrics["split"].eq("validation")
                & metrics["forecast_window_days"].eq(horizon)
            ]
            test = metrics.loc[
                metrics["split"].eq("test")
                & metrics["forecast_window_days"].eq(horizon)
            ]
            selected = validation.loc[validation["MAE"].idxmin(), "model"]
            self.assertEqual(
                selected,
                metadata["selected_models_by_horizon"][str(horizon)]["selected_model"],
            )
            selected_test = test.loc[test["model"].eq(selected)].iloc[0]
            self.assertTrue(0 <= selected_test["coverage_90_interval"] <= 1)
        self.assertFalse(metadata["complete_daily_coverage_independently_verified"])

    def test_final_model_artifact_is_persisted_and_reproduces_selection(self):
        model_path = MODELS_DIR / "final_volume_forecast_model.joblib"
        metadata_path = MODELS_DIR / "final_volume_forecast_model_metadata.json"
        selection_path = REPORTS / "volume_forecast_model_selection.json"
        for path in [model_path, metadata_path, selection_path]:
            if not path.is_file():
                self.skipTest("Run Notebook 03 / run_volume_forecast to persist the final model.")
        try:
            import joblib
        except ImportError:
            self.skipTest("joblib not installed in this environment (lightweight test image).")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_name"], selection["selected_model"])
        self.assertEqual(metadata["primary_horizon_days"], selection["primary_horizon_days"])
        self.assertGreater(metadata["trained_on_rows"], 0)
        self.assertTrue(model_path.is_relative_to(ROOT))
        artifact = joblib.load(model_path)
        self.assertEqual(artifact.model_name, metadata["model_name"])
        daily = pd.read_csv(REPORTS / "daily_ticket_counts.csv", parse_dates=["date_created"])
        history = pd.Series(
            daily["tickets"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(daily["date_created"], utc=True)),
        )
        forecast = artifact.predict(history, metadata["primary_horizon_days"])
        self.assertEqual(len(forecast), metadata["primary_horizon_days"])
        self.assertTrue(np.all(np.isfinite(forecast)))

    def test_forecast_artifacts_record_corrected_calendar_alignment(self):
        def reject_nonfinite(value):
            raise ValueError(f"Non-standard JSON constant: {value}")

        paths = [
            REPORTS / "volume_forecast_experiment.json",
            REPORTS / "volume_forecast_model_selection.json",
            MODELS_DIR / "final_volume_forecast_model_metadata.json",
            PROJECT_REPORTS / "dashboard_data_contract.json",
            PROJECT_REPORTS / "powerbi" / "dashboard_data_contract.json",
            ROOT / "presentations" / "volume_forecast_experiment.json",
        ]
        for path in paths:
            if not path.is_file():
                self.skipTest("Run Notebook 03 to produce forecast artifacts.")
            metadata = json.loads(
                path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
            )
            if path.name != "dashboard_data_contract.json":
                self.assertEqual(
                    metadata.get("forecast_calendar_policy"),
                    FORECAST_CALENDAR_POLICY,
                    f"{path.name} predates the forecast-calendar fix; re-run Notebook 03.",
                )
            selection = metadata.get("daily_volume_forecast", metadata)
            for model in selection.get("selected_models_by_horizon", {}).values():
                for metric in model["validation_metrics"]:
                    self.assertEqual(metric["split"], "validation")
                    self.assertIsNone(metric["coverage_90_interval"])

    def test_mlflow_run_summary_matches_persisted_model(self):
        summary_path = PROJECT_REPORTS / "mlflow_run_summary.json"
        metadata_path = MODELS_DIR / "final_volume_forecast_model_metadata.json"
        if not summary_path.is_file() or not metadata_path.is_file():
            self.skipTest("Run Notebook 07 / log_experiment_to_mlflow to record an MLflow run.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["experiment_name"], "helpdesk_volume_forecast")
        self.assertTrue(summary["run_id"])
        self.assertEqual(summary["logged_params"]["model_name"], metadata["model_name"])
        self.assertAlmostEqual(
            summary["logged_metrics"]["validation_mae"], metadata["validation_mae"]
        )
        self.assertTrue(summary["tracking_uri"].startswith("sqlite:///"))
        self.assertEqual(
            summary["logged_params"]["primary_horizon_days"],
            metadata["primary_horizon_days"],
        )

    def test_reporting_contract_keeps_sla_and_volume_targets_separate(self):
        contract_path = PROJECT_REPORTS / "dashboard_data_contract.json"
        if not contract_path.is_file():
            self.skipTest("Run Notebook 06 to generate the reporting contract.")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(contract["daily_volume_forecast"]["horizon_days"], 30)
        self.assertEqual(contract["daily_volume_forecast"]["evaluated_horizons_days"], [7, 30])
        self.assertFalse(contract["daily_volume_forecast"]["production_ready"])
        self.assertIn(
            "not policy-validated",
            contract["sla_reference"]["policy_status"].lower(),
        )

    def test_deployment_app_serves_persisted_model_forecast(self):
        model_path = MODELS_DIR / "final_volume_forecast_model.joblib"
        metadata_path = MODELS_DIR / "final_volume_forecast_model_metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            self.skipTest("Run Notebook 03 / run_volume_forecast to persist the final model.")
        try:
            from fastapi.testclient import TestClient

            from src.app import app
        except ImportError:
            self.skipTest("fastapi/joblib not installed in this environment (lightweight test image).")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        client = TestClient(app)

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["model_name"], metadata["model_name"])

        metadata_response = client.get("/model/metadata")
        self.assertEqual(metadata_response.status_code, 200)
        self.assertEqual(metadata_response.json(), metadata)
        default = client.get("/forecast")
        self.assertEqual(default.status_code, 200)
        self.assertEqual(default.json()["horizon_days"], 30)

        for horizon in (1, 7, 30, 90):
            with self.subTest(horizon=horizon):
                forecast = client.get("/forecast", params={"horizon_days": horizon})
                self.assertEqual(forecast.status_code, 200)
                body = forecast.json()
                self.assertEqual(body["model_name"], metadata["model_name"])
                self.assertEqual(body["horizon_days"], horizon)
                self.assertEqual(len(body["forecast"]), horizon)
                self.assertEqual(body["history_ends"], metadata["trained_on_date_end"])
                self.assertEqual(body["trained_on_date_end"], metadata["trained_on_date_end"])
                start = date.fromisoformat(body["history_ends"]) + timedelta(days=1)
                self.assertEqual(
                    [row["date"] for row in body["forecast"]],
                    [(start + timedelta(days=lead)).isoformat() for lead in range(horizon)],
                )
                for row in body["forecast"]:
                    self.assertEqual(set(row), {"date", "forecast_tickets"})
                    self.assertTrue(np.isfinite(row["forecast_tickets"]))
                    self.assertGreaterEqual(row["forecast_tickets"], 0)
                self.assertEqual(body["limitations"], metadata["limitations"])
                self.assertIn("historical", body["caveat"].lower())

        for horizon in (0, -1, 91, "abc", "7.5"):
            with self.subTest(invalid_horizon=horizon):
                response = client.get("/forecast", params={"horizon_days": horizon})
                self.assertEqual(response.status_code, 422)

    def test_rebuilt_decks_are_within_rubric_slide_range(self):
        from pptx import Presentation

        deck_dir = ROOT / "presentations"
        for name in ["technical_capstone.pptx", "business_capstone.pptx"]:
            path = deck_dir / name
            if not path.is_file():
                self.skipTest("Run presentation generator to produce decks.")
            self.assertTrue(8 <= len(Presentation(path).slides) <= 12)

    def test_final_report_covers_all_rubric_sections(self):
        report_path = PROJECT_REPORTS / "reports" / "final_capstone_report.md"
        if not report_path.is_file():
            self.skipTest("Run src/generate_final_report.py to author the report.")
        report = report_path.read_text(encoding="utf-8")
        for heading in [
            "## 1. Problem understanding and framing",
            "## 2. Data collection and understanding",
            "## 3. Preprocessing, EDA and feature engineering",
            "## 4. Model implementation and comparison",
            "## 5. Critical thinking, ethical AI, and bias auditing",
            "## 6. Deployment readiness, monitoring, and experiment tracking",
            "## 7. Reproducibility guide",
            "## 8. Final presentation and communication",
            "## 9. GitHub profile and repository readiness",
            "## 10. Limitations and responsible use",
            "## 11. Conclusion and next steps",
            "## 12. Rubric evidence and completeness self-review",
        ]:
            self.assertIn(heading, report)
        self.assertIn("uvicorn src.app:app", report)
        self.assertIn("helpdesk-project-venv", report)
        self.assertIn("19.92%", report)
        self.assertIn("4.67", report)


if __name__ == "__main__":
    unittest.main()
