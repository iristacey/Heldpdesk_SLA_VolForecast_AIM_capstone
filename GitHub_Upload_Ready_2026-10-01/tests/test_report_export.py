from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import generate_final_report as report


class ReportExportTests(unittest.TestCase):
    def test_missing_pdf_is_an_error_and_preserves_previous_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF-old")
            html = root / "report.html"
            html.write_text("temporary HTML", encoding="utf-8")
            with patch.multiple(report, REPORT_PDF=pdf, REPORT_HTML=html), patch.object(
                report.subprocess, "run",
            ):
                with self.assertRaisesRegex(RuntimeError, "did not produce"):
                    report.build_pdf("pandoc", "browser")
            self.assertEqual(pdf.read_bytes(), b"%PDF-old")
            self.assertFalse(html.exists())

    def test_browser_uses_isolated_profile_and_publishes_created_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            html = root / "report.html"
            calls = []

            def convert(command, **kwargs):
                calls.append(command)
                if command[0] == "pandoc":
                    html.write_text("report", encoding="utf-8")
                else:
                    destination = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--print-to-pdf="))
                    Path(destination).write_bytes(b"%PDF-new")

            with patch.multiple(report, REPORT_PDF=pdf, REPORT_HTML=html), patch.object(
                report.subprocess, "run", side_effect=convert,
            ):
                report.build_pdf("pandoc", "browser")
            self.assertEqual(pdf.read_bytes(), b"%PDF-new")
            self.assertIn("--metadata=pagetitle:Final Capstone Report", calls[0])
            self.assertNotIn("--metadata=title:Final Capstone Report", calls[0])
            self.assertTrue(any(arg.startswith("--user-data-dir=") for arg in calls[1]))
            self.assertFalse(html.exists())

    def test_default_export_does_not_silently_skip_missing_browser(self):
        with patch.object(report, "REPORT_MD", Path(__file__)), patch(
            "sys.argv", ["generate_final_report.py"],
        ), patch.object(
            report, "_require_pandoc", return_value="pandoc",
        ), patch.object(report, "build_docx"), patch.object(
            report, "_find_browser", side_effect=SystemExit("Browser required"),
        ):
            with self.assertRaisesRegex(SystemExit, "Browser required"):
                report.main()


if __name__ == "__main__":
    unittest.main()
