import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"),
    "Poster rendering requires the full requirements.txt environment",
)
class PosterTests(unittest.TestCase):
    def setUp(self):
        from src import generate_poster

        self.poster = generate_poster
        self.addCleanup(generate_poster.plt.close, "all")

    def _require_poster_artifacts(self):
        for path in (
            self.poster.VOLUME / "volume_forecast_experiment.json",
            self.poster.VOLUME / "volume_forecast_rolling_metrics.csv",
            self.poster.VOLUME / "xgboost_validation_shap_summary.csv",
            self.poster.REPORTS / "sla_attainment_by_priority.csv",
        ):
            if not path.is_file():
                self.skipTest(f"Run the analysis before rendering the poster: {path.name}")

    def test_canonical_content_fits_without_overlapping_text(self):
        self._require_poster_artifacts()
        numbers = self.poster._load_numbers()
        fig = self.poster._create_poster(numbers)
        self.poster._validate_layout(fig)
        texts = [text.get_text() for text in fig.axes[0].texts]
        for number in range(1, 8):
            self.assertIn(str(number), texts)
        self.assertIn(f"{numbers['winner_test']['MAE']:.2f}", texts)
        self.assertIn(f"{numbers['overall_sla']['reference_met_pct']:.2f}%", texts)
        content = "\n".join(texts)
        self.assertIn("not current-operations evidence", content)
        self.assertIn("demographic fairness cannot be assessed", content)
        self.assertIn("not approved policy", content)
        self.assertEqual(len(fig.axes[0].images), 0)

    def test_poster_marks_pre_correction_comparison(self):
        self._require_poster_artifacts()
        numbers = self.poster._load_numbers()
        numbers["manifest"].pop("forecast_calendar_policy", None)
        fig = self.poster._create_poster(numbers)
        self.poster._validate_layout(fig)
        self.assertTrue(any(
            "RE-RUN REQUIRED" in text.get_text() for text in fig.axes[0].texts
        ))
        numbers["manifest"]["forecast_calendar_policy"] = self.poster.FORECAST_CALENDAR_POLICY
        fig = self.poster._create_poster(numbers)
        self.assertFalse(any(
            "RE-RUN REQUIRED" in text.get_text() for text in fig.axes[0].texts
        ))

    def test_presentation_generators_share_one_destination(self):
        from presentations import generate_presentation
        from src.generate_capstone_decks import OUTPUT

        expected = self.poster.ROOT / "presentations"
        self.assertEqual(self.poster.PRESENTATION, expected)
        self.assertEqual(OUTPUT, expected)
        self.assertEqual(generate_presentation.OUTPUT, expected)

    def test_presentation_packaging_writes_to_shared_destination(self):
        from presentations import generate_presentation

        for name in ("volume_forecast_experiment.json", "daily_ticket_counts.csv"):
            if not (generate_presentation.REPORTS / name).is_file():
                self.skipTest(f"Run the forecast analysis before packaging: {name}")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "technical_capstone.pptx"
            source.write_bytes(b"existing deck")
            with patch.object(generate_presentation, "OUTPUT", output), patch.object(
                generate_presentation, "build_all",
            ) as build:
                generate_presentation.main()
            build.assert_called_once_with()
            self.assertEqual(
                (output / "technical_presentation.pptx").read_bytes(),
                source.read_bytes(),
            )
            for name in (
                "volume_forecast_experiment.json",
                "daily_ticket_counts.csv",
                "daily_ticket_counts.png",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_long_copy_fails_instead_of_shrinking(self):
        fig, ax = self.poster.plt.subplots()
        with self.assertRaisesRegex(ValueError, "exceeds 2 lines"):
            self.poster._paragraph(ax, 0, 0, "Long text " * 100, 100)

    def test_layout_rejects_collisions_and_clipping(self):
        fig, ax = self.poster.plt.subplots()
        ax.text(0.5, 0.5, "First")
        second = ax.text(0.5, 0.5, "Second")
        with self.assertRaisesRegex(ValueError, "Overlapping poster text"):
            self.poster._validate_layout(fig)
        second.set_position((10, 10))
        with self.assertRaisesRegex(ValueError, "Text outside poster"):
            self.poster._validate_layout(fig)

    def test_print_exports_keep_page_size_and_resolution(self):
        from PIL import Image

        self._require_poster_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            png, pdf = root / "poster.png", root / "poster.pdf"
            with patch.multiple(
                self.poster, PRESENTATION=root, POSTER_PNG=png, POSTER_PDF=pdf,
            ):
                self.assertEqual(self.poster.build_poster(), png)
            with Image.open(png) as image:
                self.assertEqual(image.size, (3600, 5400))
                self.assertAlmostEqual(image.info["dpi"][0], 300, places=1)
            pdf_bytes = pdf.read_bytes()
            self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
            self.assertRegex(pdf_bytes, rb"/MediaBox\s*\[\s*0\s+0\s+864\s+1296\s*\]")
            self.assertRegex(pdf_bytes, rb"/Count\s+1\b")


if __name__ == "__main__":
    unittest.main()
