import hashlib
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from src.verify_dataset_source import verify, verify_kaggle


class SourceVerificationTests(unittest.TestCase):
    def test_retained_identity_matches_canonical_experiment(self):
        root = Path(__file__).resolve().parents[1]
        evidence = json.loads((root / "data" / "source_verification.json").read_text(encoding="utf-8"))
        local = evidence["local_file"]
        public = evidence["mendeley"]["file"]
        self.assertEqual(local["sha256"], public["published_sha256"])
        self.assertEqual(local["bytes"], public["bytes"])
        experiment = root / "output" / "issue_helpdesk" / "volume_forecast" / "volume_forecast_experiment.json"
        if experiment.is_file():
            self.assertEqual(
                json.loads(experiment.read_text(encoding="utf-8"))["source_sha256"],
                public["published_sha256"],
                "Changed source requires a new documented identity verification.",
            )

    def test_digest_size_and_downloaded_bytes_must_match(self):
        content = b"a,b\n1,2\n"
        evidence = {
            "mendeley": {"file": {"published_sha256": hashlib.sha256(content).hexdigest(),
                                  "bytes": len(content)}, "files_api_url": "https://example.org/metadata"},
            "verification_date": "2026-09-30",
            "kaggle": {"download_url": "https://www.kaggle.com/api/v1/datasets/download/"
                       "janebitor/help-desk-tickets-mendeley-data?datasetVersionNumber=1"},
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "sample.csv"
            source.write_bytes(content)
            self.assertEqual(verify(source, evidence)["local_size_bytes"], len(content))
            for downloaded, expected_match in [(content, True), (b"a,b\n3,4\n", False)]:
                buffer = BytesIO()
                with ZipFile(buffer, "w") as archive:
                    archive.writestr("issues.csv", downloaded)
                buffer.seek(0)
                with patch("src.verify_dataset_source.urlopen", return_value=buffer):
                    if expected_match:
                        self.assertEqual(verify_kaggle(source, evidence)["kaggle"]["status"], "exact_byte_match")
                    else:
                        with self.assertRaisesRegex(ValueError, "not byte-for-byte"):
                            verify_kaggle(source, evidence)
            evidence["mendeley"]["file"]["bytes"] += 1
            with self.assertRaises(ValueError):
                verify(source, evidence)
            evidence["mendeley"]["file"]["bytes"] -= 1
            source.write_bytes(b"a,b\n3,4\n")
            with self.assertRaises(ValueError):
                verify(source, evidence)


if __name__ == "__main__":
    unittest.main()
