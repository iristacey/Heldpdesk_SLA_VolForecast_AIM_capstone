"""Recheck local source identity without publishing raw data or local hashes."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def verify(source: Path, evidence: dict) -> dict:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    size = source.stat().st_size
    reference = evidence["mendeley"]["file"]
    if digest != reference["published_sha256"] or size != reference["bytes"]:
        raise ValueError("Local CSV does not match the retained Mendeley SHA-256 and byte count.")
    return {
        "status": "matches_mendeley_v2_published_sha256_and_size",
        "local_sha256": digest, "local_size_bytes": size,
        "publisher_metadata_url": evidence["mendeley"]["files_api_url"],
        "publisher_metadata_retrieved_on": evidence["verification_date"],
        "note": "Offline comparison with retained official metadata, not a fresh Mendeley download.",
    }


def verify_kaggle(source: Path, evidence: dict) -> dict:
    result = verify(source, evidence)
    url = evidence["kaggle"]["download_url"]
    if url != (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "janebitor/help-desk-tickets-mendeley-data?datasetVersionNumber=1"
    ):
        raise ValueError("Only the documented public version-1 Kaggle archive is supported.")
    with urlopen(url, timeout=90) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as downloaded:
        member = downloaded.read("issues.csv")
    if member != source.read_bytes():
        raise ValueError("Downloaded Kaggle issues.csv is not byte-for-byte equal to the local CSV.")
    result["kaggle"] = {
        "status": "exact_byte_match", "download_url": url,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "issues_csv_sha256": hashlib.sha256(member).hexdigest(),
        "issues_csv_bytes": len(member), "normalization_applied": False,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-kaggle", action="store_true",
                        help="Also fetch the public pinned archive and compare bytes in memory.")
    args = parser.parse_args()
    evidence = json.loads((ROOT / "data" / "source_verification.json").read_text(encoding="utf-8"))
    source = ROOT / "data" / "raw" / "issues.csv"
    check = verify_kaggle if args.download_kaggle else verify
    print(json.dumps(check(source, evidence), indent=2))
