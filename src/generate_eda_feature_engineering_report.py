"""Export Notebook 02 (EDA, feature engineering, selection, PCA) as a standalone
"EDA + Feature Engineering Report" in .html and .pdf.

Single source of truth: `notebooks/02_eda_feature_engineering.ipynb`, already executed
in place (code, tables and charts). This script only converts the existing executed
notebook to a report format; it does not re-run the pipeline, change any config, or
touch any other generated artifact.

Requires:
1. `nbconvert` (already a project dependency) - renders the executed notebook to a
   self-contained .html file (inlined images, no external assets).
2. A Chromium-family browser (Microsoft Edge by default, override with --browser) -
   run headless to print that HTML to .pdf via --print-to-pdf, matching the same
   approach used by `src/generate_final_report.py`.

Usage:
    python src/generate_eda_feature_engineering_report.py
    python src/generate_eda_feature_engineering_report.py --browser "C:\\path\\to\\chrome.exe"
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_eda_feature_engineering.ipynb"
REPORT_DIR = ROOT / "output" / "issue_helpdesk" / "reports"
REPORT_HTML = REPORT_DIR / "EDA_Feature_Engineering_Report.html"
REPORT_PDF = REPORT_DIR / "EDA_Feature_Engineering_Report.pdf"
REPORT_TITLE = "EDA + Feature Engineering Report"

DEFAULT_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def _find_browser(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"--browser path does not exist: {explicit}")
        return explicit
    for candidate in DEFAULT_EDGE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit(
        "No Chromium-family browser found at the default install locations. "
        "Pass --browser <path-to-msedge-or-chrome-exe>."
    )


def build_html() -> None:
    if not NOTEBOOK.exists():
        raise SystemExit(f"Notebook not found: {NOTEBOOK}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--embed-images",
            str(NOTEBOOK),
            "--output",
            REPORT_HTML.stem,
            "--output-dir",
            str(REPORT_DIR),
        ],
        check=True,
        cwd=ROOT,
    )
    html = REPORT_HTML.read_text(encoding="utf-8")
    html = html.replace("<title>02_eda_feature_engineering</title>", f"<title>{REPORT_TITLE}</title>")
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")


def build_pdf(browser: str) -> None:
    with tempfile.TemporaryDirectory(prefix="eda-fe-report-") as directory:
        temporary = Path(directory)
        pdf = temporary / REPORT_PDF.name
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                f"--user-data-dir={temporary / 'browser-profile'}",
                f"--print-to-pdf={pdf}",
                "--no-pdf-header-footer",
                REPORT_HTML.as_uri(),
            ],
            check=True,
            cwd=ROOT,
        )
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("Browser did not produce a valid EDA + Feature Engineering report PDF.")
        shutil.copyfile(pdf, REPORT_PDF)
    print(f"Wrote {REPORT_PDF}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser",
        default=None,
        help="Path to a Chromium-family browser executable for PDF export "
        "(defaults to auto-detecting Microsoft Edge or Google Chrome).",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Only build the .html (skip PDF export if no browser is available).",
    )
    args = parser.parse_args()

    build_html()
    if args.skip_pdf:
        return
    browser = _find_browser(args.browser)
    build_pdf(browser)


if __name__ == "__main__":
    main()
