"""Regenerate the final capstone report as .docx and .pdf from one Markdown source.

Single source of truth: `output/issue_helpdesk/reports/final_capstone_report.md`
(authored narrative, hand-maintained like `presentations/technical_presentation.md`
and `README.md`, not auto-populated from live artifacts). This script only
handles format conversion, so the three formats (.md/.docx/.pdf) never drift
out of sync with each other.

Requires two system tools (see `requirements-report.txt` for details, since
neither is a pip package):

1. Pandoc - converts Markdown to .docx directly, and to a self-contained
   .html file used for the PDF step.
2. A Chromium-family browser (Microsoft Edge by default, override with
   --browser) - run headless to print that HTML to .pdf via
   `--print-to-pdf`, avoiding a LaTeX/wkhtmltopdf/LibreOffice dependency.

Usage:
    python src/generate_final_report.py
    python src/generate_final_report.py --browser "C:\\path\\to\\chrome.exe"
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "output" / "issue_helpdesk" / "reports"
REPORT_MD = REPORT_DIR / "final_capstone_report.md"
REPORT_DOCX = REPORT_DIR / "final_capstone_report.docx"
REPORT_HTML = REPORT_DIR / "final_capstone_report.html"
REPORT_PDF = REPORT_DIR / "final_capstone_report.pdf"

DEFAULT_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def _require_pandoc() -> str:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise SystemExit(
            "pandoc was not found on PATH. Install it "
            "(https://pandoc.org/installing.html) and re-run this script. "
            "See requirements-report.txt for details."
        )
    return pandoc


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


def build_docx(pandoc: str) -> None:
    subprocess.run(
        [pandoc, str(REPORT_MD), "-o", str(REPORT_DOCX), "--from=gfm", "--standalone"],
        check=True,
        cwd=ROOT,
    )
    print(f"Wrote {REPORT_DOCX}")


def build_pdf(pandoc: str, browser: str) -> None:
    subprocess.run(
        [
            pandoc,
            str(REPORT_MD),
            "-o",
            str(REPORT_HTML),
            "--from=gfm",
            "--standalone",
            "--metadata=title:Final Capstone Report",
            "--css=",
        ],
        check=True,
        cwd=ROOT,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="capstone-report-") as directory:
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
                raise RuntimeError("Browser did not produce a valid report PDF.")
            shutil.copyfile(pdf, REPORT_PDF)
    finally:
        REPORT_HTML.unlink(missing_ok=True)
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
        help="Only build the .docx (skip PDF export if no browser is available).",
    )
    args = parser.parse_args()

    if not REPORT_MD.exists():
        raise SystemExit(f"Report source not found: {REPORT_MD}")

    pandoc = _require_pandoc()
    build_docx(pandoc)

    if args.skip_pdf:
        return
    browser = _find_browser(args.browser)
    build_pdf(pandoc, browser)


if __name__ == "__main__":
    main()
