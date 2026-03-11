# preview_pdf.py
# ─────────────────────────────────────────────────────────────────────────────
# Fast PDF formatting preview — no audit needed.
#
# Workflow:
#   1. Run an audit in the app and click "Save audit_fixture.json" (in the
#      "Dev: Save PDF test fixture" expander at the bottom of the results).
#   2. Edit CSS / layout in app.py (the _PDF_CSS constant or build_pdf_html).
#   3. Run:  python preview_pdf.py
#      -> writes preview.html and opens it in your default browser.
#   4. In the browser, press Ctrl+P (Cmd+P on Mac) for print preview — this
#      uses the same Chromium engine as the PDF generator, so page breaks,
#      @page margins, and margin-box footers render identically.
#   5. Repeat from step 2 — each run takes ~1 second.
#
# Optional flags:
#   --pdf      Also generate the actual PDF (preview.pdf) via Playwright.
#              Use this for a final check once you're happy with the HTML.
#   --fixture  Path to a different fixture file (default: audit_fixture.json)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT / "audit_fixture.json"
PREVIEW_HTML    = ROOT / "preview.html"
PREVIEW_PDF     = ROOT / "preview.pdf"


def load_fixture(path: Path) -> dict:
    if not path.exists():
        print(f"[preview_pdf] ERROR: fixture not found at {path}")
        print(
            "  -> Run an audit in the app, then click "
            '"Save audit_fixture.json" in the Dev expander.'
        )
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Preview PDF formatting without re-running audit.")
    parser.add_argument("--pdf",     action="store_true", help="Also generate preview.pdf via Playwright")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Path to fixture JSON")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    print(f"[preview_pdf] Loading fixture: {fixture_path}")
    data = load_fixture(fixture_path)

    # Import build_pdf_html from app.py (in the same directory)
    sys.path.insert(0, str(ROOT))
    from app import build_pdf_html

    print("[preview_pdf] Building HTML...")
    html = build_pdf_html(
        report_md = data["report_md"],
        ps_raw    = data.get("ps_raw", {}),
        url       = data["url"],
        site_type = data.get("site_type", "SaaS"),
    )

    PREVIEW_HTML.write_text(html, encoding="utf-8")
    print(f"[preview_pdf] HTML written -> {PREVIEW_HTML}")

    if args.pdf:
        print("[preview_pdf] Generating PDF via Playwright (this takes a few seconds)...")
        from app import generate_pdf_bytes
        pdf_bytes = generate_pdf_bytes(html)
        PREVIEW_PDF.write_bytes(pdf_bytes)
        print(f"[preview_pdf] PDF written  -> {PREVIEW_PDF}")

    # Open HTML in browser
    url = PREVIEW_HTML.as_uri()
    print(f"[preview_pdf] Opening in browser: {url}")
    print("  -> Press Ctrl+P (Cmd+P) in the browser for print preview.")
    webbrowser.open(url)


if __name__ == "__main__":
    main()
