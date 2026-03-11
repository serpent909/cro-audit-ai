# pdf_worker.py
# Called as a subprocess by app.py.
# Reads {"html_path": "<path>"} from stdin, renders the HTML file to PDF
# with Playwright, and writes the base64-encoded PDF bytes to stdout.
import sys
import json
import base64

from playwright.sync_api import sync_playwright


def main():
    payload = json.loads(sys.stdin.read())
    html_path = payload["html_path"]

    with open(html_path, "r", encoding="utf-8") as fh:
        html = fh.read()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # set_content renders immediately from string — no network needed
        page.set_content(html, wait_until="domcontentloaded")
        # Brief pause so any CSS calc / font rendering settles
        page.wait_for_timeout(800)
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
        )
        browser.close()

    sys.stdout.write(base64.b64encode(pdf_bytes).decode())
    sys.stdout.flush()


if __name__ == "__main__":
    main()
