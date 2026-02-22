# capture.py
import sys
import json
import base64
import io
import os
import subprocess
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

MAX_IMAGE_WIDTH = 1400
QUALITY = 85
VIEWPORT = {"width": 1440, "height": 900}
NAV_TIMEOUT_MS = 45000


def png_bytes_to_jpeg_data_url(png_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / float(img.width)
        img = img.resize((MAX_IMAGE_WIDTH, int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _install_chromium() -> None:
    """
    Install Playwright Chromium. Keep stdout clean (JSON only) by sending logs to stderr.
    """
    # Streamlit Cloud friendly cache path
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
    Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]).mkdir(parents=True, exist_ok=True)

    # Send installer output to stderr (avoid polluting stdout)
    # Using run() instead of check_call() gives us better control.
    proc = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"playwright install chromium failed (exit {proc.returncode})")


def _try_launch_browser(pw):
    """
    Attempt to launch Chromium. If missing, install and retry once.
    """
    try:
        return pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
    except Exception as e:
        # Common Streamlit Cloud error: Executable doesn't exist...
        msg = str(e).lower()
        if "executable" in msg and ("doesn't exist" in msg or "not found" in msg):
            print("[capture.py] Chromium missing. Installing...", file=sys.stderr)
            _install_chromium()
            # Retry after install
            return pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
        # Anything else: re-raise
        raise


def main():
    # args: url, max_images
    if len(sys.argv) < 2:
        return {"pages": [], "images": [], "error": "No URL argument provided to capture.py"}

    url = sys.argv[1]
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    base_url = url.rstrip("/")
    paths = ["", "/pricing", "/plans", "/compare", "/pricing/"]
    pages = [base_url + p for p in paths]

    # de-dupe (preserve order)
    seen = set()
    pages_unique = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            pages_unique.append(p)

    image_data_urls = []
    pages_captured = []

    with sync_playwright() as pw:
        browser = _try_launch_browser(pw)

        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
        )
        page = context.new_page()

        for page_url in pages_unique:
            if len(image_data_urls) >= max_images:
                break
            try:
                page.goto(page_url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
                page.wait_for_timeout(1200)
                png_bytes = page.screenshot(full_page=True, type="png")
                image_data_urls.append(png_bytes_to_jpeg_data_url(png_bytes))
                pages_captured.append(page_url)
            except Exception as e:
                # log failures to stderr only (stdout must remain JSON)
                print(f"[capture.py] Failed page: {page_url} :: {e}", file=sys.stderr)
                continue

        context.close()
        browser.close()

    return {"pages": pages_captured, "images": image_data_urls}


if __name__ == "__main__":
    # Always emit valid JSON to stdout, and keep stdout JSON-only.
    try:
        payload = main()
        print(json.dumps(payload))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"pages": [], "images": [], "error": str(e)}))
        print(f"[capture.py] ERROR: {e}", file=sys.stderr)
        # Exit 0 so app.py can parse JSON and show structured error nicely
        sys.exit(0)