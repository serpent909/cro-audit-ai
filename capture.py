# capture.py
import sys
import json
import base64
import io
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


def main():
    # args: url, max_images
    url = sys.argv[1]
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    base_url = url.rstrip("/")
    paths = ["", "/pricing", "/plans", "/compare", "/pricing/"]
    pages = []
    for p in paths:
        pages.append(base_url + p)

    # de-dupe
    seen = set()
    pages_unique = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            pages_unique.append(p)

    image_data_urls = []
    pages_captured = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
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
            except Exception:
                continue

        context.close()
        browser.close()

    print(json.dumps({
        "pages": pages_captured,
        "images": image_data_urls
    }))


if __name__ == "__main__":
    main()