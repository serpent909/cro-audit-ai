# capture.py
import sys
import json
import base64
import io
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

# --- Screenshot tuning (Step 3) ---
MAX_IMAGE_WIDTH = 900          # reduce width ~900
QUALITY = 75                   # JPEG quality ~70–80
VIEWPORT = {"width": 1200, "height": 900}

# --- Navigation robustness ---
NAV_TIMEOUT_MS = 15000
POST_GOTO_PAUSE_MS = 500

PRICING_HINTS = [
    "pricing", "plans", "plan", "subscriptions", "subscription", "billing",
    "upgrade", "tiers", "compare", "buy", "checkout", "purchase"
]
DEBOOST_HINTS = ["blog", "docs", "help", "support", "changelog", "status", "careers", "jobs"]


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
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
    Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]).mkdir(parents=True, exist_ok=True)

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
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
        )
    except Exception as e:
        msg = str(e).lower()
        if "executable" in msg and ("doesn't exist" in msg or "not found" in msg):
            print("[capture.py] Chromium missing. Installing...", file=sys.stderr)
            _install_chromium()
            return pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            )
        raise


def _norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    p = urlparse(u)
    return p._replace(fragment="").geturl()


def _same_site(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()
    except Exception:
        return False


def _score_candidate(abs_url: str, text: str) -> int:
    s = (abs_url + " " + (text or "")).lower()
    score = 0
    for kw in PRICING_HINTS:
        if kw in s:
            score += 5
    for bad in DEBOOST_HINTS:
        if bad in s:
            score -= 3

    path = urlparse(abs_url).path.lower().strip("/")
    if path in ("pricing", "plans"):
        score += 10
    if "compare" in path:
        score += 3
    return score


def _discover_pricing_like_urls(page, root_url: str, limit: int = 8):
    """
    Step 2 — smarter discovery:
    Scan homepage links and find likely pricing/plans/upgrade URLs.
    """
    root_url = _norm_url(root_url)

    anchors = page.eval_on_selector_all(
        "a[href]",
        """(els) => els.map(a => ({
            href: a.getAttribute('href') || '',
            text: (a.innerText || '').trim().slice(0, 80)
        }))"""
    )

    candidates = {}
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        abs_url = _norm_url(urljoin(root_url, href))
        if not abs_url or not _same_site(root_url, abs_url):
            continue
        if any(abs_url.lower().endswith(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg", ".zip")):
            continue

        score = _score_candidate(abs_url, a.get("text", ""))
        if score <= 0:
            continue
        candidates[abs_url] = max(score, candidates.get(abs_url, 0))

    # fallback guesses (low score so discovered links win)
    for fallback in [urljoin(root_url, "/pricing"), urljoin(root_url, "/plans"), urljoin(root_url, "/compare")]:
        fb = _norm_url(fallback)
        if _same_site(root_url, fb):
            candidates.setdefault(fb, 1)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [u for u, _ in ranked[:limit]]


def _try_dismiss_common_popups(page):
    """
    Best-effort: close cookie modals / newsletter popups.
    Safe if it does nothing.
    """
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass

    # Try common "accept/close" buttons (best effort)
    selectors = [
        "button:has-text('Accept')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('Got it')",
        "button:has-text('Close')",
        "button[aria-label*='close' i]",
        "[aria-label*='close' i]",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if btn and btn.is_visible(timeout=250):
                btn.click(timeout=500)
                break
        except Exception:
            continue


def _trigger_lazy_load(page):
    """
    Scroll through the full page in steps so intersection-observer-based
    lazy loaders fire for every image, then scroll back to top.
    """
    try:
        total_height = page.evaluate("document.body.scrollHeight")
        step = 600  # px per scroll step — roughly half a viewport
        y = 0
        while y < total_height:
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(80)
            y += step
        # Pause at bottom so final images can start loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(200)
        # Wait for all <img> elements to finish loading (5 s cap)
        try:
            page.wait_for_function(
                "() => [...document.images].every(img => img.complete)",
                timeout=5000,
            )
        except Exception:
            pass
        # Return to top for the full-page screenshot
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(150)
    except Exception:
        pass  # non-fatal — screenshot proceeds regardless


def _goto_robust(page, url: str):
    """
    Avoid 'networkidle' hangs:
    - goto(wait_until='domcontentloaded') with timeout
    - then bounded extra waits (load + small sleep)
    """
    notes = []
    t0 = time.time()

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    except PWTimeoutError:
        notes.append("goto_timeout_domcontentloaded")
    except Exception as e:
        notes.append(f"goto_error:{type(e).__name__}")

    # bounded wait for 'load' (doesn't hang forever)
    try:
        page.wait_for_load_state("load", timeout=8000)
    except Exception:
        notes.append("load_state_timeout")

    _try_dismiss_common_popups(page)

    try:
        page.wait_for_timeout(POST_GOTO_PAUSE_MS)
    except Exception:
        pass

    final_url = ""
    title = ""
    try:
        final_url = page.url
    except Exception:
        final_url = url
    try:
        title = page.title() or ""
    except Exception:
        title = ""

    elapsed_ms = int((time.time() - t0) * 1000)
    return final_url, title, elapsed_ms, ";".join(notes)


def main():
    if len(sys.argv) < 2:
        return {"shots": [], "pages": [], "images": [], "error": "No URL argument provided to capture.py"}

    url = _norm_url(sys.argv[1])
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    shots = []
    errors = []

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

        # 1) Load homepage first
        homepage = url.rstrip("/")
        final_url, title, elapsed_ms, notes = _goto_robust(page, homepage)
        try:
            _trigger_lazy_load(page)
            png = page.screenshot(full_page=True, type="png")
            img_url = png_bytes_to_jpeg_data_url(png)
            shots.append({
                "url": homepage,
                "final_url": final_url,
                "title": title,
                "notes": notes,
                "image": img_url,
                "elapsed_ms": elapsed_ms,
            })
        except Exception as e:
            errors.append(f"homepage_screenshot_failed:{e}")
            print(f"[capture.py] Homepage screenshot failed: {e}", file=sys.stderr)

        # 2) Discover pricing-like pages via homepage links (Step 2)
        discovered = []
        try:
            discovered = _discover_pricing_like_urls(page, homepage, limit=10)
        except Exception as e:
            errors.append(f"discovery_failed:{e}")
            print(f"[capture.py] Discovery failed: {e}", file=sys.stderr)

        # 3) Visit discovered pages and screenshot until max_images
        visited = set([homepage])
        for cand in discovered:
            if len(shots) >= max_images:
                break
            if cand in visited:
                continue
            visited.add(cand)

            try:
                final_url, title, elapsed_ms, notes = _goto_robust(page, cand)
                _trigger_lazy_load(page)
                png = page.screenshot(full_page=True, type="png")
                img_url = png_bytes_to_jpeg_data_url(png)
                shots.append({
                    "url": cand,
                    "final_url": final_url,
                    "title": title,
                    "notes": notes,
                    "image": img_url,
                    "elapsed_ms": elapsed_ms,
                })
            except Exception as e:
                errors.append(f"capture_failed:{cand}:{e}")
                print(f"[capture.py] Failed page: {cand} :: {e}", file=sys.stderr)
                continue

        context.close()

        # Mobile screenshot — always homepage only, labelled so the AI knows
        mobile_context = browser.new_context(
            viewport={"width": 390, "height": 844},
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            ),
            is_mobile=True,
            has_touch=True,
        )
        mobile_page = mobile_context.new_page()
        try:
            final_url_m, title_m, elapsed_m, notes_m = _goto_robust(mobile_page, homepage)
            _trigger_lazy_load(mobile_page)
            png_m = mobile_page.screenshot(full_page=True, type="png")
            shots.append({
                "url": homepage,
                "final_url": final_url_m,
                "title": title_m,
                "notes": f"mobile-viewport-390px{';' + notes_m if notes_m else ''}",
                "image": png_bytes_to_jpeg_data_url(png_m),
                "elapsed_ms": elapsed_m,
            })
        except Exception as e:
            errors.append(f"mobile_screenshot_failed:{e}")
            print(f"[capture.py] Mobile screenshot failed: {e}", file=sys.stderr)
        finally:
            mobile_context.close()

        browser.close()

    # Back-compat keys (so your app doesn't break):
    pages = [s.get("final_url") or s.get("url") for s in shots]
    images = [s.get("image") for s in shots]

    return {
        "root": homepage,
        "discovered_urls": discovered,
        "pages": pages,          # back-compat
        "images": images,        # back-compat
        "shots": shots,          # new (Step 1 UI)
        "errors": errors,
    }


if __name__ == "__main__":
    try:
        payload = main()
        print(json.dumps(payload))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"shots": [], "pages": [], "images": [], "error": str(e)}))
        print(f"[capture.py] ERROR: {e}", file=sys.stderr)
        sys.exit(0)