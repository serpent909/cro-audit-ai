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

NAV_CTA_HINTS = [
    "pricing", "plans", "plan", "subscriptions", "features", "feature",
    "product", "solutions", "solution", "demo", "contact", "about",
    "how", "tour", "overview", "services", "upgrade", "buy",
]
MAX_EXTRA_PAGES = 5   # additional pages beyond homepage (per viewport)


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

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [u for u, _ in ranked[:limit]]


def _discover_nav_pages(page, root_url: str, limit: int = MAX_EXTRA_PAGES) -> list[str]:
    """
    Discover all L1 pages likely to have CTAs: nav/header links score highest,
    pages matching NAV_CTA_HINTS score positively, DEBOOST_HINTS score negatively.
    Returns up to `limit` URLs (excluding the homepage itself).
    """
    root_norm = _norm_url(root_url).rstrip("/")

    # Collect raw hrefs from nav/header for baseline bonus
    try:
        nav_hrefs: set[str] = set(page.eval_on_selector_all(
            "nav a[href], header a[href]",
            "els => els.map(a => (a.getAttribute('href') || '').trim())"
        ))
    except Exception:
        nav_hrefs = set()

    try:
        anchors = page.eval_on_selector_all(
            "a[href]",
            """(els) => els.map(a => ({
                href: a.getAttribute('href') || '',
                text: (a.innerText || '').trim().slice(0, 80)
            }))"""
        )
    except Exception:
        return []

    candidates: dict[str, int] = {}
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        abs_url = _norm_url(urljoin(root_url, href)).rstrip("/")
        if not abs_url or not _same_site(root_url, abs_url):
            continue
        if abs_url == root_norm:
            continue
        if any(abs_url.lower().endswith(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg", ".zip", ".svg")):
            continue

        combined = (abs_url + " " + (a.get("text") or "")).lower()
        score = 0
        if href in nav_hrefs:
            score += 3
        for hint in NAV_CTA_HINTS:
            if hint in combined:
                score += 4
        for bad in DEBOOST_HINTS:
            if bad in combined:
                score -= 4

        if score > 0:
            candidates[abs_url] = max(score, candidates.get(abs_url, 0))

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [u for u, _ in ranked[:limit]]


def _detect_overlay(page) -> bool:
    """
    Return True if a large fixed/absolute element is covering >40% of the viewport
    with a z-index >= 10 (i.e. a modal / promotional overlay).
    """
    try:
        return page.evaluate("""() => {
            const vw = window.innerWidth, vh = window.innerHeight;
            const minArea = vw * vh * 0.40;
            for (const el of document.querySelectorAll('*')) {
                const s = window.getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden') continue;
                if (parseFloat(s.opacity || '1') < 0.1) continue;
                if (parseInt(s.zIndex) < 10) continue;
                if (s.position !== 'fixed' && s.position !== 'absolute') continue;
                const r = el.getBoundingClientRect();
                if (r.width * r.height >= minArea) return true;
            }
            return false;
        }""")
    except Exception:
        return False


def _force_dismiss_overlay(page) -> None:
    """
    Last resort: remove large fixed/absolute overlay elements from the DOM
    and restore any body overflow lock the popup may have applied.
    """
    try:
        page.evaluate("""() => {
            const vw = window.innerWidth, vh = window.innerHeight;
            const minArea = vw * vh * 0.40;
            const toRemove = [];
            for (const el of document.querySelectorAll('*')) {
                const s = window.getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden') continue;
                if (parseFloat(s.opacity || '1') < 0.1) continue;
                if (parseInt(s.zIndex) < 10) continue;
                if (s.position !== 'fixed' && s.position !== 'absolute') continue;
                const r = el.getBoundingClientRect();
                if (r.width * r.height >= minArea) toRemove.push(el);
            }
            toRemove.forEach(el => el.remove());
            document.body.style.overflow = '';
            document.documentElement.style.overflow = '';
        }""")
    except Exception:
        pass


def _try_dismiss_common_popups(page):
    """
    Best-effort: close cookie modals / newsletter / promotional popups.
    Tries Escape first, then a broad set of dismiss button patterns,
    then a backdrop click. Safe if it does nothing.
    """
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass

    selectors = [
        # Accept / consent
        "button:has-text('Accept')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('Got it')",
        # Close
        "button:has-text('Close')",
        "button:has-text('Dismiss')",
        "button:has-text('×')",
        "button:has-text('X')",
        # Soft-dismiss (email / promo popups)
        "button:has-text('No thanks')",
        "button:has-text('No, thanks')",
        "button:has-text('Maybe later')",
        "button:has-text('Not now')",
        "button:has-text('Skip')",
        "a:has-text('No thanks')",
        "a:has-text('No, thanks')",
        "a:has-text('Maybe later')",
        "a:has-text('Not now')",
        "a:has-text('Skip')",
        # ARIA labels
        "button[aria-label*='close' i]",
        "[aria-label*='close' i]",
        "[aria-label*='dismiss' i]",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if btn and btn.is_visible(timeout=250):
                btn.click(timeout=500)
                break
        except Exception:
            continue

    # Backdrop click — click a corner in case the overlay closes on outside-click
    try:
        page.mouse.click(10, 10)
    except Exception:
        pass


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
    Returns (final_url, title, elapsed_ms, notes, http_status).
    http_status is 0 if the navigation failed or timed out.
    """
    notes = []
    t0 = time.time()
    http_status = 0

    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        if resp is not None:
            http_status = resp.status
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
    return final_url, title, elapsed_ms, ";".join(notes), http_status


def _take_pricing_shot(page, homepage: str, pricing_url: str | None) -> dict | None:
    """
    Find and screenshot the pricing/plans section (legacy — not called from main).
    Kept for reference; main() now uses _discover_nav_pages loop instead.
    """
    # CSS selectors for common pricing/plans containers
    pricing_selectors = [
        "#pricing", "#plans", "#packages", "#pricing-section",
        "#pricing-table", "#plan", "#subscription",
        "[id*='pricing']", "[id*='plans']", "[id*='plan-']",
        ".pricing", ".plans", ".pricing-section", ".pricing-table",
        ".pricing-plans", ".plan-list", ".plan-cards",
        "[class*='pricing-section']", "[class*='pricing-table']",
        "[class*='plans-section']", "[data-section*='pric']",
        "section:has(h2:text-matches('(?i)pricing|plans|packages|subscription'))",
    ]

    pricing_found = False

    for sel in pricing_selectors:
        try:
            el = page.locator(sel).first
            if el.count() > 0 and el.is_visible(timeout=600):
                el.scroll_into_view_if_needed()
                page.wait_for_timeout(350)
                pricing_found = True
                break
        except Exception:
            pass

    if not pricing_found:
        # Try any heading whose text contains a pricing keyword
        for keyword in ("Pricing", "Plans", "Packages", "Subscriptions", "Tiers"):
            try:
                heading = page.locator(
                    f'h1:has-text("{keyword}"), h2:has-text("{keyword}"), h3:has-text("{keyword}")'
                ).first
                if heading.count() > 0 and heading.is_visible(timeout=600):
                    heading.scroll_into_view_if_needed()
                    page.wait_for_timeout(350)
                    pricing_found = True
                    break
            except Exception:
                pass

    if pricing_found:
        try:
            png = page.screenshot(full_page=True, type="png")
            return {
                "url": page.url,
                "final_url": page.url,
                "title": page.title() or "",
                "notes": "pricing-section",
                "image": png_bytes_to_jpeg_data_url(png),
                "elapsed_ms": 0,
            }
        except Exception:
            pass

    # Fall back: navigate to the dedicated pricing page if different from homepage
    if pricing_url and _norm_url(pricing_url) != _norm_url(homepage):
        try:
            final_url, title, elapsed_ms, notes, http_status = _goto_robust(page, pricing_url)
            if http_status >= 400:
                return None
            _trigger_lazy_load(page)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(200)
            png = page.screenshot(full_page=True, type="png")
            return {
                "url": pricing_url,
                "final_url": final_url,
                "title": title,
                "notes": f"pricing-page{';' + notes if notes else ''}",
                "image": png_bytes_to_jpeg_data_url(png),
                "elapsed_ms": elapsed_ms,
            }
        except Exception:
            pass

    return None


def _take_page_shots(
    page, url: str, final_url: str, title: str,
    elapsed_ms: int, nav_notes: str, notes_prefix: str,
    shots: list, errors: list,
) -> None:
    """
    After _goto_robust has already navigated to the page:
    1. Wait 2 s for delayed/triggered popups.
    2. If an overlay is detected, take a viewport screenshot tagged ';overlay'
       then attempt a graduated dismissal (button click → backdrop click → DOM removal).
    3. Trigger lazy-load, scroll to top, take the clean full-page screenshot.
    All shots are appended to `shots` in-place.
    """
    base_notes = f"{notes_prefix}{';' + nav_notes if nav_notes else ''}"

    # Wait for any time-delayed popup to appear
    try:
        page.wait_for_timeout(2000)
    except Exception:
        pass

    if _detect_overlay(page):
        # --- Overlay screenshot (viewport only — captures the popup as presented) ---
        try:
            png_ov = page.screenshot(full_page=False, type="png")
            shots.append({
                "url": url,
                "final_url": final_url,
                "title": title,
                "notes": base_notes + ";overlay",
                "image": png_bytes_to_jpeg_data_url(png_ov),
                "elapsed_ms": elapsed_ms,
            })
        except Exception as e:
            errors.append(f"overlay_shot_failed:{url}:{e}")

        # --- Graduated dismissal ---
        _try_dismiss_common_popups(page)
        try:
            page.wait_for_timeout(400)
        except Exception:
            pass
        if _detect_overlay(page):
            _force_dismiss_overlay(page)
            try:
                page.wait_for_timeout(300)
            except Exception:
                pass

    # --- Clean full-page screenshot ---
    try:
        _trigger_lazy_load(page)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(200)
        png = page.screenshot(full_page=True, type="png")
        shots.append({
            "url": url,
            "final_url": final_url,
            "title": title,
            "notes": base_notes,
            "image": png_bytes_to_jpeg_data_url(png),
            "elapsed_ms": elapsed_ms,
        })
    except Exception as e:
        errors.append(f"clean_shot_failed:{url}:{e}")


def _path_slug(url: str) -> str:
    """Derive a short label from a URL path, e.g. /pricing -> 'pricing'."""
    path = urlparse(url).path.strip("/").replace("/", "-")
    return path or "home"


def main():
    if len(sys.argv) < 2:
        return {"shots": [], "pages": [], "images": [], "error": "No URL argument provided to capture.py"}

    url = _norm_url(sys.argv[1])

    shots: list  = []
    errors: list = []
    discovered: list[str] = []  # L1 pages, shared between desktop and mobile passes

    with sync_playwright() as pw:
        browser = _try_launch_browser(pw)
        homepage = url.rstrip("/")

        # ── DESKTOP ──────────────────────────────────────────────────────────
        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
        )
        page = context.new_page()
        try:
            # Desktop homepage
            final_url, title, elapsed_ms, notes, _status = _goto_robust(page, homepage)
            _take_page_shots(page, homepage, final_url, title, elapsed_ms, notes,
                             "desktop-homepage", shots, errors)

            # Discover all relevant L1 pages (run after homepage is stable/clean)
            try:
                discovered = _discover_nav_pages(page, homepage, limit=MAX_EXTRA_PAGES)
                print(f"[capture.py] Discovered {len(discovered)} L1 pages: {discovered}", file=sys.stderr)
            except Exception as e:
                errors.append(f"discovery_failed:{e}")
                print(f"[capture.py] Discovery failed: {e}", file=sys.stderr)

            # Screenshots for each discovered page (desktop)
            for purl in discovered:
                try:
                    slug = _path_slug(purl)
                    p_final, p_title, p_elapsed, p_notes, p_status = _goto_robust(page, purl)
                    if p_status >= 400:
                        continue
                    _take_page_shots(page, purl, p_final, p_title, p_elapsed, p_notes,
                                     f"desktop-{slug}", shots, errors)
                except Exception as e:
                    errors.append(f"desktop_page_failed:{purl}:{e}")
                    print(f"[capture.py] Desktop page failed ({purl}): {e}", file=sys.stderr)

        except Exception as e:
            errors.append(f"desktop_failed:{e}")
            print(f"[capture.py] Desktop capture failed: {e}", file=sys.stderr)
        finally:
            context.close()

        # ── MOBILE ───────────────────────────────────────────────────────────
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
            # Mobile homepage
            final_url_m, title_m, elapsed_m, notes_m, _status_m = _goto_robust(mobile_page, homepage)
            _take_page_shots(mobile_page, homepage, final_url_m, title_m, elapsed_m, notes_m,
                             "mobile-homepage", shots, errors)

            # Screenshots for each discovered page (mobile)
            for purl in discovered:
                try:
                    slug = _path_slug(purl)
                    p_final, p_title, p_elapsed, p_notes, p_status = _goto_robust(mobile_page, purl)
                    if p_status >= 400:
                        continue
                    _take_page_shots(mobile_page, purl, p_final, p_title, p_elapsed, p_notes,
                                     f"mobile-{slug}", shots, errors)
                except Exception as e:
                    errors.append(f"mobile_page_failed:{purl}:{e}")
                    print(f"[capture.py] Mobile page failed ({purl}): {e}", file=sys.stderr)

        except Exception as e:
            errors.append(f"mobile_failed:{e}")
            print(f"[capture.py] Mobile capture failed: {e}", file=sys.stderr)
        finally:
            mobile_context.close()

        browser.close()

    pages  = [s.get("final_url") or s.get("url") for s in shots]
    images = [s.get("image") for s in shots]

    return {
        "root": homepage,
        "discovered_urls": discovered,
        "pages": pages,
        "images": images,
        "shots": shots,
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