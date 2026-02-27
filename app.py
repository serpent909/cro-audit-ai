# app.py
import json
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI

# ----------------------------
# Load environment (.env / Streamlit Secrets)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except Exception:
    pass


# ----------------------------
# Config
# ----------------------------
MODEL_TEXT = "gpt-4.1-mini"
MODEL_VISION = "gpt-4.1"  # full model for vision quality

MAX_CHARS = 20000
MAX_IMAGES = 3

PAGESPEED_API = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
PSI_RUNS = 3  # parallel runs per strategy; median is taken to reduce variability

# Core Web Vitals thresholds for colour-coding in the UI
CWV_THRESHOLDS = {
    "lcp":          {"good": 2500,  "poor": 4000,  "unit": "ms", "label": "LCP"},
    "fcp":          {"good": 1800,  "poor": 3000,  "unit": "ms", "label": "FCP"},
    "cls":          {"good": 0.1,   "poor": 0.25,  "unit": "",   "label": "CLS"},
    "tbt":          {"good": 200,   "poor": 600,   "unit": "ms", "label": "TBT"},
    "inp":          {"good": 200,   "poor": 500,   "unit": "ms", "label": "INP"},
    "ttfb":         {"good": 800,   "poor": 1800,  "unit": "ms", "label": "TTFB"},
    "speed_index":  {"good": 3400,  "poor": 5800,  "unit": "ms", "label": "Speed Index"},
}

SCRAPE_PATHS = [
    "",
    "/pricing",
    "/plans",
    "/compare",
    "/features",
    "/product",
    "/about",
    "/signup",
    "/register",
]

# Keywords used to score homepage links during URL discovery
DISCOVERY_HINTS = [
    # Pricing / subscription signals
    "pricing", "plans", "plan", "subscription", "subscriptions", "billing",
    "upgrade", "tiers", "compare",
    # E-commerce / purchase signals
    "shop", "store", "product", "products", "collection", "collections",
    "buy", "purchase", "order", "packages", "package", "bundle", "bundles",
    "offer", "offers",
]
DISCOVERY_DEBOOST = [
    "blog", "docs", "help", "support", "changelog", "status",
    "careers", "jobs", "news", "press", "privacy", "terms", "faq", "cookie",
]
MAX_DISCOVERED = 8  # max additional pages from link discovery

_PROMPT_RULES = """
You are a senior CRO (Conversion Rate Optimization) consultant. Produce an audit that is specific, evidence-based, and action-oriented.

NON-NEGOTIABLE RULES
- Only make claims supported by the provided text or screenshots.
- If you cannot confirm something, write: "Not observed in provided content."
- Every issue MUST include Evidence (quote text OR describe what is visible and where: 'hero area', 'header nav', 'pricing table', 'product page', etc.).
- If CTA candidates exist, list them and evaluate them (do not claim no CTAs).
- Always evaluate what is visible ABOVE THE FOLD (without scrolling) separately from below-the-fold content.
- Compare CTAs and messaging across pages — flag any inconsistencies.
- If PageSpeed data is provided, use it to inform the Friction and Mobile scorecard scores and flag any Poor Core Web Vitals (LCP >4s, CLS >0.25, INP >500ms, TTFB >1.8s) as concrete conversion issues with their impact on bounce rate and user experience.
- IMPORTANT — screenshots only show ONE STATE of interactive elements. Carousels, sliders, tabs, and accordions display only one slide/panel/item at a time in a screenshot. Always cross-reference with the TEXT CONTEXT and INTERACTIVE ELEMENTS section, which contains ALL DOM content including hidden slides. Never report "only 1 testimonial" if the text context or INTERACTIVE ELEMENTS section shows a carousel or multiple items.
"""

PROMPT_SAAS = _PROMPT_RULES + """
SITE TYPE: SaaS / Software product
Focus on: free trial / demo conversion, pricing plan clarity, signup friction, feature differentiation, and trust signals relevant to software buyers (security, integrations, customer logos, case studies).

OUTPUT FORMAT (use headings exactly)

## 0) Business context
In 1–2 sentences: identify the SaaS category, likely buyer (SMB / mid-market / enterprise), and primary conversion goal (free trial / demo / paid signup). Use this to inform all findings.

## 1) Funnel map
List each page captured, its apparent purpose, and the primary CTA on that page (trial / demo / signup / upgrade etc.).

## 2) Executive summary
- Biggest conversion blocker (what stops someone starting a trial or booking a demo?)
- Biggest trust/credibility gap (what makes a buyer hesitate to hand over card details or data?)
- Biggest messaging/positioning gap (is the value prop differentiated from generic competitors?)
- Biggest friction point (signup steps, pricing confusion, missing integrations info?)
- Highest-impact quick win
- CTA consistency verdict: do homepage promise, pricing page, and signup page tell the same story?

## 3) Conversion scorecard (weighted)
Score each dimension 0–10 with one sentence of rationale. Then compute a weighted overall score.

| Dimension | Score /10 | Weight | Weighted |
|---|---|---|---|
| Value prop & differentiation | ? | 2× | |
| Trial / demo CTA clarity | ? | 2× | |
| Pricing plan clarity | ? | 1.5× | |
| Trust & social proof (logos, case studies, reviews) | ? | 1.5× | |
| Signup / onboarding friction | ? | 1× | |
| Feature communication | ? | 1× | |
| Mobile experience | ? | 1× | |

Overall score = sum(weighted) / 10 → show as **X.X / 10**

## 4) Above-the-fold analysis
For EACH page captured: what does a visitor see before scrolling? Is the value prop and differentiator clear? Is a trial/demo CTA visible? What is missing or unclear from the first impression?

## 5) Top 7 issues (ranked by Impact × Confidence)
For each issue:
- **Issue** (one line)
- Impact: High/Med/Low
- Effort: S/M/L
- Confidence: High/Med/Low
- Evidence (quote or describe exact location on page)
- Recommendation (specific, actionable)
- Test idea: hypothesis → primary metric → guardrail metric

## 6) Mobile considerations
Identify 3–5 mobile-specific CRO risks for SaaS: demo/trial CTA tap targets, pricing table horizontal scroll, form field usability on mobile keyboards, sticky header CTAs, load speed on 4G. Base on visual evidence if a mobile screenshot is provided.

## 7) Copy & CTA improvements (write the actual copy)
- 3 improved headline/value prop options (include a one-line rationale for each, focused on outcome or pain relief for the target buyer)
- 3 improved primary CTA label options (e.g. "Start free trial", "See it in action", "Get your free account")
- 3 trust microcopy examples to place near the primary CTA or signup form (e.g. "No credit card required · Cancel anytime")

## 8) Experiment plan (2 weeks)
Week 1: 2 quick wins (low effort, high confidence) — e.g. CTA label, headline, trust badge placement
Week 2: 2 bigger tests — e.g. pricing page layout, social proof section, trial vs demo CTA
For each test: hypothesis | primary metric | guardrail metric | minimum detectable effect
"""

PROMPT_ECOMMERCE = _PROMPT_RULES + """
SITE TYPE: E-commerce / Online store
Focus on: product clarity, Add-to-Cart / Buy Now conversion, price anchoring, trust signals relevant to online shoppers (reviews, returns, shipping, payment options), and reducing cart abandonment.

OUTPUT FORMAT (use headings exactly)

## 0) Business context
In 1–2 sentences: identify the product category/niche, likely customer (demographics, intent level), and primary conversion goal (first purchase / repeat purchase / subscription). Use this to inform all findings.

## 1) Funnel map
List each page captured, its apparent purpose, and the primary CTA (Shop Now / Add to Cart / Buy Now / View Collection etc.).

## 2) Executive summary
- Biggest conversion blocker (what stops someone adding to cart or completing checkout?)
- Biggest trust gap (what makes a visitor hesitant to buy — no reviews, unclear returns, unfamiliar brand?)
- Biggest product/value clarity gap (is it clear what is being sold, what it does, and why it is worth the price?)
- Biggest checkout/cart friction point
- Highest-impact quick win
- CTA consistency verdict: do homepage, collection pages, and product pages tell a consistent purchase story?

## 3) Conversion scorecard (weighted)
Score each dimension 0–10 with one sentence of rationale. Then compute a weighted overall score.

| Dimension | Score /10 | Weight | Weighted |
|---|---|---|---|
| Product clarity & imagery | ? | 2× | |
| Buy / Add-to-Cart CTA prominence | ? | 2× | |
| Trust signals (reviews, ratings, returns) | ? | 1.5× | |
| Price anchoring & value perception | ? | 1.5× | |
| Cart & checkout friction | ? | 1× | |
| Mobile shopping experience | ? | 1× | |
| Delivery, returns & payment visibility | ? | 1× | |

Overall score = sum(weighted) / 10 → show as **X.X / 10**

## 4) Above-the-fold analysis
For EACH page captured: what does a visitor see before scrolling? Is the product/offer immediately clear? Is a Buy/Add-to-Cart CTA visible? Is there any price, social proof, or urgency signal above the fold? What is missing?

## 5) Top 7 issues (ranked by Impact × Confidence)
For each issue:
- **Issue** (one line)
- Impact: High/Med/Low
- Effort: S/M/L
- Confidence: High/Med/Low
- Evidence (quote or describe exact location: 'product hero', 'below product images', 'cart drawer', etc.)
- Recommendation (specific, actionable — e.g. "Add star rating summary directly below product title")
- Test idea: hypothesis → primary metric → guardrail metric

## 6) Mobile shopping considerations
Identify 3–5 mobile-specific CRO risks for e-commerce: thumb-friendly Add-to-Cart button size and position, product image pinch-zoom or swipe gallery, sticky buy button on scroll, checkout autofill, payment method visibility (Apple Pay / Google Pay), load speed impact on mobile shoppers. Base on visual evidence if a mobile screenshot is provided.

## 7) Copy & CTA improvements (write the actual copy)
- 3 improved product headline / hero copy options (focus on outcome, transformation, or key differentiator — not just product name)
- 3 improved primary CTA label options (tailored to purchase intent: e.g. "Add to Cart", "Buy Now — Ships in 24h", "Get Yours Today")
- 3 trust microcopy examples to place near the buy button or at checkout (e.g. "Free returns within 30 days · Secure checkout")

## 8) Experiment plan (2 weeks)
Week 1: 2 quick wins — e.g. CTA label, trust badge near buy button, review summary above the fold
Week 2: 2 bigger tests — e.g. product page layout, urgency/scarcity messaging, price anchoring (was/now), image gallery vs video
For each test: hypothesis | primary metric | guardrail metric | minimum detectable effect
"""


def _get_prompt(site_type: str) -> str:
    return PROMPT_ECOMMERCE if site_type == "E-commerce" else PROMPT_SAAS


# ----------------------------
# OpenAI client helper
# ----------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OPENAI_API_KEY not found.\n\n"
            "Set it using one of these:\n"
            "1) Streamlit Cloud → App → Settings → Secrets:\n"
            '   OPENAI_API_KEY="sk-..."\n'
            "2) Local: create a .env file next to app.py containing:\n"
            "   OPENAI_API_KEY=sk-...\n"
            "3) Local: export OPENAI_API_KEY in your shell.\n\n"
            f"Checked for .env at: {ENV_PATH}"
        )
        st.stop()

    return OpenAI(api_key=api_key)


# ----------------------------
# Helpers
# ----------------------------
def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def discover_urls_from_homepage(homepage_html: str, base_url: str) -> list[str]:
    """
    Parse homepage links and return candidate URLs ranked by pricing/shop relevance.
    Nav/header links receive a base score so they're included even without keyword matches —
    this catches non-standard pricing URLs like /pages/our-creams on Shopify stores.
    """
    soup = BeautifulSoup(homepage_html, "lxml")
    base_netloc = urlparse(base_url).netloc.lower()
    base_norm = base_url.rstrip("/")

    # Collect raw hrefs from nav/header elements for bonus scoring
    nav_hrefs: set[str] = set()
    for container in soup.find_all(["nav", "header"]):
        for a in container.find_all("a", href=True):
            nav_hrefs.add((a.get("href") or "").strip())

    candidates: dict[str, int] = {}

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)

        if parsed.netloc.lower() != base_netloc:
            continue
        if any(parsed.path.lower().endswith(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg", ".zip", ".svg")):
            continue

        # Normalise: drop fragment and query string
        norm = parsed._replace(fragment="", query="").geturl().rstrip("/")
        if norm == base_norm:
            continue  # skip homepage itself

        text = a.get_text(" ", strip=True).lower()
        path = parsed.path.lower().strip("/")
        combined = norm.lower() + " " + text

        score = 0
        if href in nav_hrefs:
            score += 3  # nav baseline: nav links are included unless deboost wins
        for hint in DISCOVERY_HINTS:
            if hint in combined:
                score += 4
        for bad in DISCOVERY_DEBOOST:
            if bad in combined:
                score -= 4

        # Extra bonus for known e-commerce/pricing exact paths
        if path in ("pricing", "plans", "shop", "products", "collections", "store", "buy"):
            score += 6

        if score > 0:
            candidates[norm] = max(score, candidates.get(norm, 0))

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [u for u, _ in ranked[:MAX_DISCOVERED]]


def extract_from_single_page(page_url: str, headers: dict, html: str | None = None) -> str | None:
    try:
        if html is None:
            r = requests.get(page_url, headers=headers, timeout=15)
            if r.status_code != 200:
                return None
            html = r.text

        full = BeautifulSoup(html, "lxml")

        # Meta signals
        title_tag = full.find("title")
        title_text = title_tag.get_text(strip=True) if title_tag else ""

        meta_desc = full.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        meta_desc_text = (meta_desc.get("content") or "").strip() if meta_desc else ""

        # Main readable text
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        main_text = soup.get_text("\n", strip=True)

        # CTAs
        ctas = []
        for b in full.find_all("button"):
            txt = b.get_text(" ", strip=True)
            if txt:
                ctas.append(f"BUTTON: {txt}")

        for a in full.find_all("a"):
            txt = a.get_text(" ", strip=True)
            href = (a.get("href") or "").strip()
            if txt and any(
                k in txt.lower()
                for k in [
                    "book", "demo", "start", "get", "try", "sign", "contact",
                    "pricing", "plans", "buy", "join", "trial",
                ]
            ):
                ctas.append(f"LINK: {txt} -> {href}")

        ctas = list(dict.fromkeys(ctas))[:40]

        # Form analysis
        form_summaries = []
        for form in full.find_all("form"):
            fields = form.find_all(["input", "select", "textarea"])
            visible = [f for f in fields if f.get("type") not in ("hidden", "submit", "button")]
            labels = [lbl.get_text(strip=True) for lbl in form.find_all("label")]
            placeholders = [f.get("placeholder", "") for f in visible if f.get("placeholder")]
            form_summaries.append(
                f"{len(visible)} visible field(s) | labels: {labels[:8]} | placeholders: {placeholders[:8]}"
            )

        # Social proof — extract ALL items including hidden carousel slides
        # Skip container elements (those that hold 2+ matching children) to avoid
        # double-counting parent + children. This correctly counts each slide once.
        proof_keywords = [
            "testimonial", "review", "rating", "stars", "trustpilot",
            "g2", "capterra", "case-study", "customers", "clients",
        ]
        proof_elements = []
        seen_proof: set[str] = set()
        for kw in proof_keywords:
            all_matches = full.find_all(class_=re.compile(kw, re.I))
            items = []
            for el in all_matches:
                # Skip if this element is a container holding 2+ same-keyword children
                if len(el.find_all(class_=re.compile(kw, re.I))) >= 2:
                    continue
                txt = el.get_text(" ", strip=True)
                if txt and 10 < len(txt) < 1000 and txt not in seen_proof:
                    seen_proof.add(txt)
                    items.append(txt[:250])
            if items:
                proof_elements.append(f"[{kw}] — {len(items)} item(s) found:")
                for t in items[:6]:
                    proof_elements.append(f"  • {t}")
                if len(items) > 6:
                    proof_elements.append(f"  (+ {len(items) - 6} more not shown)")

        # Interactive elements — carousels, tabs, accordions are only partially
        # visible in screenshots. Report counts so the AI has the full picture.
        interactive_notes = []

        _carousel_cls = re.compile(r"(swiper|slick|carousel|slider|splide|glide|flickity|owl)", re.I)
        _slide_cls    = re.compile(r"(slide|swiper-slide|slick-slide|carousel-item|splide__slide)", re.I)
        for container in full.find_all(class_=_carousel_cls)[:5]:
            slides = container.find_all(class_=_slide_cls)
            if len(slides) >= 2:
                interactive_notes.append(
                    f"Carousel/slider with {len(slides)} slides — screenshots show only 1 at a time"
                )
                break  # report once per page

        tab_panels = full.find_all(attrs={"role": "tabpanel"})
        if len(tab_panels) >= 2:
            interactive_notes.append(
                f"{len(tab_panels)} tab panels — screenshots show only the active tab"
            )

        details_els = full.find_all("details")
        if len(details_els) >= 2:
            interactive_notes.append(
                f"{len(details_els)} expandable accordion/FAQ items — most are collapsed in screenshots"
            )

        # Trust signals
        trust_signals = []
        for img in full.find_all("img"):
            alt = (img.get("alt") or "").lower()
            if any(t in alt for t in ["ssl", "secure", "guarantee", "certified", "award", "badge", "verified"]):
                trust_signals.append(f"IMG ALT: {alt}")

        phones = re.findall(r'\+?[\d\s\-\(\)]{10,16}', full.get_text())
        if phones:
            trust_signals.append(f"Phone number present: {phones[0].strip()}")

        return f"""
PAGE: {page_url}
TITLE: {title_text}
META DESCRIPTION: {meta_desc_text}

TEXT:
{main_text}

CTAs:
{chr(10).join(ctas) if ctas else "None detected"}

FORMS:
{chr(10).join(form_summaries) if form_summaries else "No forms detected"}

SOCIAL PROOF:
{chr(10).join(proof_elements) if proof_elements else "None detected"}

INTERACTIVE ELEMENTS (not fully visible in screenshots):
{chr(10).join(interactive_notes) if interactive_notes else "None detected"}

TRUST SIGNALS:
{chr(10).join(trust_signals[:10]) if trust_signals else "None detected"}
"""

    except Exception:
        return None


def extract_text_from_url(url: str) -> tuple[str, list[str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    base = url.rstrip("/")

    # Phase 1: Fetch homepage once — used for both discovery and content extraction
    homepage_html: str | None = None
    try:
        r = requests.get(base, headers=headers, timeout=15)
        if r.status_code == 200:
            homepage_html = r.text
    except Exception:
        pass

    # Phase 2: Discover pricing/shop/product URLs from homepage navigation links
    discovered: list[str] = []
    if homepage_html:
        discovered = discover_urls_from_homepage(homepage_html, base)

    # Phase 3: Build deduped URL list
    # Hardcoded paths (excluding "" since homepage already fetched), then discovered
    other_hardcoded = [base + p for p in SCRAPE_PATHS if p != ""]
    all_other = list(dict.fromkeys(other_hardcoded + discovered))

    def fetch(page_url: str) -> tuple[str, str | None]:
        return page_url, extract_from_single_page(page_url, headers)

    results: dict[str, str] = {}

    # Homepage content from cached HTML (no extra request)
    if homepage_html:
        hp_content = extract_from_single_page(base, headers, html=homepage_html)
        if hp_content:
            results[base] = hp_content

    # Fetch remaining pages in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch, u): u for u in all_other}
        for future in as_completed(futures):
            page_url, content = future.result()
            if content:
                results[page_url] = content

    # Assemble in stable order: homepage → hardcoded paths → discovered
    ordered = [base] + all_other
    bundles = []
    scraped_pages = []
    for u in ordered:
        if u in results and u not in scraped_pages:
            bundles.append(results[u])
            scraped_pages.append(u)

    if not bundles:
        return "No content extracted.", []

    combined = "\n\n---\n\n".join(bundles)
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()
    return combined, scraped_pages


def _cwv_label(key: str, raw_value) -> str:
    """Return display string with Good/Needs improvement/Poor label."""
    if raw_value is None:
        return "N/A"
    t = CWV_THRESHOLDS.get(key)
    if not t:
        return str(raw_value)
    try:
        v = float(raw_value)
    except (TypeError, ValueError):
        return str(raw_value)
    if v <= t["good"]:
        status = "Good"
    elif v <= t["poor"]:
        status = "Needs improvement"
    else:
        status = "Poor"
    display = f"{v / 1000:.2f} s" if t["unit"] == "ms" and v > 10 else str(v)
    return f"{display} ({status})"


def _parse_lhr(lhr: dict) -> dict:
    """Extract key metrics from a Lighthouse result dict."""
    audits = lhr.get("audits", {})
    perf_score = lhr.get("categories", {}).get("performance", {}).get("score")

    def ms(key: str):
        n = audits.get(key, {}).get("numericValue")
        return round(n) if n is not None else None

    def display(key: str):
        return audits.get(key, {}).get("displayValue", "N/A")

    def fmt_ms(n):
        """Format a millisecond numeric value into a short display string."""
        if n is None:
            return "N/A"
        return f"{n / 1000:.2f} s" if n >= 1000 else f"{round(n)} ms"

    ttfb_n = ms("server-response-time")

    return {
        "score": int(perf_score * 100) if perf_score is not None else None,
        "lcp_ms":       ms("largest-contentful-paint"),
        "fcp_ms":       ms("first-contentful-paint"),
        "cls_raw":      audits.get("cumulative-layout-shift", {}).get("numericValue"),
        "tbt_ms":       ms("total-blocking-time"),
        "inp_ms":       ms("interaction-to-next-paint"),
        "ttfb_ms":      ttfb_n,
        "speed_ms":     ms("speed-index"),
        # display strings (already formatted by Lighthouse, except TTFB which
        # returns the full sentence "Root document took X ms" — format it ourselves)
        "lcp":          display("largest-contentful-paint"),
        "fcp":          display("first-contentful-paint"),
        "cls":          display("cumulative-layout-shift"),
        "tbt":          display("total-blocking-time"),
        "inp":          display("interaction-to-next-paint"),
        "ttfb":         fmt_ms(ttfb_n),
        "speed_index":  display("speed-index"),
    }


def fetch_pagespeed(url: str) -> tuple[str, dict]:
    """
    Fetch PageSpeed Insights for mobile + desktop in parallel.
    Runs PSI_RUNS times per strategy concurrently and takes the median of each
    numeric metric to reduce Lighthouse variability.
    Returns (ai_context_str, {mobile: metrics, desktop: metrics}).
    On total failure returns ("", {}).
    """
    api_key = os.getenv("PAGESPEED_API_KEY", "")

    def _fetch(strategy: str, _: int) -> tuple[str, dict | None]:
        params = {"url": url, "strategy": strategy, "category": "performance"}
        if api_key:
            params["key"] = api_key
        try:
            r = requests.get(PAGESPEED_API, params=params, timeout=60)
            if r.status_code != 200:
                return strategy, {"_http_error": r.status_code}
            data = r.json()
            lhr = data.get("lighthouseResult", {})
            return strategy, _parse_lhr(lhr) if lhr else None
        except Exception as e:
            return strategy, {"_http_error": str(e)}

    def _median_metrics(runs: list[dict]) -> dict:
        """Compute element-wise median across PSI runs and rebuild display strings."""
        def med(key):
            vals = [r[key] for r in runs if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        def fmt_ms(n):
            if n is None:
                return "N/A"
            return f"{n / 1000:.2f} s" if n >= 1000 else f"{round(n)} ms"

        score_vals = [r["score"] for r in runs if r.get("score") is not None]
        score = round(statistics.median(score_vals)) if score_vals else None

        lcp_ms  = med("lcp_ms")
        fcp_ms  = med("fcp_ms")
        cls_raw = med("cls_raw")
        tbt_ms  = med("tbt_ms")
        inp_ms  = med("inp_ms")
        ttfb_ms = med("ttfb_ms")
        speed_ms = med("speed_ms")

        return {
            "score":      score,
            "lcp_ms":     lcp_ms,
            "fcp_ms":     fcp_ms,
            "cls_raw":    cls_raw,
            "tbt_ms":     tbt_ms,
            "inp_ms":     inp_ms,
            "ttfb_ms":    ttfb_ms,
            "speed_ms":   speed_ms,
            # recompute display strings from median numerics
            "lcp":        fmt_ms(lcp_ms),
            "fcp":        fmt_ms(fcp_ms),
            "cls":        f"{cls_raw:.3f}" if cls_raw is not None else "N/A",
            "tbt":        fmt_ms(tbt_ms),
            "inp":        fmt_ms(inp_ms),
            "ttfb":       fmt_ms(ttfb_ms),
            "speed_index": fmt_ms(speed_ms),
            "_run_count": len(runs),
        }

    # Submit PSI_RUNS requests for each strategy in parallel (2 × PSI_RUNS total)
    runs_by_strategy: dict[str, list[dict]] = {"mobile": [], "desktop": []}
    psi_errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=PSI_RUNS * 2) as executor:
        futures = [
            executor.submit(_fetch, s, i)
            for s in ("mobile", "desktop")
            for i in range(PSI_RUNS)
        ]
        for future in as_completed(futures):
            strategy, metrics = future.result()
            if metrics and "_http_error" not in metrics:
                runs_by_strategy[strategy].append(metrics)
            else:
                err = (metrics or {}).get("_http_error", "no data returned")
                # Only keep the first error per strategy
                psi_errors.setdefault(strategy, str(err))

    raw: dict[str, dict] = {}
    for strategy, runs in runs_by_strategy.items():
        if runs:
            raw[strategy] = _median_metrics(runs)

    if not raw:
        return "", {"_errors": psi_errors}

    raw["_errors"] = psi_errors

    lines = [
        f"PAGE SPEED (Google PageSpeed Insights — median of {PSI_RUNS} runs — "
        "include in scorecard and mobile section)"
    ]
    for strategy in ("mobile", "desktop"):
        m = raw.get(strategy)
        if not m:
            lines.append(f"\n{strategy.capitalize()}: data unavailable")
            continue
        n = m.get("_run_count", PSI_RUNS)
        score = m["score"] if m["score"] is not None else "N/A"
        lines.append(f"\n{strategy.capitalize()} — Performance score: {score}/100 (median of {n} runs)")
        lines.append(f"  LCP:         {m['lcp']}  (Good <2.5s, Poor >4s)")
        lines.append(f"  FCP:         {m['fcp']}  (Good <1.8s, Poor >3s)")
        lines.append(f"  CLS:         {m['cls']}  (Good <0.1, Poor >0.25)")
        lines.append(f"  TBT:         {m['tbt']}  (Good <200ms, Poor >600ms)")
        lines.append(f"  INP:         {m['inp']}  (Good <200ms, Poor >500ms)")
        lines.append(f"  TTFB:        {m['ttfb']}  (Good <0.8s, Poor >1.8s)")
        lines.append(f"  Speed Index: {m['speed_index']}")

    return "\n".join(lines), raw


def run_ai_text(text: str, pagespeed: str = "", site_type: str = "SaaS") -> str:
    client = get_openai_client()
    prompt = _get_prompt(site_type)
    extra = f"\n\n{pagespeed}" if pagespeed else ""
    response = client.responses.create(
        model=MODEL_TEXT,
        input=f"{prompt}\n\n{text[:MAX_CHARS]}{extra}",
    )
    return response.output_text


def run_ai_vision(text_context: str, shots: list, pagespeed: str = "", site_type: str = "SaaS") -> str:
    client = get_openai_client()
    prompt = _get_prompt(site_type)

    visited_lines = []
    for s in shots:
        line = f"- {s.get('final_url') or s.get('url')}"
        if s.get("title"):
            line += f" | title: {s.get('title')}"
        if s.get("notes"):
            line += f" | notes: {s.get('notes')}"
        visited_lines.append(line)

    content = [
        {
            "type": "input_text",
            "text": (
                f"{prompt}\n\n"
                f"Visited pages (use these URLs when referencing evidence):\n"
                + "\n".join(visited_lines)
            ),
        }
    ]

    if pagespeed:
        content.append({"type": "input_text", "text": f"\n\n{pagespeed}"})

    if text_context:
        content.append(
            {
                "type": "input_text",
                "text": f"\n\nText context:\n{text_context[:MAX_CHARS]}",
            }
        )

    for s in shots:
        img = s.get("image")
        if img:
            content.append({"type": "input_image", "image_url": img})

    response = client.responses.create(
        model=MODEL_VISION,
        input=[{"role": "user", "content": content}],
    )

    return response.output_text


def take_auto_screenshots(url: str):
    cmd = [sys.executable, "capture.py", url, str(MAX_IMAGES)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    debug = {
        "cmd": " ".join(cmd),
        "python": sys.executable,
        "returncode": result.returncode,
        "stdout_preview": stdout[:2000],
        "stderr_preview": stderr[:2000],
    }

    if result.returncode != 0:
        raise RuntimeError(
            "capture.py failed\n"
            f"returncode: {result.returncode}\n\n"
            f"stderr (first 2000 chars):\n{debug['stderr_preview']}\n\n"
            f"stdout (first 2000 chars):\n{debug['stdout_preview']}"
        )

    try:
        data = json.loads(stdout)
    except Exception as e:
        raise RuntimeError(
            "capture.py did not return valid JSON.\n"
            f"JSON error: {e}\n\n"
            f"stdout (first 2000 chars):\n{debug['stdout_preview']}\n\n"
            f"stderr (first 2000 chars):\n{debug['stderr_preview']}"
        )

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(
            "capture.py returned an error payload:\n"
            f"{data.get('error')}\n\n"
            f"stderr (first 2000 chars):\n{debug['stderr_preview']}"
        )

    shots = data.get("shots") if isinstance(data, dict) else None
    if not shots:
        images = data.get("images", []) if isinstance(data, dict) else []
        pages = data.get("pages", []) if isinstance(data, dict) else []
        shots = [
            {"image": img, "url": p, "final_url": p, "title": "", "notes": ""}
            for img, p in zip(images, pages)
        ]

    return shots, debug, data


def render_shots_gallery(shots: list):
    st.subheader("Captured screenshots")
    for i, s in enumerate(shots, start=1):
        notes = (s.get("notes") or "").lower()
        is_mobile = "mobile" in notes
        label = f"Mobile — {s.get('url')}" if is_mobile else f"Page {i} — {s.get('url')}"

        cols = st.columns([1, 2])
        with cols[0]:
            st.image(s.get("image"), use_container_width=True)
        with cols[1]:
            st.markdown(f"**{label}**")
            if s.get("final_url") and s.get("final_url") != s.get("url"):
                st.markdown(f"**Final URL:** `{s.get('final_url')}`")
            if s.get("title"):
                st.markdown(f"**Title:** {s.get('title')}")
            if s.get("notes"):
                st.markdown(f"**Notes:** `{s.get('notes')}`")
        st.divider()


METRIC_HELP = {
    "lcp":  ("Largest Contentful Paint",  "How long until the biggest visible element (hero image, headline) fully loads. "
                                           "Slow LCP is the #1 reason visitors assume a page is broken and leave. "
                                           "🟢 Good: <2.5 s  🟡 Needs work: 2.5–4 s  🔴 Poor: >4 s"),
    "fcp":  ("First Contentful Paint",    "Time until any content first appears on screen. "
                                           "Sets the visitor's initial perception of speed — even a spinner counts. "
                                           "🟢 Good: <1.8 s  🟡 Needs work: 1.8–3 s  🔴 Poor: >3 s"),
    "cls":  ("Cumulative Layout Shift",   "Measures how much page elements jump around while loading. "
                                           "A high score means buttons/links move just as users try to click them, causing mis-clicks and frustration. "
                                           "🟢 Good: <0.1  🟡 Needs work: 0.1–0.25  🔴 Poor: >0.25"),
    "tbt":  ("Total Blocking Time",       "Total time the page is frozen and unresponsive to clicks after the first content loads. "
                                           "High TBT makes the page feel laggy even if it looks loaded. Usually caused by heavy JavaScript. "
                                           "🟢 Good: <200 ms  🟡 Needs work: 200–600 ms  🔴 Poor: >600 ms"),
    "inp":  ("Interaction to Next Paint", "How fast the page visually responds when a user clicks, taps, or types. "
                                           "A poor INP makes forms and buttons feel broken or unresponsive. "
                                           "🟢 Good: <200 ms  🟡 Needs work: 200–500 ms  🔴 Poor: >500 ms"),
    "ttfb": ("Time to First Byte",        "How quickly the server sends back the first byte of data after a request. "
                                           "A slow TTFB delays everything — no content can load until this completes. Usually a hosting or caching issue. "
                                           "🟢 Good: <0.8 s  🟡 Needs work: 0.8–1.8 s  🔴 Poor: >1.8 s"),
    "speed_index": ("Speed Index",        "How quickly the page content is visually filled in during load. "
                                           "Unlike LCP, this captures the overall visual progress, not just one element. "
                                           "🟢 Good: <3.4 s  🟡 Needs work: 3.4–5.8 s  🔴 Poor: >5.8 s"),
}

PERF_SCORE_HELP = ("Overall Performance Score",
                   "Google Lighthouse score (0–100) combining all speed metrics with weighted importance. "
                   "🟢 90–100: Fast  🟡 50–89: Needs improvement  🔴 0–49: Slow")


def render_pagespeed(raw: dict):
    """Render a compact PageSpeed metrics card for mobile + desktop."""
    if not raw:
        return

    errors = raw.get("_errors", {})
    strategies = [s for s in ("mobile", "desktop") if s in raw]

    if not strategies:
        if errors:
            msgs = ", ".join(f"{k}: {v}" for k, v in errors.items())
            st.warning(
                f"PageSpeed data unavailable ({msgs}). "
                "The free API allows ~400 req/day per IP. "
                "Add `PAGESPEED_API_KEY=your-key` to `.env` for higher limits."
            )
        else:
            st.warning("PageSpeed data unavailable.")
        return

    def score_colour(score):
        if score is None: return "⚪"
        if score >= 90:   return "🟢"
        if score >= 50:   return "🟡"
        return "🔴"

    def cwv_colour(key: str, val):
        if val is None: return "⚪"
        t = CWV_THRESHOLDS.get(key, {})
        if not t: return "⚪"
        if val <= t["good"]: return "🟢"
        if val <= t["poor"]: return "🟡"
        return "🔴"

    cols = st.columns(len(strategies))
    for col, strategy in zip(cols, strategies):
        m = raw[strategy]
        score = m.get("score")
        with col:
            run_count = m.get("_run_count", PSI_RUNS)
            st.metric(
                label=f"{strategy.capitalize()} — Performance Score (median of {run_count} runs)",
                value=f"{score}/100" if score is not None else "N/A",
                delta=score_colour(score),
                delta_color="off",
                help=f"**{PERF_SCORE_HELP[0]}** — {PERF_SCORE_HELP[1]}",
            )
            for key, num_key, display_key in [
                ("lcp",          "lcp_ms",  "lcp"),
                ("fcp",          "fcp_ms",  "fcp"),
                ("cls",          "cls_raw", "cls"),
                ("tbt",          "tbt_ms",  "tbt"),
                ("inp",          "inp_ms",  "inp"),
                ("ttfb",         "ttfb_ms", "ttfb"),
                ("speed_index",  "speed_ms","speed_index"),
            ]:
                label_name, help_text = METRIC_HELP[key]
                icon = cwv_colour(key, m.get(num_key))
                st.metric(
                    label=f"{icon} {label_name}",
                    value=m.get(display_key, "N/A"),
                    help=f"**{label_name}** — {help_text}",
                )

    if errors:
        st.caption(f"Note: {', '.join(errors.keys())} data unavailable ({', '.join(errors.values())})")


def _audit_filename(url: str, suffix: str) -> str:
    slug = re.sub(r"https?://", "", url).rstrip("/").replace("/", "-").replace(".", "-")
    return f"cro-audit-{suffix}-{slug}.md"


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI CRO Audit Tool", page_icon="📈")
st.title("📈 AI CRO Audit Tool")

site_type = st.radio(
    "Site type",
    options=["SaaS", "E-commerce"],
    horizontal=True,
    help=(
        "**SaaS** — optimises for trial/demo conversion, pricing plan clarity, signup friction, "
        "and software-buyer trust signals.\n\n"
        "**E-commerce** — optimises for Add-to-Cart / Buy Now conversion, product clarity, "
        "price anchoring, shopper trust (reviews, returns, shipping), and cart friction."
    ),
)

tab1, tab2 = st.tabs(["URL Audit", "Vision (Auto)"])


with tab1:
    url = st.text_input("Website URL", key="url_audit")

    if st.button("Run Audit", key="run_audit", type="primary"):
        if not url or not is_valid_url(url):
            st.error("Please enter a valid URL starting with https://")
            st.stop()

        with st.spinner("Extracting pages + fetching PageSpeed data..."):
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_text = executor.submit(extract_text_from_url, url)
                f_psi  = executor.submit(fetch_pagespeed, url)
            content, scraped_pages = f_text.result()
            ps_context, ps_raw = f_psi.result()

        if scraped_pages:
            st.info(f"Scraped {len(scraped_pages)} page(s): {', '.join(scraped_pages)}")
        else:
            st.warning("No pages could be scraped. Check the URL and try again.")
            st.stop()

        with st.spinner("Analyzing with AI..."):
            result = run_ai_text(content, pagespeed=ps_context, site_type=site_type)

        st.session_state["text_result"] = result
        st.session_state["text_url"] = url
        st.session_state["text_scraped_pages"] = scraped_pages
        st.session_state["text_ps_raw"] = ps_raw
        st.session_state["text_site_type"] = site_type

    if "text_result" in st.session_state:
        if "text_site_type" in st.session_state:
            st.caption(f"Audit type: {st.session_state['text_site_type']}")
        if "text_scraped_pages" in st.session_state:
            pages = st.session_state["text_scraped_pages"]
            st.info(f"Scraped {len(pages)} page(s): {', '.join(pages)}")

        if "text_ps_raw" in st.session_state:
            with st.expander("PageSpeed Insights", expanded=True):
                render_pagespeed(st.session_state["text_ps_raw"])

        st.subheader("Results")
        st.write(st.session_state["text_result"])

        st.download_button(
            label="Download audit (.md)",
            data=st.session_state["text_result"],
            file_name=_audit_filename(st.session_state["text_url"], "text"),
            mime="text/markdown",
        )


with tab2:
    vurl = st.text_input("Website URL for screenshots", key="vision_url")
    include_text = st.checkbox("Include text context", value=True, key="vision_include_text")

    if st.button("Run Vision Audit", key="run_vision", type="primary"):
        if not vurl or not is_valid_url(vurl):
            st.error("Please enter a valid URL starting with https://")
            st.stop()

        with st.spinner("Capturing screenshots (desktop + mobile)..."):
            try:
                shots, debug, raw = take_auto_screenshots(vurl)
            except Exception as e:
                st.error(f"Screenshot failed: {e}")
                st.stop()

        with st.expander("Debug (capture.py)"):
            st.json(debug)
            if isinstance(raw, dict) and (raw.get("discovered_urls") or raw.get("errors")):
                st.write("**Discovered URLs:**")
                st.write(raw.get("discovered_urls", []))
                if raw.get("errors"):
                    st.write("**Capture errors:**")
                    st.write(raw.get("errors", []))

        if not shots:
            st.error("No screenshots captured. The site may block headless browsing.")
            st.stop()

        text_context = ""
        scraped_pages = []
        ps_context = ""
        ps_raw: dict = {}

        with st.spinner("Extracting text + fetching PageSpeed data..."):
            tasks: list = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_psi = executor.submit(fetch_pagespeed, vurl)
                if include_text:
                    f_text = executor.submit(extract_text_from_url, vurl)
                    tasks.append(f_text)
                tasks.append(f_psi)
            ps_context, ps_raw = f_psi.result()
            if include_text:
                text_context, scraped_pages = f_text.result()
                if scraped_pages:
                    st.info(f"Text scraped from {len(scraped_pages)} page(s): {', '.join(scraped_pages)}")

        with st.spinner("Running vision audit (gpt-4.1)..."):
            result = run_ai_vision(text_context, shots, pagespeed=ps_context, site_type=site_type)

        st.session_state["vision_result"] = result
        st.session_state["vision_result_url"] = vurl
        st.session_state["vision_shots"] = shots
        st.session_state["vision_scraped_pages"] = scraped_pages
        st.session_state["vision_ps_raw"] = ps_raw
        st.session_state["vision_site_type"] = site_type

    if "vision_result" in st.session_state:
        if "vision_site_type" in st.session_state:
            st.caption(f"Audit type: {st.session_state['vision_site_type']}")
        if "vision_scraped_pages" in st.session_state and st.session_state["vision_scraped_pages"]:
            pages = st.session_state["vision_scraped_pages"]
            st.info(f"Text scraped from {len(pages)} page(s): {', '.join(pages)}")

        if "vision_ps_raw" in st.session_state:
            with st.expander("PageSpeed Insights", expanded=True):
                render_pagespeed(st.session_state["vision_ps_raw"])

        if "vision_shots" in st.session_state:
            render_shots_gallery(st.session_state["vision_shots"])

        st.subheader("Results")
        st.write(st.session_state["vision_result"])

        st.download_button(
            label="Download audit (.md)",
            data=st.session_state["vision_result"],
            file_name=_audit_filename(st.session_state["vision_result_url"], "vision"),
            mime="text/markdown",
        )

        with st.expander("Visited pages (final URLs)"):
            shots_display = st.session_state.get("vision_shots", [])
            st.write([s.get("final_url") or s.get("url") for s in shots_display])
