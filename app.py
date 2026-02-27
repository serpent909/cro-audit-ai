# app.py
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

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

MAX_CHARS = 12000
MAX_IMAGES = 3

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

PROMPT = """
You are a senior CRO (Conversion Rate Optimization) consultant. Produce an audit that is specific, evidence-based, and action-oriented.

NON-NEGOTIABLE RULES
- Only make claims supported by the provided text or screenshots.
- If you cannot confirm something, write: "Not observed in provided content."
- Every issue MUST include Evidence (quote text OR describe what is visible and where: 'hero area', 'header nav', 'pricing table', etc.).
- If CTA candidates exist, list them and evaluate them (do not claim no CTAs).
- Always evaluate what is visible ABOVE THE FOLD (without scrolling) separately from below-the-fold content.
- Compare CTAs and messaging across pages — flag any inconsistencies between the homepage promise and pricing/signup pages.

OUTPUT FORMAT (use headings exactly)

## 0) Business context
In 1–2 sentences: identify the apparent industry, business model (B2B SaaS / e-commerce / lead gen / marketplace / etc.), and primary target audience. Use this context to inform all findings below.

## 1) Funnel map
List each page captured, its apparent purpose, and the primary CTA on that page.

## 2) Executive summary
- Biggest conversion blocker
- Biggest trust/credibility gap
- Biggest message/positioning gap
- Biggest friction point
- Highest-impact quick win
- CTA consistency verdict: are CTAs and messaging consistent across pages, or contradictory?

## 3) Conversion scorecard (weighted)
Score each dimension 0–10 with one sentence of rationale. Then compute a weighted overall score.

| Dimension | Score /10 | Weight | Weighted |
|---|---|---|---|
| Value prop clarity | ? | 2× | |
| CTA clarity | ? | 2× | |
| Trust & social proof | ? | 1.5× | |
| Pricing clarity | ? | 1.5× | |
| Friction (forms/steps) | ? | 1× | |
| Visual hierarchy | ? | 1× | |
| Mobile experience | ? | 1× | |

Overall score = sum(weighted) / 10 → show as **X.X / 10**

## 4) Above-the-fold analysis
For EACH page captured: what does a visitor see before scrolling? Is the value prop clear? Is there a visible CTA? What is missing or unclear from the first impression?

## 5) Top 7 issues (ranked by Impact × Confidence)
For each issue:
- **Issue** (one line)
- Impact: High/Med/Low
- Effort: S/M/L
- Confidence: High/Med/Low
- Evidence (quote or describe exact location on page)
- Recommendation (specific, actionable — not vague advice)
- Test idea: hypothesis → primary metric → guardrail metric

## 6) Mobile considerations
Identify 3–5 mobile-specific CRO risks: tap target sizes, truncated headlines, sticky CTAs, form usability on small screens, etc. If a mobile screenshot is provided, base findings on visual evidence. Otherwise, infer from page structure and content.

## 7) Copy & CTA improvements (write the actual copy)
- 3 improved headline/value prop options (include a one-line rationale for each)
- 3 improved primary CTA label options (tailored to the specific page intent)
- 3 trust microcopy examples (to place near CTA or form submit button)

## 8) Experiment plan (2 weeks)
Week 1: 2 quick wins (low effort, high confidence)
Week 2: 2 bigger tests (higher effort, higher potential impact)
For each test: hypothesis | primary metric | guardrail metric | minimum detectable effect
"""


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


def extract_from_single_page(page_url: str, headers: dict) -> str | None:
    try:
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

        # Social proof
        proof_keywords = [
            "testimonial", "review", "rating", "stars", "trustpilot",
            "g2", "capterra", "case-study", "customers", "clients",
        ]
        proof_elements = []
        for kw in proof_keywords:
            for el in full.find_all(class_=re.compile(kw, re.I))[:2]:
                txt = el.get_text(" ", strip=True)[:200]
                if txt:
                    proof_elements.append(f"[{kw}]: {txt}")

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
{chr(10).join(proof_elements[:10]) if proof_elements else "None detected"}

TRUST SIGNALS:
{chr(10).join(trust_signals[:10]) if trust_signals else "None detected"}
"""

    except Exception:
        return None


def extract_text_from_url(url: str) -> tuple[str, list[str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    base = url.rstrip("/")

    def fetch(path: str) -> tuple[str, str | None]:
        page_url = base + path
        return page_url, extract_from_single_page(page_url, headers)

    # Fetch all paths in parallel
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch, p): p for p in SCRAPE_PATHS}
        for future in as_completed(futures):
            page_url, content = future.result()
            if content:
                results[page_url] = content

    # Reassemble in original path order
    bundles = []
    scraped_pages = []
    for p in SCRAPE_PATHS:
        page_url = base + p
        if page_url in results:
            bundles.append(results[page_url])
            scraped_pages.append(page_url)

    if not bundles:
        return "No content extracted.", []

    combined = "\n\n---\n\n".join(bundles)
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()
    return combined, scraped_pages


def run_ai_text(text: str) -> str:
    client = get_openai_client()
    response = client.responses.create(
        model=MODEL_TEXT,
        input=f"{PROMPT}\n\n{text[:MAX_CHARS]}",
    )
    return response.output_text


def run_ai_vision(text_context: str, shots: list) -> str:
    client = get_openai_client()

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
                f"{PROMPT}\n\n"
                f"Visited pages (use these URLs when referencing evidence):\n"
                + "\n".join(visited_lines)
            ),
        }
    ]

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


def _audit_filename(url: str, suffix: str) -> str:
    slug = re.sub(r"https?://", "", url).rstrip("/").replace("/", "-").replace(".", "-")
    return f"cro-audit-{suffix}-{slug}.md"


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI CRO Audit Tool", page_icon="📈")
st.title("📈 AI CRO Audit Tool")

tab1, tab2 = st.tabs(["URL Audit", "Vision (Auto)"])


with tab1:
    url = st.text_input("Website URL", key="url_audit")

    if st.button("Run Audit", key="run_audit", type="primary"):
        if not url or not is_valid_url(url):
            st.error("Please enter a valid URL starting with https://")
            st.stop()

        with st.spinner("Extracting pages in parallel..."):
            content, scraped_pages = extract_text_from_url(url)

        if scraped_pages:
            st.info(f"Scraped {len(scraped_pages)} page(s): {', '.join(scraped_pages)}")
        else:
            st.warning("No pages could be scraped. Check the URL and try again.")
            st.stop()

        with st.spinner("Analyzing with AI..."):
            result = run_ai_text(content)

        st.session_state["text_result"] = result
        st.session_state["text_url"] = url
        st.session_state["text_scraped_pages"] = scraped_pages

    if "text_result" in st.session_state:
        if "text_scraped_pages" in st.session_state:
            pages = st.session_state["text_scraped_pages"]
            st.info(f"Scraped {len(pages)} page(s): {', '.join(pages)}")

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
        if include_text:
            with st.spinner("Extracting text from pages in parallel..."):
                text_context, scraped_pages = extract_text_from_url(vurl)
                if scraped_pages:
                    st.info(f"Text scraped from {len(scraped_pages)} page(s): {', '.join(scraped_pages)}")

        with st.spinner("Running vision audit (gpt-4.1)..."):
            result = run_ai_vision(text_context, shots)

        st.session_state["vision_result"] = result
        st.session_state["vision_result_url"] = vurl
        st.session_state["vision_shots"] = shots
        st.session_state["vision_scraped_pages"] = scraped_pages

    if "vision_result" in st.session_state:
        if "vision_scraped_pages" in st.session_state and st.session_state["vision_scraped_pages"]:
            pages = st.session_state["vision_scraped_pages"]
            st.info(f"Text scraped from {len(pages)} page(s): {', '.join(pages)}")

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
