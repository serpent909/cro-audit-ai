import json
import re
import subprocess
import sys
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
MODEL_TEXT = "gpt-4.1-mini"
MODEL_VISION = "gpt-4.1-mini"

MAX_CHARS = 12000
MAX_IMAGES = 3

PROMPT = """
You are a senior CRO consultant. Your job is to produce an audit that is specific, evidence-based, and action-oriented.

NON-NEGOTIABLE RULES
- Only make claims supported by provided text or screenshots.
- If you cannot confirm something, write: "Not observed in provided content."
- Every issue MUST include Evidence (quote text OR describe what is visible and where: 'hero area', 'header nav', 'pricing table', etc.)
- If CTA candidates exist, list them and evaluate them (do not claim no CTAs).

OUTPUT FORMAT (use headings exactly)

## 1) Funnel map
List the pages captured (homepage/pricing/etc.) and what each appears to be for.

## 2) Executive summary
- Biggest conversion blocker
- Biggest trust/credibility gap
- Biggest message/positioning gap
- Biggest friction point
- Highest-impact quick win

## 3) Conversion scorecard (0–10)
Score each with one sentence rationale:
- Value prop clarity
- CTA clarity
- Trust & social proof
- Pricing clarity
- Friction (forms/steps)
- Visual hierarchy

## 4) Top 7 issues (ranked)
For each:
- Issue (one line)
- Impact (High/Med/Low)
- Effort (S/M/L)
- Confidence (High/Med/Low)
- Evidence
- Recommendation (specific change)
- Test idea (hypothesis + metric)

## 5) Copy & CTA improvements (write the copy)
- 3 improved headline/value prop options
- 3 improved primary CTA label options (tailored to the page intent)
- 3 trust microcopy examples (near CTA/form)

## 6) Experiment plan (2 weeks)
Provide a simple sequencing:
Week 1: 2 quick wins
Week 2: 2 bigger tests
Metrics + guardrails
"""


# ----------------------------
# Helpers
# ----------------------------
def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def extract_from_single_page(page_url: str, headers: dict):
    try:
        r = requests.get(page_url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None

        html = r.text

        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "lxml")
        main_text = soup.get_text("\n", strip=True)

        full = BeautifulSoup(html, "lxml")

        ctas = []

        for b in full.find_all("button"):
            txt = b.get_text(" ", strip=True)
            if txt:
                ctas.append(f"BUTTON: {txt}")

        for a in full.find_all("a"):
            txt = a.get_text(" ", strip=True)
            href = (a.get("href") or "").strip()
            if txt and any(k in txt.lower() for k in [
                "book", "demo", "start", "get", "try", "sign", "contact",
                "pricing", "plans", "buy", "join", "trial"
            ]):
                ctas.append(f"LINK: {txt} -> {href}")

        ctas = list(dict.fromkeys(ctas))[:40]

        return f"""
PAGE: {page_url}

TEXT:
{main_text}

CTAs:
{chr(10).join(ctas) if ctas else "None detected"}
"""

    except Exception:
        return None


def extract_text_from_url(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    base = url.rstrip("/")

    paths = ["", "/pricing", "/plans", "/compare"]
    bundles = []

    for p in paths:
        page = base + p
        content = extract_from_single_page(page, headers)
        if content:
            bundles.append(content)

    if not bundles:
        return "No content extracted."

    combined = "\n\n---\n\n".join(bundles)
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()
    return combined


def run_ai_text(text: str):
    client = OpenAI()
    response = client.responses.create(
        model=MODEL_TEXT,
        input=f"{PROMPT}\n\n{text[:MAX_CHARS]}"
    )
    return response.output_text


def run_ai_vision(text_context, image_data_urls, pages):
    client = OpenAI()

    content = [{
        "type": "input_text",
        "text": f"{PROMPT}\n\nCaptured pages:\n" + "\n".join(f"- {p}" for p in pages)
    }]

    if text_context:
        content.append({
            "type": "input_text",
            "text": f"\n\nText context:\n{text_context[:MAX_CHARS]}"
        })

    for img in image_data_urls:
        content.append({
            "type": "input_image",
            "image_url": img
        })

    response = client.responses.create(
        model=MODEL_VISION,
        input=[{"role": "user", "content": content}]
    )

    return response.output_text


def take_auto_screenshots(url: str):
    """
    Runs capture.py using the SAME python executable as this Streamlit app (.venv),
    so capture.py can import playwright.
    """
    result = subprocess.run(
        [sys.executable, "capture.py", url, str(MAX_IMAGES)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "capture.py failed with no stderr output")

    data = json.loads(result.stdout)
    return data.get("images", []), data.get("pages", [])


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI CRO Audit Tool", page_icon="📈")
st.title("📈 AI CRO Audit Tool")

tab1, tab2, tab3 = st.tabs(["URL Audit", "Paste Text", "Vision (Auto)"])


# --- TEXT AUDIT ---
with tab1:
    url = st.text_input("Website URL", key="url_audit")

    if st.button("Run Audit", key="run_audit", type="primary"):
        if not url or not is_valid_url(url):
            st.error("Please enter a valid URL starting with https://")
            st.stop()

        with st.spinner("Extracting..."):
            content = extract_text_from_url(url)

        with st.spinner("Analyzing..."):
            result = run_ai_text(content)

        st.subheader("Results")
        st.write(result)


# --- PASTE ---
with tab2:
    text = st.text_area("Paste content", key="paste_text", height=240)

    if st.button("Run Text Audit", key="run_text_audit", type="primary"):
        if not text.strip():
            st.error("Paste something first.")
            st.stop()

        result = run_ai_text(text)
        st.subheader("Results")
        st.write(result)


# --- VISION ---
with tab3:
    vurl = st.text_input("Website URL for screenshots", key="vision_url")
    include_text = st.checkbox("Include text context", value=True, key="vision_include_text")

    if st.button("Run Vision Audit", key="run_vision", type="primary"):
        if not vurl or not is_valid_url(vurl):
            st.error("Please enter a valid URL starting with https://")
            st.stop()

        with st.spinner("Capturing screenshots (auto)..."):
            try:
                images, pages = take_auto_screenshots(vurl)
            except Exception as e:
                st.error(f"Screenshot failed: {e}")
                st.stop()

        if not images:
            st.error("No screenshots captured. The site may block headless browsing.")
            st.stop()

        text_context = ""
        if include_text:
            with st.spinner("Extracting text context..."):
                text_context = extract_text_from_url(vurl)

        with st.spinner("Running vision audit..."):
            result = run_ai_vision(text_context, images, pages)

        st.subheader("Results")
        st.write(result)

        with st.expander("Captured pages"):
            st.write(pages)