# app.py
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI

# ✅ NEW: load .env early (before OpenAI() is constructed)
from dotenv import load_dotenv


# ----------------------------
# Load environment (.env)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# load_dotenv will:
# - load from ENV_PATH if it exists
# - not override already-set env vars by default
load_dotenv(dotenv_path=ENV_PATH)

# Optional: also allow OS env vars to win if already set
# (default behavior is fine)


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
# OpenAI client helper
# ----------------------------
def get_openai_client() -> OpenAI:
    """
    Creates an OpenAI client. Requires OPENAI_API_KEY in environment.
    We keep it in a function so we can fail with a friendly message.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Show a useful Streamlit error and stop the run
        st.error(
            "OPENAI_API_KEY not found.\n\n"
            "Make sure you have a `.env` file in the same folder as app.py with:\n"
            "OPENAI_API_KEY=your_key_here\n\n"
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
            if txt and any(
                k in txt.lower()
                for k in [
                    "book", "demo", "start", "get", "try", "sign", "contact",
                    "pricing", "plans", "buy", "join", "trial",
                ]
            ):
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
    client = get_openai_client()
    response = client.responses.create(
        model=MODEL_TEXT,
        input=f"{PROMPT}\n\n{text[:MAX_CHARS]}",
    )
    return response.output_text


def run_ai_vision(text_context, shots):
    client = get_openai_client()

    visited_lines = []
    for s in shots:
        visited_lines.append(
            f"- {s.get('final_url') or s.get('url')}"
            + (f" | title: {s.get('title')}" if s.get("title") else "")
        )

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
        content.append({"type": "input_text", "text": f"\n\nText context:\n{text_context[:MAX_CHARS]}"})

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
        timeout=140,
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
        shots = [{"image": img, "url": p, "final_url": p, "title": "", "notes": ""} for img, p in zip(images, pages)]

    return shots, debug, data


def render_shots_gallery(shots):
    st.subheader("Captured screenshots")
    for i, s in enumerate(shots, start=1):
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(s.get("image"), width='stretch')
        with cols[1]:
            st.markdown(f"**{i}. URL:** `{s.get('url')}`")
            if s.get("final_url") and s.get("final_url") != s.get("url"):
                st.markdown(f"**Final:** `{s.get('final_url')}`")
            if s.get("title"):
                st.markdown(f"**Title:** {s.get('title')}")
            if s.get("notes"):
                st.markdown(f"**Notes:** `{s.get('notes')}`")
        st.divider()


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

        with st.spinner("Extracting..."):
            content = extract_text_from_url(url)

        with st.spinner("Analyzing..."):
            result = run_ai_text(content)

        st.subheader("Results")
        st.write(result)

with tab2:
    vurl = st.text_input("Website URL for screenshots", key="vision_url")
    include_text = st.checkbox("Include text context", value=True, key="vision_include_text")

    if st.button("Run Vision Audit", key="run_vision", type="primary"):
        if not vurl or not is_valid_url(vurl):
            st.error("Please enter a valid URL starting with https://")
            st.stop()

        with st.spinner("Capturing screenshots (auto)..."):
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

        render_shots_gallery(shots)

        text_context = ""
        if include_text:
            with st.spinner("Extracting text context..."):
                text_context = extract_text_from_url(vurl)

        with st.spinner("Running vision audit..."):
            result = run_ai_vision(text_context, shots)

        st.subheader("Results")
        st.write(result)

        with st.expander("Visited pages (final URLs)"):
            st.write([s.get("final_url") or s.get("url") for s in shots])