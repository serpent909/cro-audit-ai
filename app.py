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
MODEL_TEXT = "gpt-5.2"
MODEL_VISION = "gpt-5.2"

MAX_CHARS = 20000
MAX_IMAGES = 3

PAGESPEED_API = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
PSI_RUNS = 3  # parallel runs per strategy; median taken to reduce variability

# Injected once at app start — polishes the AI-generated markdown report
_REPORT_CSS = """<style>
/* ── section headings ─────────────────────────────────── */
[data-testid="stMarkdownContainer"] h2 {
    margin-top: 2rem !important;
    margin-bottom: 0.55rem !important;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(49, 51, 63, 0.12);
}
[data-testid="stMarkdownContainer"] h3 {
    margin-top: 1.4rem !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stMarkdownContainer"] h4 {
    margin-top: 1rem !important;
    margin-bottom: 0.2rem !important;
}
/* ── body copy ────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p {
    line-height: 1.75 !important;
    margin-bottom: 0.7rem !important;
}
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol {
    line-height: 1.75 !important;
    margin-bottom: 0.75rem !important;
    padding-left: 1.4rem !important;
}
[data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.25rem !important;
}
[data-testid="stMarkdownContainer"] li > ul,
[data-testid="stMarkdownContainer"] li > ol {
    margin-bottom: 0.2rem !important;
}
/* ── scorecard table ──────────────────────────────────── */
[data-testid="stMarkdownContainer"] table {
    margin-top: 0.6rem !important;
    margin-bottom: 1.25rem !important;
    width: 100%;
    border-collapse: collapse;
}
[data-testid="stMarkdownContainer"] th {
    background-color: rgba(49, 51, 63, 0.06) !important;
    text-align: left;
    padding: 0.45rem 0.75rem !important;
}
[data-testid="stMarkdownContainer"] td {
    padding: 0.4rem 0.75rem !important;
    vertical-align: top;
}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td {
    background-color: rgba(49, 51, 63, 0.02) !important;
}
/* ── horizontal rules ─────────────────────────────────── */
[data-testid="stMarkdownContainer"] hr {
    margin: 1.75rem 0 !important;
    border-color: rgba(49, 51, 63, 0.15) !important;
}
/* ── strong / bold ────────────────────────────────────── */
[data-testid="stMarkdownContainer"] strong {
    font-weight: 600 !important;
}
/* ── inline code ──────────────────────────────────────── */
[data-testid="stMarkdownContainer"] code {
    padding: 0.15em 0.35em !important;
    border-radius: 4px !important;
}
/* ── button label reset (prevent p styles leaking in) ─── */
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-secondary"] p,
button p {
    line-height: normal !important;
    margin-bottom: 0 !important;
}
/* ── custom tooltip for PageSpeed metric rows ────────── */
[data-testid="stMarkdownContainer"] {
    overflow: visible !important;
}
.cro-tip {
    position: relative;
    display: inline-flex;
    align-items: center;
}
.cro-tip-box {
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.65;
    min-width: 220px;
    max-width: 300px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    z-index: 9999;
    white-space: normal;
    pointer-events: none;
}
.cro-tip:hover .cro-tip-box {
    display: block;
}
</style>"""

# ----------------------------
# PDF report styles
# ----------------------------
_PDF_CSS = """
/* ── Reset ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Page geometry ──────────────────────────────────────── */
/* Full-bleed for all pages; @page margin-box footer is injected dynamically */
@page { size: A4; margin: 0; }

:root {
    --navy:    #0f172a;
    --navy2:   #1e3a5f;
    --accent:  #6366f1;
    --accent2: #8b5cf6;
    --green:   #059669;
    --amber:   #d97706;
    --red:     #dc2626;
    --text:    #1e293b;
    --muted:   #64748b;
    --border:  #e2e8f0;
    --bg:      #f8fafc;
    --white:   #ffffff;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, Helvetica, sans-serif;
    color: var(--text);
    font-size: 10.5pt;
    line-height: 1.65;
    margin: 0;
    padding: 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

/* ── COVER PAGE ─────────────────────────────────────────── */
.cover {
    width: 210mm;
    min-height: 297mm;
    background: var(--navy);
    color: white;
    display: flex;
    flex-direction: column;
    page-break-after: always;
    position: relative;
    z-index: 1;
    overflow: hidden;
}

/* subtle radial glows for depth */
.cover::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 85% 15%, rgba(99,102,241,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 10% 85%, rgba(139,92,246,0.2) 0%, transparent 60%);
    pointer-events: none;
}

.cover-accent-bar {
    height: 5px;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    flex-shrink: 0;
    position: relative;
    z-index: 1;
}

.cover-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 52px 64px 44px;
    position: relative;
    z-index: 1;
}

.cover-eyebrow {
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0;
}

.cover-spacer { flex: 1; min-height: 28px; }

.cover-title {
    font-size: 48pt;
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: white;
    margin: 20px 0 8px;
}

.cover-subtitle {
    font-size: 16pt;
    font-weight: 300;
    color: #94a3b8;
    letter-spacing: 0.03em;
    margin-bottom: 44px;
}

.cover-url-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 12px;
    padding: 18px 24px;
    display: inline-flex;
    flex-direction: column;
    gap: 6px;
    max-width: 440px;
    margin-bottom: 20px;
}

.cover-url-label {
    font-size: 7pt;
    color: #475569;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
}

.cover-url-value {
    font-size: 13pt;
    font-weight: 600;
    color: #e2e8f0;
    word-break: break-all;
    line-height: 1.3;
}

.cover-meta-row {
    display: flex;
    gap: 32px;
}

.cover-meta-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.cover-meta-label {
    font-size: 7pt;
    color: #475569;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
}

.cover-meta-value {
    font-size: 11pt;
    font-weight: 600;
    color: #cbd5e1;
}

.cover-score-badge {
    width: 116px;
    height: 116px;
    border-radius: 50%;
    border: 3px solid;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(0,0,0,0.25);
    flex-shrink: 0;
}

.cover-score-number {
    font-size: 27pt;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}

.cover-score-denom {
    font-size: 9pt;
    font-weight: 400;
    color: #94a3b8;
    line-height: 1;
}

.cover-score-label {
    font-size: 6pt;
    color: #94a3b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

.cover-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 64px 22px;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #475569;
    font-size: 8pt;
    position: relative;
    z-index: 1;
}

.cover-footer-brand {
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.05em;
    font-size: 9pt;
}

/* ── MAIN CONTENT ───────────────────────────────────────── */
/* @page handles top/side/bottom margins; no padding needed here */
.main-content {
    padding: 0;
}

/* ── TABLE OF CONTENTS PAGE ─────────────────────────────── */
.toc-page {
    page-break-after: always;
    padding: 4mm 0 18mm;
    /* No min-height — page-break-after:always handles it; min-height was causing overflow */
    display: flex;
    flex-direction: column;
    position: relative;
    z-index: 1;
    background: white;
}

.toc-eyebrow {
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

.toc-heading {
    font-size: 30pt;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.03em;
    margin: 0 0 36px;
    padding: 0;
    border: none;              /* override global h2 border */
    page-break-before: avoid; /* override section-start */
}

.toc-accent {
    width: 48px;
    height: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    margin-bottom: 32px;
}

.toc-entries { flex: 1; }

.toc-entry {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 13px 0;
    border-bottom: 1px solid var(--border);
    break-inside: avoid;
}

.toc-entry:last-child { border-bottom: none; }

.toc-num {
    font-size: 8.5pt;
    font-weight: 800;
    color: var(--accent);
    width: 24px;
    flex-shrink: 0;
    text-align: right;
}

.toc-name {
    font-size: 11pt;
    font-weight: 500;
    color: var(--text);
    flex: 1;
    line-height: 1.3;
}

.toc-cat {
    font-size: 7.5pt;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    flex-shrink: 0;
}

/* ── TYPOGRAPHY ─────────────────────────────────────────── */
/* h2 = primary accent colour (indigo) — highest visual level */
/* h3 = deep navy — secondary level                          */
/* h4 = slate — tertiary level                               */

.main-content h2 {
    font-size: 15pt;
    font-weight: 800;
    color: var(--accent);          /* indigo — primary section colour */
    margin: 0 0 8px;
    padding-bottom: 9px;
    border-bottom: 3px solid var(--accent);
    letter-spacing: -0.02em;
    page-break-before: always;
    break-before: always;
    page-break-after: avoid;
    break-after: avoid;
}

/* ── EXECUTIVE SUMMARY CARD ─────────────────────────────── */
.exec-summary-card {
    background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
    border: 1px solid #c7d2fe;
    border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 18px 22px 14px;
    margin: 8px 0 0;
}
.exec-summary-card ul  { margin-bottom: 4px; }
.exec-summary-card li  { margin-bottom: 7px; line-height: 1.65; }
.exec-summary-card p   { margin-bottom: 7px; }

/* First h2 in the page — no forced break (already new page after TOC) */
.main-content h2:first-of-type {
    page-break-before: auto;
    break-before: auto;
}

h3 {
    font-size: 12pt;
    font-weight: 700;
    color: var(--navy);            /* navy — secondary heading */
    margin: 22px 0 6px;
    page-break-after: avoid;
    break-after: avoid;
}

/* ── Orphan prevention: keep headings with their list content ── */
/* break-before:avoid  → list must start on the same page as its h3    */
/* break-inside:avoid  → list must not split internally between li rows */
/* Together: the browser's only valid break point is BEFORE the h3,    */
/* so the whole heading + list is pushed to the next page as a unit.   */
h3 + ul, h3 + ol {
    page-break-before: avoid;
    break-before: avoid;
    page-break-inside: avoid;
    break-inside: avoid;
}
h3 + p, h3 + h4 {
    page-break-before: avoid;
    break-before: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #334155;                /* slate — tertiary heading */
    margin: 16px 0 5px;
    page-break-after: avoid;
    break-after: avoid;
}

h4 + ul, h4 + ol, h4 + p,
h4 + * + ul, h4 + * + ol, h4 + * + p {
    page-break-before: avoid;
    break-before: avoid;
}

p    { margin: 0 0 9px; orphans: 3; widows: 3; }

/* Proper bullet hierarchy */
ul            { list-style-type: disc;    margin: 0 0 11px 1.6rem; padding: 0; }
ul ul         { list-style-type: circle;  margin: 4px 0 6px 1.6rem; }
ul ul ul      { list-style-type: square; }
ol            { list-style-type: decimal; margin: 0 0 11px 1.6rem; padding: 0; }
ol li         { margin-bottom: 7px; }
ol li > ul    { margin-top: 5px; }
ol li > p     { margin: 0 0 4px; }
li            { margin-bottom: 5px; break-inside: avoid; }

strong { font-weight: 700; color: var(--navy); }
em   { font-style: italic; }
hr   { border: none; border-top: 1px solid var(--border); margin: 22px 0; }

/* ── COMMENTARY BOXES ────────────────────────────────────── */
.commentary-block {
    background: #f0f4ff;
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 16px 0 20px;
    break-inside: avoid;
}

.commentary-block p { margin: 0 0 6px; font-size: 9.5pt; color: #374151; }
.commentary-block p:last-child { margin: 0; }
.commentary-block strong { color: var(--accent); }

/* ── TABLES ─────────────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 20px;
    font-size: 9.5pt;
    break-inside: avoid;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--border), 0 2px 6px rgba(0,0,0,0.05);
}

thead tr { background: var(--navy); }

thead th {
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 8.5pt;
    color: white;
    letter-spacing: 0.04em;
}

tbody tr:nth-child(even) { background: var(--bg); }
tbody tr:nth-child(odd)  { background: white; }

tbody td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

tbody tr:last-child td { border-bottom: none; }

/* ── IMPACT / EFFORT / CONFIDENCE BADGES ────────────────── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    white-space: nowrap;
    vertical-align: middle;
}

.badge-impact-high { background: #fee2e2; color: #dc2626; }
.badge-impact-med  { background: #fef3c7; color: #d97706; }
.badge-impact-low  { background: #dcfce7; color: #059669; }
.badge-effort-s    { background: #dcfce7; color: #059669; }
.badge-effort-m    { background: #fef3c7; color: #d97706; }
.badge-effort-l    { background: #fee2e2; color: #dc2626; }
.badge-conf-high   { background: #ede9fe; color: #7c3aed; }
.badge-conf-med    { background: #fef3c7; color: #d97706; }
.badge-conf-low    { background: #f1f5f9; color: #64748b; }

/* ── PAGESPEED SECTION ──────────────────────────────────── */
.psi-section-label {
    font-size: 7.5pt;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: -4px 0 14px;
}

.psi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin: 0 0 26px;
}

.psi-card {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    break-inside: avoid;
}

.psi-card-header {
    padding: 18px 22px 14px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

.psi-strategy { font-size: 8pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.psi-score-row { display: flex; align-items: baseline; gap: 3px; }
.psi-score-big { font-size: 34pt; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
.psi-score-denom { font-size: 11pt; font-weight: 400; color: inherit; opacity: 0.6; }

.psi-status-pill {
    font-size: 7.5pt;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: white;
    align-self: flex-start;
}

.psi-metrics { padding: 4px 22px 16px; background: white; }

.psi-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 8.5pt;
}

.psi-metric-row:last-child { border-bottom: none; }
.psi-metric-name { color: var(--muted); }
.psi-metric-val  { font-weight: 700; }
.psi-metric-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

"""

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
MAX_DISCOVERED = 8   # max candidates returned per discovery pass
MAX_L2_SOURCES = 2   # top L1 pages to also crawl for L2 links
MAX_PAGES = 12       # total cap on discovered pages sent to the AI (excl. homepage)

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
def normalise_url(url: str) -> str:
    """Prepend https:// if no scheme is present."""
    url = url.strip()
    if url and not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


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

    # Phase 2: L1 discovery — links found directly on the homepage
    l1_urls: list[str] = []
    if homepage_html:
        l1_urls = discover_urls_from_homepage(homepage_html, base)

    # Phase 2b: L2 discovery — fetch top L1 pages and collect their links too.
    # This catches sub-pages that are only linked from e.g. /pricing, not the homepage.
    def _fetch_l2(l1_url: str) -> list[str]:
        try:
            r = requests.get(l1_url, headers=headers, timeout=10)
            if r.status_code == 200:
                return discover_urls_from_homepage(r.text, base)
        except Exception:
            pass
        return []

    l2_urls: list[str] = []
    l2_sources = l1_urls[:MAX_L2_SOURCES]
    if l2_sources:
        with ThreadPoolExecutor(max_workers=len(l2_sources)) as ex:
            for extra in ex.map(_fetch_l2, l2_sources):
                l2_urls.extend(extra)

    # Phase 3: Merge L1 + L2, deduplicate (L1 takes priority), cap total.
    # Only URLs that actually exist on the site are included — no hardcoded guesses.
    all_discovered = list(dict.fromkeys(l1_urls + l2_urls))
    all_other = [u for u in all_discovered if u != base][:MAX_PAGES]

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

    # Assemble in stable order: homepage → L1 → L2
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
        """Compute element-wise median across PSI runs, rebuild display strings."""
        def med(key):
            vals = [r[key] for r in runs if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        def fmt_ms(n):
            if n is None:
                return "N/A"
            return f"{n / 1000:.2f} s" if n >= 1000 else f"{round(n)} ms"

        score_vals = [r["score"] for r in runs if r.get("score") is not None]
        score = round(statistics.median(score_vals)) if score_vals else None

        lcp_ms   = med("lcp_ms")
        fcp_ms   = med("fcp_ms")
        cls_raw  = med("cls_raw")
        tbt_ms   = med("tbt_ms")
        inp_ms   = med("inp_ms")
        ttfb_ms  = med("ttfb_ms")
        speed_ms = med("speed_ms")

        return {
            "score":       score,
            "lcp_ms":      lcp_ms,
            "fcp_ms":      fcp_ms,
            "cls_raw":     cls_raw,
            "tbt_ms":      tbt_ms,
            "inp_ms":      inp_ms,
            "ttfb_ms":     ttfb_ms,
            "speed_ms":    speed_ms,
            # recompute display strings from median numerics
            "lcp":         fmt_ms(lcp_ms),
            "fcp":         fmt_ms(fcp_ms),
            "cls":         f"{cls_raw:.3f}" if cls_raw is not None else "N/A",
            "tbt":         fmt_ms(tbt_ms),
            "inp":         fmt_ms(inp_ms),
            "ttfb":        fmt_ms(ttfb_ms),
            "speed_index": fmt_ms(speed_ms),
            "_run_count":  len(runs),
        }

    # PSI_RUNS requests per strategy all fire in parallel (2 × PSI_RUNS total)
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

    _score_palette = {
        "🟢": ("#e8f9f0", "#34a853"),
        "🟡": ("#fef9e5", "#f9ab00"),
        "🔴": ("#fde8e8", "#e53935"),
        "⚪": ("#f4f4f4", "#9e9e9e"),
    }

    cols = st.columns(len(strategies))
    for col, strategy in zip(cols, strategies):
        m = raw[strategy]
        score = m.get("score")
        run_count = m.get("_run_count", PSI_RUNS)
        icon = score_colour(score)
        bg, accent = _score_palette.get(icon, ("#f4f4f4", "#9e9e9e"))
        score_display = f"{score}/100" if score is not None else "N/A"

        with col:
            # ── Primary score card (visually dominant) ──────────────
            st.markdown(
                f"""<div style="background:{bg};border-left:4px solid {accent};
                    border-radius:6px;padding:0.85rem 1rem 0.8rem;margin-bottom:1rem;">
                  <p style="margin:0 0 0.25rem;font-size:0.7rem;font-weight:700;
                             text-transform:uppercase;letter-spacing:0.06em;color:#555;">
                    {icon}&nbsp;{strategy.capitalize()} Performance Score
                  </p>
                  <p style="margin:0;font-size:2.4rem;font-weight:800;
                             line-height:1;color:#111;">
                    {score_display}
                  </p>
                  <p style="margin:0.3rem 0 0;font-size:0.68rem;color:#777;">
                    Median of {run_count} Lighthouse runs
                  </p>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── Contributing metrics (custom HTML for precise ? alignment) ──
            rows = []
            for key, num_key, display_key in [
                ("lcp",         "lcp_ms",  "lcp"),
                ("fcp",         "fcp_ms",  "fcp"),
                ("cls",         "cls_raw", "cls"),
                ("tbt",         "tbt_ms",  "tbt"),
                ("inp",         "inp_ms",  "inp"),
                ("ttfb",        "ttfb_ms", "ttfb"),
                ("speed_index", "speed_ms","speed_index"),
            ]:
                label_name, help_text = METRIC_HELP[key]
                metric_icon = cwv_colour(key, m.get(num_key))
                val = m.get(display_key, "N/A")
                safe_tip = help_text.replace("<", "&lt;").replace(">", "&gt;")
                rows.append(
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:0.5rem 0;border-bottom:1px solid rgba(0,0,0,0.06);">'
                    f'<div style="display:flex;align-items:center;gap:0.35rem;'
                    f'font-size:0.8rem;color:#444;">'
                    f'<span>{metric_icon}</span>'
                    f'<span>{label_name}</span>'
                    f'<span class="cro-tip">'
                    f'<span style="cursor:help;font-size:0.6rem;color:#bbb;border:1px solid #ddd;'
                    f'border-radius:50%;width:13px;height:13px;display:inline-flex;'
                    f'align-items:center;justify-content:center;flex-shrink:0;'
                    f'font-weight:700;line-height:1;">?</span>'
                    f'<span class="cro-tip-box">{safe_tip}</span>'
                    f'</span>'
                    f'</div>'
                    f'<div style="font-size:0.9rem;font-weight:700;color:#111;">{val}</div>'
                    f'</div>'
                )
            st.markdown(
                '<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:0.06em;color:#999;margin:0 0 0.15rem;">'
                'Contributing metrics</p>'
                f'<div>{"".join(rows)}</div>',
                unsafe_allow_html=True,
            )

    if errors:
        st.caption(f"Note: {', '.join(errors.keys())} data unavailable ({', '.join(errors.values())})")


def _audit_filename(url: str, suffix: str) -> str:
    slug = re.sub(r"https?://", "", url).rstrip("/").replace("/", "-").replace(".", "-")
    return f"cro-audit-{suffix}-{slug}.md"


def _pdf_filename(url: str) -> str:
    slug = re.sub(r"https?://", "", url).rstrip("/").replace("/", "-").replace(".", "-")
    return f"cro-audit-{slug}.pdf"


# ── PDF helpers ──────────────────────────────────────────────────────────────

def _extract_overall_score(report_md: str) -> str | None:
    """Pull the X.X / 10 overall score from the AI markdown, if present."""
    m = re.search(r"\*\*(\d+\.?\d*)\s*/\s*10\*\*", report_md)
    return m.group(1) if m else None


def _score_band(score_int) -> tuple[str, str, str]:
    """(bg_color, text_color, label) for a 0-100 Lighthouse performance score."""
    if score_int is None:
        return "#f1f5f9", "#64748b", "N/A"
    if score_int >= 90:
        return "#d1fae5", "#059669", "Good"
    if score_int >= 50:
        return "#fef3c7", "#d97706", "Needs Work"
    return "#fee2e2", "#dc2626", "Poor"


def _cwv_dot_color(key: str, raw_val) -> str:
    """CSS color for a metric dot indicator."""
    if raw_val is None:
        return "#cbd5e1"
    t = CWV_THRESHOLDS.get(key, {})
    if not t:
        return "#cbd5e1"
    if raw_val <= t["good"]:
        return "#059669"
    if raw_val <= t["poor"]:
        return "#d97706"
    return "#dc2626"


_PSI_COMMENTARY = """
<div class="commentary-block">
<p><strong>What these metrics measure</strong> &mdash; Google Lighthouse scores every page
across five dimensions: <em>Performance</em> (raw speed), <em>Accessibility</em>, <em>Best Practices</em>,
<em>SEO</em>, and <em>Progressive Web App</em>. The scores below focus on <strong>Performance</strong>
and its Core Web Vitals &mdash; the specific signals Google uses as ranking factors and that
directly determine whether visitors stay or leave.</p>
<p><strong>Why it matters for conversion</strong> &mdash; Google research shows a 1-second increase
in load time reduces conversions by up to <strong>7%</strong> and increases bounce rate by <strong>32%</strong>.
Pages scoring below 50 typically lose a significant portion of visitors before they see any content.
Core Web Vitals are also a confirmed Google Search ranking signal &mdash; poor scores harm organic
traffic as well as on-site conversion.</p>
<p><strong>LCP</strong> (how fast the main content appears) drives first impressions &mdash; above 4s
most visitors assume the page is broken. <strong>CLS</strong> (layout shifts) causes mis-clicks on
CTAs as buttons jump while the page loads. <strong>TBT / INP</strong> measure how responsive the page
feels to clicks and form inputs &mdash; critical for sign-up and checkout flows. <strong>TTFB</strong>
is the server&rsquo;s response time and is the foundation every other metric builds on.</p>
</div>
"""

def _build_psi_html(ps_raw: dict) -> str:
    """Render PageSpeed cards as HTML for the PDF report."""
    strategies = [s for s in ("mobile", "desktop") if s in ps_raw]
    if not strategies:
        return ""

    _metrics = [
        ("lcp",         "lcp_ms",  "lcp",         "Largest Contentful Paint"),
        ("fcp",         "fcp_ms",  "fcp",         "First Contentful Paint"),
        ("cls",         "cls_raw", "cls",         "Cumulative Layout Shift"),
        ("tbt",         "tbt_ms",  "tbt",         "Total Blocking Time"),
        ("inp",         "inp_ms",  "inp",         "Interaction to Next Paint"),
        ("ttfb",        "ttfb_ms", "ttfb",        "Time to First Byte"),
        ("speed_index", "speed_ms","speed_index", "Speed Index"),
    ]

    cards = ""
    for strategy in strategies:
        m = ps_raw[strategy]
        score = m.get("score")
        bg, col, label = _score_band(score)
        score_display = str(score) if score is not None else "—"

        rows = ""
        for key, num_key, disp_key, metric_label in _metrics:
            val = m.get(disp_key, "N/A")
            dot_col = _cwv_dot_color(key, m.get(num_key))
            rows += (
                f'<div class="psi-metric-row">'
                f'<span style="display:flex;align-items:center;gap:6px;">'
                f'<span class="psi-metric-dot" style="background:{dot_col};"></span>'
                f'<span class="psi-metric-name">{metric_label}</span>'
                f'</span>'
                f'<span class="psi-metric-val" style="color:{dot_col}">{val}</span>'
                f'</div>'
            )

        cards += (
            f'<div class="psi-card">'
            f'<div class="psi-card-header" style="background:{bg};">'
            f'<div>'
            f'<div class="psi-strategy" style="color:{col};">{strategy.capitalize()}</div>'
            f'<div class="psi-score-row">'
            f'<span class="psi-score-big" style="color:{col};">{score_display}</span>'
            f'<span class="psi-score-denom" style="color:{col};">/100</span>'
            f'</div>'
            f'</div>'
            f'<div class="psi-status-pill" style="background:{col};">{label}</div>'
            f'</div>'
            f'<div class="psi-metrics">{rows}</div>'
            f'</div>'
        )

    return (
        f'<h2>Performance Metrics</h2>'
        f'<p class="psi-section-label">Google Lighthouse &mdash; median of {PSI_RUNS} runs per strategy</p>'
        f'<div class="psi-grid">{cards}</div>'
        f'{_PSI_COMMENTARY}'
    )



def _build_toc_html(report_md: str, has_psi: bool) -> str:
    """Build the Table of Contents page from markdown headings."""
    entries: list[tuple[str, str]] = []  # (display_name, category_hint)

    if has_psi:
        entries.append(("Performance Metrics", "Analytics"))

    for line in report_md.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            # Strip leading ordinal like "0) " or "1) "
            heading = re.sub(r"^\d+\)\s*", "", heading)
            # Strip parenthetical explanations e.g. "(write the actual copy)"
            heading = re.sub(r"\s*\([^)]+\)\s*$", "", heading).strip()
            # Capitalise first letter
            heading = heading[0].upper() + heading[1:] if heading else heading
            # Truncate very long headings
            if len(heading) > 60:
                heading = heading[:57] + "…"
            # Derive a short category hint from the heading
            lh = heading.lower()
            if any(k in lh for k in ("business", "context")):
                cat = "Overview"
            elif any(k in lh for k in ("funnel", "map")):
                cat = "Navigation"
            elif any(k in lh for k in ("executive", "summary")):
                cat = "Summary"
            elif any(k in lh for k in ("scorecard", "score")):
                cat = "Scoring"
            elif any(k in lh for k in ("above", "fold")):
                cat = "UX"
            elif any(k in lh for k in ("issue", "problem")):
                cat = "Issues"
            elif any(k in lh for k in ["mobile"]):
                cat = "Mobile"
            elif any(k in lh for k in ("copy", "cta", "headline")):
                cat = "Copywriting"
            elif any(k in lh for k in ("experiment", "plan", "test")):
                cat = "Action Plan"
            else:
                cat = ""
            entries.append((heading, cat))

    rows = ""
    for i, (name, cat) in enumerate(entries, start=1):
        cat_html = f'<span class="toc-cat">{cat}</span>' if cat else ""
        rows += (
            f'<div class="toc-entry">'
            f'<span class="toc-num">{i:02d}</span>'
            f'<span class="toc-name">{name}</span>'
            f'{cat_html}'
            f'</div>'
        )

    return (
        f'<div class="toc-page">'
        f'<div class="toc-eyebrow">CRO Audit Report</div>'
        f'<h2 class="toc-heading">Contents</h2>'
        f'<div class="toc-accent"></div>'
        f'<div class="toc-entries">{rows}</div>'
        f'</div>'
    )


def build_pdf_html(
    report_md: str,
    ps_raw: dict,
    url: str,
    site_type: str,
) -> str:
    """Assemble the complete styled HTML document for PDF generation."""
    import markdown as _md

    # ── Cover: score badge ───────────────────────────────────────────────────
    raw_score = _extract_overall_score(report_md)
    score_badge_html = ""
    if raw_score:
        val = float(raw_score)
        sc  = "#34d399" if val >= 7.5 else ("#fbbf24" if val >= 5.0 else "#f87171")
        score_badge_html = (
            f'<div class="cover-score-badge" style="border-color:{sc};">'
            f'<div class="cover-score-number" style="color:{sc};">{raw_score}</div>'
            f'<div class="cover-score-denom" style="color:{sc};">/10</div>'
            f'<div class="cover-score-label">CRO Score</div>'
            f'</div>'
        )

    # ── Markdown → HTML ──────────────────────────────────────────────────────
    report_html = _md.markdown(report_md, extensions=["tables"])

    # ── Badge injection ───────────────────────────────────────────────────────
    badge_subs = [
        (r"Impact:\s*High\b",            '<span class="badge badge-impact-high">Impact: High</span>'),
        (r"Impact:\s*Med(?:ium)?\b",      '<span class="badge badge-impact-med">Impact: Med</span>'),
        (r"Impact:\s*Low\b",              '<span class="badge badge-impact-low">Impact: Low</span>'),
        (r"Effort:\s*S\b",                '<span class="badge badge-effort-s">Effort: S</span>'),
        (r"Effort:\s*M\b",                '<span class="badge badge-effort-m">Effort: M</span>'),
        (r"Effort:\s*L\b",                '<span class="badge badge-effort-l">Effort: L</span>'),
        (r"Confidence:\s*High\b",         '<span class="badge badge-conf-high">Confidence: High</span>'),
        (r"Confidence:\s*Med(?:ium)?\b",  '<span class="badge badge-conf-med">Confidence: Med</span>'),
        (r"Confidence:\s*Low\b",          '<span class="badge badge-conf-low">Confidence: Low</span>'),
    ]
    for pattern, repl in badge_subs:
        report_html = re.sub(pattern, repl, report_html)

    # ── Bullet/dash fix ───────────────────────────────────────────────────────
    # Strip leading dash inside <li> (double-marker: CSS bullet + literal dash)
    report_html = re.sub(r"(<li[^>]*>)\s*[-–]\s+", r"\1", report_html)
    # Convert bare "- text" paragraphs → list items (re.DOTALL handles multi-line)
    report_html = re.sub(
        r"<p>\s*[-–]\s+(.+?)</p>",
        r"<ul><li>\1</li></ul>",
        report_html,
        flags=re.DOTALL,
    )
    # Handle the common AI pattern: numbered item followed by "- key: value" sub-items
    # that Python markdown renders as one <p> with <br /> line breaks, e.g.:
    #   <p>1) <strong>Homepage</strong><br />\n   - Purpose: text<br />\n   - CTA: ...</p>
    # Split such paragraphs so the intro becomes <p> and the dash lines become <ul><li>.
    _br_dash = re.compile(r'<br\s*/?>\s*[-–]\s+', re.IGNORECASE)

    def _split_br_dashes(m: re.Match) -> str:
        content = m.group(1)
        if not _br_dash.search(content):
            return m.group(0)
        parts = _br_dash.split(content)
        intro = parts[0].rstrip()
        items = [re.sub(r'<br\s*/?>\s*$', '', p, flags=re.IGNORECASE).strip()
                 for p in parts[1:] if p.strip()]
        li_html = "".join(f"<li>{item}</li>" for item in items)
        return (f"<p>{intro}</p><ul>{li_html}</ul>" if intro else f"<ul>{li_html}</ul>")

    report_html = re.sub(r"<p>(.*?)</p>", _split_br_dashes, report_html, flags=re.DOTALL)
    # Collapse adjacent <ul> blocks created by the conversions above
    report_html = re.sub(r"</ul>\s*<ul>", "\n", report_html)

    # ── Strip parenthetical explanations from headings ────────────────────────
    # e.g. "Copy & CTA improvements (write the actual copy)" → "Copy & CTA improvements"
    report_html = re.sub(
        r'\s*\([^)]+\)(?=\s*</h[23]>)',
        "",
        report_html,
        flags=re.IGNORECASE,
    )

    # ── Executive summary card ────────────────────────────────────────────────
    # Wrap the executive summary section content in a styled card div.
    report_html = re.sub(
        r'(<h2>[^<]*(?:executive[^<]*summary|summary[^<]*executive)[^<]*</h2>)(.*?)(?=<h2>|\Z)',
        lambda m: m.group(1) + '<div class="exec-summary-card">' + m.group(2).strip() + '</div>',
        report_html,
        flags=re.DOTALL | re.IGNORECASE,
        count=1,
    )

    # ── Acronym expansion (first occurrence only) ─────────────────────────────
    _acronyms = {
        "CRO":  "Conversion Rate Optimisation",
        "CTA":  "Call to Action",
        "CTAs": "Calls to Action",
        "LCP":  "Largest Contentful Paint",
        "CLS":  "Cumulative Layout Shift",
        "TBT":  "Total Blocking Time",
        "INP":  "Interaction to Next Paint",
        "TTFB": "Time to First Byte",
        "FCP":  "First Contentful Paint",
        "UX":   "User Experience",
        "SEO":  "Search Engine Optimisation",
        "CTR":  "Click-Through Rate",
        "MDE":  "Minimum Detectable Effect",
        "ROI":  "Return on Investment",
        "KPI":  "Key Performance Indicator",
        "B2B":  "Business-to-Business",
        "B2C":  "Business-to-Consumer",
    }
    for acronym, expansion in _acronyms.items():
        # Replace only the first occurrence; skip if already written as "expansion (acronym)"
        if f"{expansion} ({acronym})" not in report_html:
            report_html = re.sub(
                rf'\b{re.escape(acronym)}\b',
                f'{acronym} ({expansion})',
                report_html,
                count=1,
            )

    # ── Metadata ─────────────────────────────────────────────────────────────
    from datetime import date as _date
    today        = _date.today().strftime("%B %d, %Y")
    today_upper  = today.upper()
    display_url  = re.sub(r"^https?://", "", url).rstrip("/")

    # ── Component blocks ─────────────────────────────────────────────────────
    has_psi  = bool(ps_raw and any(s in ps_raw for s in ("mobile", "desktop")))
    psi_html = _build_psi_html(ps_raw)
    toc_html = _build_toc_html(report_md, has_psi)

    # ── Dynamic @page rule: margin-box footer (date injected at render time) ─
    # @page :first = cover page (no footer); @page = all other pages.
    page_css = f"""
@page :first {{
    size: A4;
    margin: 0;
}}
@page {{
    size: A4;
    margin: 14mm 18mm 11mm;
    @bottom-left {{
        content: "CONFIDENTIAL \\2014  {today_upper}";
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        font-size: 7pt;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.08em;
        border-top: 1px solid #e2e8f0;
        padding: 3mm 0 0 0;
        width: 50%;
    }}
    @bottom-right {{
        content: "CRO AUDIT AI";
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        font-size: 7pt;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.08em;
        border-top: 1px solid #e2e8f0;
        padding: 3mm 0 0 0;
        text-align: right;
        width: 50%;
    }}
}}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
{_PDF_CSS}
{page_css}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════ COVER PAGE -->
<div class="cover">
  <div class="cover-accent-bar"></div>
  <div class="cover-body">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;">
      <div class="cover-eyebrow">Conversion Rate Optimisation</div>
      {score_badge_html}
    </div>
    <div class="cover-spacer"></div>
    <div class="cover-title">CRO Audit<br>Report</div>
    <div class="cover-subtitle">{today}</div>
    <div class="cover-url-card">
      <div class="cover-url-label">Audited Website</div>
      <div class="cover-url-value">{display_url}</div>
    </div>
    <div class="cover-meta-row">
      <div class="cover-meta-item">
        <div class="cover-meta-label">Report Date</div>
        <div class="cover-meta-value">{today}</div>
      </div>
      <div class="cover-meta-item">
        <div class="cover-meta-label">Audit Type</div>
        <div class="cover-meta-value">{site_type}</div>
      </div>
      <div class="cover-meta-item">
        <div class="cover-meta-label">Classification</div>
        <div class="cover-meta-value">Confidential</div>
      </div>
    </div>
  </div>
  <div class="cover-footer">
    <div class="cover-footer-brand">CRO Audit AI</div>
    <div>Prepared {today} &mdash; Confidential</div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════ CONTENTS PAGE -->
{toc_html}

<!-- ═══════════════════════════════════════════════════ MAIN CONTENT -->
<div class="main-content">

{psi_html}

<div class="report-body">
{report_html}
</div>


</div>

</body>
</html>"""


def generate_pdf_bytes(html: str) -> bytes:
    """Write HTML to a temp file, call pdf_worker.py via Playwright, return PDF bytes."""
    import base64
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as fh:
        fh.write(html)
        tmp_path = fh.name

    try:
        payload = json.dumps({"html_path": tmp_path})
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "pdf_worker.py")],
            input=payload.encode(),
            capture_output=True,
            timeout=90,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace")[:600])
        return base64.b64decode(result.stdout.strip())
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _polish_report(text: str) -> str:
    """Normalise spacing in AI-generated markdown so every section breathes."""
    # Ensure a blank line after every heading if one isn't already there
    text = re.sub(r"(^#{1,4} [^\n]+)\n(?!\n)", r"\1\n\n", text, flags=re.MULTILINE)
    # Ensure a blank line before every heading (except at the very start)
    text = re.sub(r"([^\n])\n(#{1,4} )", r"\1\n\n\2", text)
    # Collapse 3+ consecutive blank lines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI CRO Audit Tool", page_icon="📈")
st.markdown(_REPORT_CSS, unsafe_allow_html=True)
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

vurl = st.text_input("Website URL for screenshots", key="vision_url")
col_opts_a, col_opts_b = st.columns(2)
with col_opts_a:
    include_text = st.checkbox("Include text context", value=True, key="vision_include_text")
with col_opts_b:
    gen_pdf = st.checkbox("Generate PDF report", value=True, key="vision_gen_pdf")

if st.button("Run Vision Audit", key="run_vision", type="primary"):
    vurl = normalise_url(vurl)
    if not vurl or not is_valid_url(vurl):
        st.error("Please enter a valid URL (e.g. example.com)")
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_psi = executor.submit(fetch_pagespeed, vurl)
            if include_text:
                f_text = executor.submit(extract_text_from_url, vurl)
        ps_context, ps_raw = f_psi.result()
        if include_text:
            text_context, scraped_pages = f_text.result()
            if scraped_pages:
                st.info(f"Text scraped from {len(scraped_pages)} page(s): {', '.join(scraped_pages)}")

    with st.spinner(f"Running vision audit ({MODEL_VISION})..."):
        result = run_ai_vision(text_context, shots, pagespeed=ps_context, site_type=site_type)

    st.session_state["vision_result"] = result
    st.session_state["vision_result_url"] = vurl
    st.session_state["vision_shots"] = shots
    st.session_state["vision_scraped_pages"] = scraped_pages
    st.session_state["vision_ps_raw"] = ps_raw
    st.session_state["vision_site_type"] = site_type
    st.session_state["vision_pdf_bytes"] = None  # reset

    if gen_pdf:
        with st.spinner("Generating PDF report..."):
            try:
                pdf_html = build_pdf_html(result, ps_raw, vurl, site_type)
                st.session_state["vision_pdf_bytes"] = generate_pdf_bytes(pdf_html)
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

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
    st.markdown(_polish_report(st.session_state["vision_result"]))

    dl_col_a, dl_col_b = st.columns([1, 1])
    with dl_col_a:
        st.download_button(
            label="⬇ Download audit (.md)",
            data=st.session_state["vision_result"],
            file_name=_audit_filename(st.session_state["vision_result_url"], "vision"),
            mime="text/markdown",
            use_container_width=True,
        )
    with dl_col_b:
        pdf_bytes = st.session_state.get("vision_pdf_bytes")
        if pdf_bytes:
            st.download_button(
                label="⬇ Download PDF report",
                data=pdf_bytes,
                file_name=_pdf_filename(st.session_state["vision_result_url"]),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            if st.button("Generate PDF report", key="gen_pdf_post", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_html = build_pdf_html(
                            st.session_state["vision_result"],
                            st.session_state.get("vision_ps_raw", {}),
                            st.session_state["vision_result_url"],
                            st.session_state.get("vision_site_type", "SaaS"),
                        )
                        st.session_state["vision_pdf_bytes"] = generate_pdf_bytes(pdf_html)
                        st.rerun()
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

    # ── Dev tool: save fixture for fast PDF preview without re-running audit ──
    with st.expander("🛠 Dev: Save PDF test fixture"):
        st.caption(
            "Saves the current audit data to `audit_fixture.json` so you can "
            "run `python preview_pdf.py` to preview formatting changes instantly "
            "without re-running the audit."
        )
        if st.button("Save audit_fixture.json", key="save_fixture"):
            import json as _json
            fixture = {
                "report_md": st.session_state["vision_result"],
                "ps_raw":    st.session_state.get("vision_ps_raw", {}),
                "url":       st.session_state["vision_result_url"],
                "site_type": st.session_state.get("vision_site_type", "SaaS"),
            }
            fixture_path = PROJECT_ROOT / "audit_fixture.json"
            fixture_path.write_text(
                _json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            st.success(f"Saved → {fixture_path}")

    with st.expander("Visited pages (final URLs)"):
        shots_display = st.session_state.get("vision_shots", [])
        st.write([s.get("final_url") or s.get("url") for s in shots_display])
