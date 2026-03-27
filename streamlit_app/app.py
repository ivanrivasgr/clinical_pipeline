"""
Clinical Alert Intelligence Dashboard — Streamlit App
Run: streamlit run streamlit_app/app.py

Tabs:
  1. Pipeline Overview
  2. Clinical View (Patient / Room Timeline)
  3. Nursing Leadership (Noise & Operations)
  4. M&M / Leadership (Escalation & Critical Patterns)
  5. BI Export (Power BI)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pipeline.layer1_ingest    import load_file
from pipeline.layer2_clean     import clean
from pipeline.layer3_events    import build_event_table
from pipeline.layer4_analytics import (
    analyze_duration, analyze_frequency, analyze_noise,
    detect_escalation, detect_bursts, compute_time_between_events
)
from pipeline.layer5_bi_output import export_bi_datasets


# ─── Color palette — Driscoll Children's Hospital Brand ───────────────────────
#     Navy #1b2e5a  |  Gold #f2b824  |  Red accent #c8352e

C = {
    # Backgrounds – Driscoll navy family
    "bg":          "#111e38",
    "bg_alt":      "#152545",
    "bg_card":     "#1a2d50",
    "bg_sidebar":  "#0e1930",
    "bg_surface":  "#1f345a",

    # Borders
    "border":      "#2a4170",
    "border_lite": "#243968",

    # Text
    "t1":          "#eef1f6",   # headings — near white
    "t2":          "#c5cfe0",   # body
    "t3":          "#7e92b4",   # muted / labels

    # Accent — Driscoll gold + navy
    "teal":        "#f2b824",   # Driscoll gold (primary accent)
    "blue":        "#4a7edd",   # lighter navy for interactive elements
    "sky":         "#3b6bc4",   # Driscoll navy mid
    "cyan":        "#5c9cf5",   # bright blue for data highlights

    # Severity — clinical standard with Driscoll red
    "green":       "#34d399",
    "amber":       "#f2b824",   # Driscoll gold doubles as warning
    "red":         "#e04040",   # derived from Driscoll red #c8352e
    "red_deep":    "#c8352e",   # exact Driscoll red

    # Extra
    "purple":      "#a78bfa",
    "pink":        "#f472b6",

    # Driscoll brand — direct reference
    "driscoll_navy":  "#1b2e5a",
    "driscoll_gold":  "#f2b824",
    "driscoll_red":   "#c8352e",

    # Chart internals
    "chart_bg":    "rgba(0,0,0,0)",
    "grid":        "rgba(126,146,180,0.08)",
}

SEV_COLOR = {1: C["green"], 2: C["amber"], 3: C["red"]}


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Alert Intelligence — Driscoll RGV",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  :root {{
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    color-scheme: dark;
  }}

  /* ── Backgrounds ── */
  .stApp {{
    background: linear-gradient(170deg, {C['bg']} 0%, #13223f 40%, #162848 100%) !important;
    color: {C['t2']} !important;
    font-family: var(--font-body) !important;
  }}
  .main .block-container {{
    background: transparent !important;
    padding-top: 2rem !important;
    color: {C['t2']} !important;
  }}

  header[data-testid="stHeader"] {{
    background: rgba(17, 30, 56, 0.90) !important;
    backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid {C['border']} !important;
  }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(195deg, {C['bg_sidebar']} 0%, #111f3a 60%, #142648 100%) !important;
    border-right: 1px solid {C['border']} !important;
  }}
  section[data-testid="stSidebar"] * {{
    color: {C['t2']} !important;
    font-family: var(--font-body) !important;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {C['bg_card']} !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid {C['border']} !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {C['t3']} !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: {C['t2']} !important;
    background: rgba(242, 184, 36, 0.06) !important;
  }}
  .stTabs [aria-selected="true"] {{
    color: {C['driscoll_gold']} !important;
    background: rgba(242, 184, 36, 0.1) !important;
    box-shadow: 0 0 14px rgba(242, 184, 36, 0.08) !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent !important; }}
  .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

  /* ── Metrics (built-in) ── */
  [data-testid="stMetric"] {{ background: transparent !important; padding: 0 !important; }}
  [data-testid="stMetricLabel"] p {{
    font-family: var(--font-body) !important;
    font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    color: {C['t3']} !important;
  }}
  [data-testid="stMetricValue"] {{
    font-family: var(--font-mono) !important;
    font-size: 26px !important; font-weight: 700 !important;
    color: {C['t1']} !important;
  }}
  [data-testid="stMetricDelta"] {{ font-family: var(--font-mono) !important; font-size: 12px !important; }}

  /* ── Headings ── */
  h1, h2, h3 {{ font-family: var(--font-body) !important; color: {C['t1']} !important; font-weight: 700 !important; }}

  /* ── DataFrames ── */
  .stDataFrame, [data-testid="stDataFrame"] {{
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
  }}

  /* ── Buttons ── */
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {C['driscoll_navy']} 0%, #243d6e 100%) !important;
    border: 1px solid rgba(242, 184, 36, 0.3) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important; font-size: 13px !important;
    letter-spacing: 0.03em !important;
    border-radius: 8px !important; padding: 10px 20px !important;
    box-shadow: 0 2px 14px rgba(242, 184, 36, 0.15) !important;
    transition: all 0.25s ease !important;
    color: {C['driscoll_gold']} !important;
  }}
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="stBaseButton-primary"]:hover {{
    box-shadow: 0 4px 22px rgba(242, 184, 36, 0.3) !important;
    transform: translateY(-1px) !important;
    border-color: {C['driscoll_gold']} !important;
  }}

  /* ── Download buttons ── */
  .stDownloadButton > button {{
    background: {C['bg_card']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['t2']} !important;
    font-family: var(--font-mono) !important; font-size: 12px !important;
    border-radius: 8px !important; transition: all 0.2s ease !important;
  }}
  .stDownloadButton > button:hover {{
    border-color: {C['driscoll_gold']} !important;
    box-shadow: 0 0 12px rgba(242, 184, 36, 0.15) !important;
  }}

  /* ── Expander ── */
  .streamlit-expanderHeader {{
    background: {C['bg_card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['t2']} !important;
  }}

  /* ── Inputs ── */
  .stSelectbox > div > div, .stSlider > div, [data-testid="stFileUploader"] {{
    font-family: var(--font-body) !important;
  }}

  /* ── Dividers ── */
  hr {{ border-color: {C['border']} !important; opacity: 0.4 !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {C['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {C['t3']}; }}

  /* ── Alert overrides ── */
  .stAlert {{ border-radius: 8px !important; font-family: var(--font-body) !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _rgba(hex_color, alpha):
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def kpi_card(label, value, subtitle="", severity="neutral", icon=""):
    accent_map = {
        "critical": C["red"],
        "warning":  C["amber"],
        "good":     C["green"],
        "info":     C["blue"],
        "neutral":  C["t3"],
        "purple":   C["purple"],
    }
    accent = accent_map.get(severity, C["t3"])
    sub_html = f'<div style="font-family:var(--font-mono);font-size:11px;color:{accent};margin-top:5px;font-weight:500;">{subtitle}</div>' if subtitle else ""
    icon_html = f'<span style="font-size:18px;margin-right:5px;">{icon}</span>' if icon else ""

    st.markdown(f"""
    <div style="
      background: {_rgba(accent, 0.06)};
      border: 1px solid {_rgba(accent, 0.16)};
      border-radius: 12px;
      padding: 20px 22px;
      position: relative;
      overflow: hidden;
    ">
      <div style="
        position:absolute;top:0;left:0;width:3px;height:100%;
        background:linear-gradient(180deg,{accent} 0%,transparent 100%);
      "></div>
      <div style="font-family:var(--font-body);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:{C['t3']};margin-bottom:8px;">
        {icon_html}{label}
      </div>
      <div style="font-family:var(--font-mono);font-size:30px;font-weight:700;color:{C['t1']};line-height:1.1;">
        {value}
      </div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle="", icon=""):
    sub_html = f'<span style="font-size:13px;color:{C["t3"]};font-weight:400;margin-left:10px;">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:28px 0 14px 0;display:flex;align-items:baseline;gap:8px;">
      <span style="font-size:18px;">{icon}</span>
      <span style="font-family:var(--font-body);font-size:18px;font-weight:700;color:{C['t1']};">{title}</span>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def clinical_layout(fig, height=380, show_legend=False, xtitle="", ytitle="", title_text=""):
    """Apply clinical theme. Explicitly sets title text to prevent 'undefined'."""
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color=C["t1"], family="Inter, sans-serif")),
        template="plotly_dark",
        paper_bgcolor=C["chart_bg"],
        plot_bgcolor=C["chart_bg"],
        font=dict(family="Inter, sans-serif", color=C["t3"], size=11),
        height=height,
        margin=dict(l=16, r=16, t=44 if title_text else 24, b=16),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color=C["t3"]),
        ),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"],
                   title=dict(text=xtitle, font=dict(size=11, color=C["t3"]))),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"],
                   title=dict(text=ytitle, font=dict(size=11, color=C["t3"]))),
        hoverlabel=dict(
            bgcolor=C["bg_card"], bordercolor=C["border"],
            font=dict(family="JetBrains Mono, monospace", size=11, color=C["t2"]),
        ),
    )
    return fig


def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom:22px;">
      <h1 style="font-family:var(--font-body);font-size:28px;font-weight:800;color:{C['t1']};margin:0 0 6px 0;letter-spacing:-0.01em;">
        {title}
      </h1>
      <div style="width:50px;height:3px;background:linear-gradient(90deg,{C['driscoll_gold']},{C['driscoll_red']});border-radius:2px;margin-bottom:8px;"></div>
      <p style="font-family:var(--font-body);font-size:13px;color:{C['t3']};margin:0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def empty_state(icon, message):
    st.markdown(f"""
    <div style="text-align:center;padding:50px 20px;border:1px dashed {C['border']};border-radius:12px;background:{_rgba(C['green'], 0.03)};margin-top:16px;">
      <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
      <div style="font-family:var(--font-body);font-size:13px;color:{C['t3']};">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 8px 0;">
      <div style="font-size:34px;margin-bottom:4px;">🫀</div>
      <div style="font-family:var(--font-body);font-size:18px;font-weight:800;color:{C['t1']};letter-spacing:-0.02em;line-height:1.2;">
        Clinical Alert<br>Intelligence
      </div>
      <div style="font-family:var(--font-mono);font-size:10px;font-weight:500;color:{C['t3']};margin-top:6px;letter-spacing:0.08em;text-transform:uppercase;">
        PIC iX Monitor Analyzer
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="border-top:1px solid {C["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)

    # Status light
    if st.session_state.pipeline_run:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:{_rgba(C['green'],0.07)};border:1px solid {_rgba(C['green'],0.18)};border-radius:8px;margin-bottom:14px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{C['green']};box-shadow:0 0 8px {C['green']};"></div>
          <span style="font-family:var(--font-mono);font-size:11px;color:{C['green']};font-weight:600;">PIPELINE ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:{_rgba(C['t3'],0.06)};border:1px solid {_rgba(C['t3'],0.12)};border-radius:8px;margin-bottom:14px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{C['t3']};"></div>
          <span style="font-family:var(--font-mono);font-size:11px;color:{C['t3']};font-weight:600;">AWAITING DATA</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:10px;font-weight:600;color:{C['t3']};letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
      ▸ Data Input
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Alert Export",
        type=['csv', 'xlsx', 'xls', 'xlm'],
        help="CSV or Excel export from PIC iX monitoring system",
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:{_rgba(C['blue'],0.06)};border:1px solid {_rgba(C['blue'],0.16)};border-radius:8px;margin:8px 0;">
          <span style="font-size:14px;">📄</span>
          <span style="font-family:var(--font-mono);font-size:11px;color:{C['blue']};font-weight:500;word-break:break-all;">{uploaded_file.name}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-family:var(--font-mono);font-size:10px;font-weight:600;color:{C['t3']};letter-spacing:0.1em;text-transform:uppercase;margin:16px 0 8px 0;">
          ▸ Parameters
        </div>
        """, unsafe_allow_html=True)

        noise_thresh = st.slider(
            "Noise threshold (sec)",
            min_value=5, max_value=60, value=10,
            help="Alarms shorter than this are flagged as noise",
        )

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        if st.button("⚡  Execute Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing alert data..."):
                import tempfile, re
                safe_name = re.sub(r'\s+', '_', uploaded_file.name)
                tmp_path  = Path(tempfile.gettempdir()) / safe_name
                tmp_path.write_bytes(uploaded_file.read())

                raw_df    = load_file(tmp_path)
                clean_df  = clean(raw_df)
                events_df = build_event_table(clean_df)
                events_df = compute_time_between_events(events_df)

                events_df['is_noise'] = events_df['duration_sec'] < noise_thresh

                esc_df, pre_crit_df = detect_escalation(events_df)
                bursts_df           = detect_bursts(events_df)
                duration_stats      = analyze_duration(events_df)
                freq_stats          = analyze_frequency(events_df)
                noise_stats         = analyze_noise(events_df)

                st.session_state.events_df      = events_df
                st.session_state.clean_df       = clean_df
                st.session_state.esc_df         = esc_df
                st.session_state.pre_crit_df    = pre_crit_df
                st.session_state.bursts_df      = bursts_df
                st.session_state.duration_stats = duration_stats
                st.session_state.freq_stats     = freq_stats
                st.session_state.noise_stats    = noise_stats
                st.session_state.pipeline_run   = True
                st.session_state.noise_thresh   = noise_thresh

            st.success("Pipeline complete!")
            st.rerun()

    st.markdown(f'<div style="border-top:1px solid {C["border"]};margin:20px 0 12px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;padding:4px 0;">
      <div style="width:40px;height:3px;background:linear-gradient(90deg,{C['driscoll_gold']},{C['driscoll_red']});margin:0 auto 8px auto;border-radius:2px;"></div>
      <div style="font-family:var(--font-body);font-size:12px;font-weight:700;color:{C['driscoll_gold']};">Driscoll Children's Hospital</div>
      <div style="font-family:var(--font-mono);font-size:9px;color:{C['t3']};margin-top:4px;letter-spacing:0.05em;">RGV Clinical Informatics</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.pipeline_run:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px 30px 20px;">
      <div style="font-size:52px;margin-bottom:12px;">🫀</div>
      <h1 style="font-family:var(--font-body);font-size:36px;font-weight:800;color:{C['t1']};letter-spacing:-0.02em;margin:0;">
        Clinical Alert Intelligence
      </h1>
      <div style="width:80px;height:3px;background:linear-gradient(90deg,{C['driscoll_gold']},{C['driscoll_red']});border-radius:2px;margin:12px auto;"></div>
      <p style="font-family:var(--font-body);font-size:16px;color:{C['t3']};margin-top:4px;font-weight:400;">
        Transform unstructured PIC iX monitoring data into actionable clinical insight
      </p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Raw Ingestion",  "Load CSV / Excel from PIC iX export",                    C["blue"]),
        ("02", "Clean & Parse",  "Extract severity, event type, metric, threshold",         C["cyan"]),
        ("03", "Event Pairing",  "Pair Generated ↔ Ended events, compute duration",        C["purple"]),
        ("04", "Analytics",      "Duration, frequency, noise, escalation, burst detection", C["amber"]),
        ("05", "BI Output",      "Power BI-ready Parquet + CSV star-schema datasets",       C["green"]),
    ]

    cols = st.columns(5)
    for i, (num, title, desc, color) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="
              background: {_rgba(color, 0.05)};
              border: 1px solid {_rgba(color, 0.15)};
              border-radius: 12px;
              padding: 24px 16px;
              text-align: center;
              min-height: 175px;
              display: flex; flex-direction: column; align-items: center; justify-content: center;
            ">
              <div style="font-family:var(--font-mono);font-size:28px;font-weight:800;color:{color};opacity:0.65;margin-bottom:8px;">
                {num}
              </div>
              <div style="font-family:var(--font-body);font-size:14px;font-weight:700;color:{C['t1']};margin-bottom:6px;">
                {title}
              </div>
              <div style="font-family:var(--font-body);font-size:11.5px;color:{C['t3']};line-height:1.5;">
                {desc}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:40px;padding:18px;border:1px dashed {C['border']};border-radius:12px;background:{_rgba(C['blue'],0.03)};">
      <span style="font-family:var(--font-body);font-size:14px;color:{C['t3']};">
        Upload a PIC iX export file in the sidebar to begin analysis
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

events_df      = st.session_state.events_df
esc_df         = st.session_state.esc_df
pre_crit_df    = st.session_state.pre_crit_df
bursts_df      = st.session_state.bursts_df
duration_stats = st.session_state.duration_stats
freq_stats     = st.session_state.freq_stats
noise_stats    = st.session_state.noise_stats


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Overview  ",
    "  Clinical View  ",
    "  Nursing Leadership  ",
    "  M&M / Leadership  ",
    "  BI Export  ",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    page_header("Pipeline Overview", "High-level summary of processed alarm data from PIC iX")

    total    = len(events_df)
    critical = int((events_df['severity'] == 3).sum())
    noise_n  = int(events_df['is_noise'].sum())
    noise_p  = round(noise_n / total * 100, 1) if total else 0
    rooms    = events_df['room_id'].nunique()
    esc_cnt  = len(esc_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Alarms", f"{total:,}", icon="📊", severity="info")
    with c2: kpi_card("Unique Rooms", f"{rooms}", icon="🏠", severity="neutral")
    with c3: kpi_card("Critical ★★★", f"{critical:,}", f"{critical/total*100:.1f}% of total" if total else "", severity="critical", icon="🔴")
    with c4:
        sev = "critical" if noise_p > 40 else ("warning" if noise_p > 25 else "good")
        kpi_card("Noise Rate", f"{noise_p}%", f"{noise_n:,} alarms", severity=sev, icon="📢")
    with c5: kpi_card("Escalations", f"{esc_cnt:,}", f"{len(bursts_df)} bursts", severity="warning" if esc_cnt > 100 else "info", icon="⚡")

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Alarms by Event Type", icon="📈")
        tc = events_df['event_type'].value_counts().reset_index()
        tc.columns = ['event_type', 'count']
        fig = px.bar(
            tc.head(15), x='count', y='event_type', orientation='h',
            color='count',
            color_continuous_scale=[[0, "#1b2e5a"], [0.5, "#3b6bc4"], [1, "#f2b824"]],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        clinical_layout(fig, height=420, xtitle="Count")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        section_header("Severity Distribution", icon="🎯")
        sev_df = events_df['severity'].value_counts().sort_index().reset_index()
        sev_df.columns = ['severity', 'count']
        sev_df['label'] = sev_df['severity'].map({1: '★ Low', 2: '★★ Medium', 3: '★★★ Critical'})
        fig2 = px.pie(
            sev_df, values='count', names='label',
            color='label',
            color_discrete_map={'★ Low': C['green'], '★★ Medium': C['amber'], '★★★ Critical': C['red']},
            hole=0.55,
        )
        fig2.update_traces(
            textposition='outside', textinfo='label+percent',
            textfont=dict(size=12, family="Inter, sans-serif"),
            marker=dict(line=dict(color=C['bg'], width=2)),
        )
        fig2.update_layout(
            annotations=[dict(
                text=f"<b>{total:,}</b><br><span style='font-size:10px;color:{C['t3']}'>total</span>",
                x=0.5, y=0.5, font_size=22, font_family="JetBrains Mono, monospace",
                font_color=C['t1'], showarrow=False,
            )]
        )
        clinical_layout(fig2, height=420)
        st.plotly_chart(fig2, width="stretch")

    section_header("Alarm Activity by Hour", subtitle="24-hour distribution with critical overlay", icon="🕐")
    hourly = freq_stats['by_hour'].copy()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=hourly['hour'], y=hourly['total_alerts'],
        name='Total', marker_color=C['blue'], opacity=0.35,
        marker_line_width=0,
    ))
    if 'critical_alerts' in hourly.columns:
        fig3.add_trace(go.Bar(
            x=hourly['hour'], y=hourly['critical_alerts'],
            name='Critical', marker_color=C['red'], opacity=0.9,
            marker_line_width=0,
        ))
    clinical_layout(fig3, height=300, show_legend=True, xtitle="Hour of Day", ytitle="Alarm Count")
    fig3.update_layout(barmode='overlay')
    st.plotly_chart(fig3, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLINICAL VIEW
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    page_header("Clinical View", "Patient-level temporal view — identify deterioration patterns per room")

    col_room, col_date = st.columns([1, 2])
    with col_room:
        selected_room = st.selectbox("Select Room", options=sorted(events_df['room_id'].unique()))
    with col_date:
        dates = sorted(events_df['date'].unique())
        if len(dates) > 1:
            date_range = st.select_slider(
                "Date Range",
                options=[str(d) for d in dates],
                value=(str(dates[0]), str(dates[-1])),
            )
        else:
            date_range = (str(dates[0]), str(dates[0]))

    room_df = events_df[
        (events_df['room_id'] == selected_room) &
        (events_df['date'].astype(str) >= date_range[0]) &
        (events_df['date'].astype(str) <= date_range[1])
    ].copy()

    if room_df.empty:
        empty_state("🔍", "No alarm data for the selected room and date range")
    else:
        r_total = len(room_df)
        r_crit  = int((room_df['severity'] == 3).sum())
        r_noise = room_df['is_noise'].mean() * 100
        r_med   = room_df['duration_sec'].median()

        r1, r2, r3, r4 = st.columns(4)
        with r1: kpi_card("Room Alarms", f"{r_total:,}", icon="📋", severity="info")
        with r2: kpi_card("Critical", f"{r_crit}", f"{r_crit/r_total*100:.0f}% of room" if r_total else "", severity="critical" if r_crit > 0 else "good", icon="🔴")
        with r3: kpi_card("Noise Rate", f"{r_noise:.0f}%", severity="critical" if r_noise > 50 else ("warning" if r_noise > 30 else "good"), icon="📢")
        with r4: kpi_card("Median Duration", f"{r_med:.0f}s", severity="neutral", icon="⏱")

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        section_header(f"Alert Timeline — Room {selected_room}", icon="📐")
        fig_tl = go.Figure()
        for _, row in room_df.iterrows():
            if pd.isna(row['end_time']):
                continue
            sv = int(row['severity'])
            fig_tl.add_trace(go.Scatter(
                x=[row['start_time'], row['end_time'], None],
                y=[row['event_type']] * 3,
                mode='lines',
                line=dict(color=SEV_COLOR.get(sv, '#555'), width=6 + sv * 3),
                name=row['severity_label'],
                hovertemplate=(
                    f"<b>{row['event_type']}</b><br>"
                    f"Duration: {row['duration_sec']:.0f}s<br>"
                    f"Severity: {row['severity_label']}<br>"
                    f"Value: {row['metric_value']} {row['condition']} {row['threshold']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
        clinical_layout(fig_tl, height=480, xtitle="Time")
        fig_tl.update_layout(hovermode='closest')
        st.plotly_chart(fig_tl, width="stretch")

        section_header("Severity Progression Over Time", icon="📉")
        fig_sev = px.scatter(
            room_df, x='start_time', y='event_type',
            size='duration_sec', color='severity',
            color_continuous_scale=[C['green'], C['amber'], C['red']],
            hover_data=['metric_value', 'threshold', 'duration_sec'],
        )
        fig_sev.update_traces(marker=dict(line=dict(width=0), opacity=0.8))
        clinical_layout(fig_sev, height=350)
        fig_sev.update_layout(coloraxis_colorbar=dict(
            title=dict(text="Severity"), tickvals=[1, 2, 3], ticktext=["Low", "Med", "Crit"],
            len=0.5, thickness=12,
        ))
        st.plotly_chart(fig_sev, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — NURSING LEADERSHIP
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    page_header("Nursing Leadership", "Noise analysis, threshold optimization, and alert fatigue reduction")

    noise = noise_stats

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total Alarms", f"{noise['total_noise'] + noise['total_signal']:,}", icon="📊", severity="info")
    with k2: kpi_card("Signal Alarms", f"{noise['total_signal']:,}", icon="✅", severity="good")
    with k3:
        np_val = noise['overall_noise_pct']
        kpi_card("Noise Alarms", f"{noise['total_noise']:,}", f"{np_val}% are noise", severity="critical" if np_val > 40 else "warning", icon="🔇")
    with k4: kpi_card("Mean Duration", f"{duration_stats['overall']['mean_sec']}s", icon="⏱", severity="neutral")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # Fatigue warning
    np_val = noise['overall_noise_pct']
    if np_val > 30:
        ac = C['red'] if np_val > 45 else C['amber']
        st.markdown(f"""
        <div style="
          background:{_rgba(ac,0.06)};
          border:1px solid {_rgba(ac,0.18)};
          border-left:3px solid {ac};
          border-radius:8px;
          padding:14px 18px;
          margin-bottom:18px;
        ">
          <span style="font-family:var(--font-body);font-size:13px;color:{C['t2']};">
            <b style="color:{ac};">⚠ ALERT FATIGUE RISK</b> — {np_val}% of all alarms are noise
            (duration &lt; {st.session_state.noise_thresh}s). These provide no clinical value and
            directly contribute to alarm desensitization among nursing staff.
          </span>
        </div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        section_header("Noise % by Event Type", icon="📊")
        noise_type = noise['noise_by_type'].sort_values('noise_pct', ascending=False)
        fig_n = px.bar(
            noise_type, x='event_type', y='noise_pct',
            color='noise_pct',
            color_continuous_scale=[[0, "#451a03"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        )
        fig_n.update_traces(marker_line_width=0)
        fig_n.add_hline(y=50, line_dash='dot', line_color=C['red'], line_width=1,
                        annotation_text="50% noise level", annotation_font_size=10,
                        annotation_font_color=C['red'])
        clinical_layout(fig_n, height=380, xtitle="Event Type", ytitle="Noise %")
        fig_n.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_n, width="stretch")

    with col_r:
        section_header("Duration Distribution", icon="⏱")
        dur_data = events_df['duration_sec'].dropna()
        dur_clip = dur_data[dur_data <= 300]
        fig_hist = px.histogram(dur_clip, nbins=60, color_discrete_sequence=[C['blue']])
        fig_hist.update_traces(marker_line_width=0, opacity=0.7)
        fig_hist.add_vline(
            x=st.session_state.noise_thresh, line_dash='dot',
            line_color=C['red'], line_width=2,
            annotation_text=f"Noise cutoff ({st.session_state.noise_thresh}s)",
            annotation_font_size=10, annotation_font_color=C['red'],
        )
        clinical_layout(fig_hist, height=380, xtitle="Duration (seconds)", ytitle="Count")
        st.plotly_chart(fig_hist, width="stretch")

    section_header("Signal vs Noise by Room", icon="🏠")
    room_stats = freq_stats['by_room'].copy()
    room_stats['signal_alerts'] = room_stats['total_alerts'] - room_stats['noise_alerts']
    room_stats = room_stats.sort_values('total_alerts', ascending=False)

    fig_room = go.Figure()
    fig_room.add_trace(go.Bar(
        name='Signal', x=room_stats['room_id'], y=room_stats['signal_alerts'],
        marker_color=C['green'], marker_line_width=0, opacity=0.85,
    ))
    fig_room.add_trace(go.Bar(
        name='Noise', x=room_stats['room_id'], y=room_stats['noise_alerts'],
        marker_color=C['red'], marker_line_width=0, opacity=0.7,
    ))
    clinical_layout(fig_room, height=360, show_legend=True, xtitle="Room", ytitle="Alarm Count")
    fig_room.update_layout(barmode='stack')
    st.plotly_chart(fig_room, width="stretch")

    section_header("Duration Statistics by Event Type", icon="📋")
    st.dataframe(
        duration_stats['by_type'].style.background_gradient(subset=['noise_pct'], cmap='YlOrRd'),
        width="stretch",
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — M&M / LEADERSHIP
# ──────────────────────────────────────────────────────────────────────────────

with tab4:
    page_header("M&M / Leadership", "Critical pattern detection — escalation chains, alert bursts, and pre-critical context")

    n_crit = int((events_df['severity'] == 3).sum())
    pre_crit_count = len(pre_crit_df) if not pre_crit_df.empty else 0

    a1, a2, a3, a4 = st.columns(4)
    with a1: kpi_card("Escalation Sequences", f"{len(esc_df)}", icon="⬆", severity="warning" if len(esc_df) > 0 else "good")
    with a2: kpi_card("Alert Bursts", f"{len(bursts_df)}", icon="💥", severity="warning" if len(bursts_df) > 0 else "good")
    with a3: kpi_card("Critical Events", f"{n_crit:,}", "★★★ severity", severity="critical", icon="🔴")
    with a4: kpi_card("Pre-Critical Context", f"{pre_crit_count:,}", "events before critical", severity="purple", icon="🔮")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    section_header("Escalation Sequences Detected", subtitle="Severity progression chains", icon="⬆")
    if not esc_df.empty:
        st.dataframe(
            esc_df[['room_id', 'escalation_time', 'from_severity', 'to_severity',
                    'trigger_event', 'prior_events_count', 'prior_event_types']],
            width="stretch",
        )

        esc_by_room = esc_df.groupby('room_id').size().reset_index(name='escalations')
        fig_esc = px.bar(
            esc_by_room.sort_values('escalations', ascending=False),
            x='room_id', y='escalations',
            color='escalations',
            color_continuous_scale=[[0, "#431407"], [0.5, "#f97316"], [1, "#fbbf24"]],
        )
        fig_esc.update_traces(marker_line_width=0)
        clinical_layout(fig_esc, height=350, xtitle="Room", ytitle="Escalation Count",
                        title_text="Escalation Sequences per Room")
        fig_esc.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_esc, width="stretch")
    else:
        empty_state("✅", "No escalation sequences detected in this dataset")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    section_header("Alert Bursts", subtitle="≥3 events within 5 minutes", icon="💥")
    if not bursts_df.empty:
        fig_burst = px.scatter(
            bursts_df, x='burst_start', y='room_id',
            size='event_count', color='severity_max',
            color_continuous_scale=[C['green'], C['amber'], C['red']],
            hover_data=['event_count', 'duration_min', 'event_types', 'has_critical'],
        )
        fig_burst.update_traces(marker=dict(line=dict(width=1, color=C['bg']), opacity=0.85))
        clinical_layout(fig_burst, height=420, xtitle="Time", ytitle="Room",
                        title_text="Burst Timeline by Room")
        fig_burst.update_layout(coloraxis_colorbar=dict(
            title=dict(text="Max Severity"), tickvals=[1, 2, 3], ticktext=["Low", "Med", "Crit"],
            len=0.4, thickness=12,
        ))
        st.plotly_chart(fig_burst, width="stretch")

        with st.expander("View burst details table"):
            st.dataframe(bursts_df, width="stretch")
    else:
        empty_state("✅", "No burst patterns detected")

    if not pre_crit_df.empty:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        section_header("Events Before Critical Alarms", subtitle="30-minute pre-critical window", icon="🔮")
        st.dataframe(
            pre_crit_df[['room_id', 'start_time', 'event_type', 'severity',
                         'critical_event_type', 'minutes_before_critical']],
            width="stretch",
        )

        pre_type = pre_crit_df.groupby('event_type').size().reset_index(name='count')
        fig_pre = px.bar(
            pre_type.sort_values('count', ascending=False),
            x='event_type', y='count',
            color='count',
            color_continuous_scale=[[0, "#3b0764"], [0.5, "#a78bfa"], [1, "#f472b6"]],
        )
        fig_pre.update_traces(marker_line_width=0)
        clinical_layout(fig_pre, height=350, xtitle="Event Type", ytitle="Frequency",
                        title_text="Most Common Precursors to Critical Alarms")
        fig_pre.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_pre, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — BI EXPORT
# ──────────────────────────────────────────────────────────────────────────────

with tab5:
    page_header("BI Export", "Download Power BI-ready datasets in star-schema format")

    st.markdown(f"""
    <div style="
      background:{_rgba(C['driscoll_gold'],0.04)};
      border:1px solid {_rgba(C['driscoll_gold'],0.14)};
      border-radius:12px;
      padding:20px 24px;
      margin-bottom:24px;
    ">
      <div style="font-family:var(--font-body);font-size:14px;font-weight:700;color:{C['driscoll_gold']};margin-bottom:10px;">
        Power BI Import Instructions
      </div>
      <div style="font-family:var(--font-body);font-size:13px;color:{C['t3']};line-height:1.9;">
        <span style="color:{C['t2']};font-weight:600;">1.</span> Open Power BI Desktop &nbsp;→&nbsp;
        <span style="color:{C['t2']};font-weight:600;">2.</span> Home → Get Data → CSV &nbsp;→&nbsp;
        <span style="color:{C['t2']};font-weight:600;">3.</span> Import <code style="background:{C['bg_card']};padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:11px;">fact_alerts.csv</code> as central fact table<br>
        <span style="color:{C['t2']};font-weight:600;">4.</span> Import <code style="background:{C['bg_card']};padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:11px;">dim_rooms.csv</code> and <code style="background:{C['bg_card']};padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:11px;">dim_event_types.csv</code> as dimensions &nbsp;→&nbsp;
        <span style="color:{C['t2']};font-weight:600;">5.</span> Join via <code style="background:{C['bg_card']};padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:11px;">room_id</code> and <code style="background:{C['bg_card']};padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:11px;">event_type</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

    output_dir = "/tmp/bi_output"
    if st.button("⚡  Generate BI Export Files", type="primary"):
        with st.spinner("Exporting datasets..."):
            outputs = export_bi_datasets(events_df, esc_df, bursts_df, output_dir)
            st.session_state.bi_exported = True
        st.success("Export complete!")

    if st.session_state.get('bi_exported'):
        section_header("Download Files", icon="📦")
        dl_cols = st.columns(3)
        for i, filename in enumerate(sorted(Path(output_dir).glob("*.csv"))):
            with dl_cols[i % 3]:
                with open(filename, 'rb') as f:
                    st.download_button(
                        label=f"⬇  {filename.name}",
                        data=f.read(),
                        file_name=filename.name,
                        mime='text/csv',
                        use_container_width=True,
                    )

    section_header("Star Schema", icon="⭐")
    st.markdown(f"""
    <div style="
      background:{C['bg_card']};
      border:1px solid {C['border']};
      border-radius:12px;
      padding:24px;
      font-family:var(--font-mono);
      font-size:12px;
      line-height:1.9;
      color:{C['t3']};
    ">
      <span style="color:{C['blue']};font-weight:700;">fact_alerts</span><br>
      &nbsp;&nbsp;├── <span style="color:{C['cyan']};">room_id</span> ──────────► <span style="color:{C['purple']};">dim_rooms</span><br>
      &nbsp;&nbsp;├── <span style="color:{C['cyan']};">event_type</span> ────────► <span style="color:{C['purple']};">dim_event_types</span><br>
      &nbsp;&nbsp;├── <span style="color:{C['cyan']};">date</span> ──────────────► <span style="color:{C['purple']};">dim_date</span> <span style="color:{C['t3']};font-size:10px;">(auto PBI)</span><br>
      &nbsp;&nbsp;└── <span style="color:{C['amber']};">measures</span><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── duration_sec<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── severity<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── is_noise<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── metric_value<br>
      <br>
      <span style="color:{C['green']};font-weight:700;">agg_room_day</span> &nbsp;&nbsp;&nbsp;Pre-aggregated room × day<br>
      <span style="color:{C['green']};font-weight:700;">agg_hourly</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pre-aggregated hourly heatmaps<br>
      <span style="color:{C['amber']};font-weight:700;">escalation_sequences</span> &nbsp;Pattern analysis<br>
      <span style="color:{C['red']};font-weight:700;">bursts</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Burst detection
    </div>
    """, unsafe_allow_html=True)