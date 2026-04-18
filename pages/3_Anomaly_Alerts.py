"""
Anomaly Alerts — rolling Z-score disruption detection across all series.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Anomaly Alerts · Supply Chain", page_icon="🚨", layout="wide")

from src.config import render_sidebar_key_input
from src.data_loader import load_all_data, SERIES_CONFIG, series_names, last_updated
from src.analysis import detect_anomalies
from src.viz import anomaly_chart

st.markdown("""
<style>
.section-header {
    font-size:1.05rem;font-weight:600;color:#374151;
    border-left:3px solid #dc2626;padding-left:0.6rem;margin:1.5rem 0 0.75rem 0;
}
.alert-card { padding:0.75rem 1rem;border-radius:6px;margin-bottom:0.5rem;font-size:0.9rem;line-height:1.6; }
.alert-extreme { background:#fef2f2;border-left:4px solid #dc2626; }
.alert-moderate { background:#fffbeb;border-left:4px solid #d97706; }
.all-clear { background:#f0fdf4;border-left:4px solid #059669;padding:1rem;border-radius:6px; }
.stat-label { font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em; }
.stat-value { font-size:1.4rem;font-weight:700;color:#111827; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start Year", list(range(2010, 2024)), index=5, key="aa_yr")
    start_date = f"{start_year}-01-01"
    st.divider()
    threshold = st.slider("Alert threshold (σ)", 1.5, 4.0, 2.0, 0.25, key="aa_thr")
    roll_window = st.slider("Rolling window (months)", 12, 48, 24, key="aa_win")
    lookback = st.slider("Alert lookback (months)", 1, 12, 6, key="aa_lb")

if not api_key:
    st.warning("Enter your FRED API key in the sidebar to load data.")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df, _ = load_all_data(api_key, start_date)

names = series_names()
all_alerts = detect_anomalies(
    df, window=roll_window, threshold=threshold,
    series_names=names, lookback_months=lookback,
)
active = [a for a in all_alerts if a.date >= df.index[-3]]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🚨 Anomaly Alert System")
st.markdown(
    "Flags when any metric moves beyond a configurable σ-band from its rolling mean — "
    "the same principle used by supply chain analysts to identify disruption signals early. "
    f"Latest data: **{last_updated(df)}**"
)

# ── Summary KPIs ──────────────────────────────────────────────────────────────
extreme = [a for a in active if a.severity == "extreme"]
moderate = [a for a in active if a.severity == "moderate"]
series_flagged = len({a.series_key for a in active})

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f'<p class="stat-label">Active Alerts</p><p class="stat-value">{len(active)}</p>', unsafe_allow_html=True)
with s2:
    st.markdown(f'<p class="stat-label">Extreme (>3σ)</p><p class="stat-value" style="color:#dc2626">{len(extreme)}</p>', unsafe_allow_html=True)
with s3:
    st.markdown(f'<p class="stat-label">Moderate (2–3σ)</p><p class="stat-value" style="color:#d97706">{len(moderate)}</p>', unsafe_allow_html=True)
with s4:
    st.markdown(f'<p class="stat-label">Series Flagged</p><p class="stat-value">{series_flagged} / {len(df.columns)}</p>', unsafe_allow_html=True)

st.divider()

# ── Alert cards ───────────────────────────────────────────────────────────────
if not active:
    st.markdown(
        '<div class="all-clear">✅ <strong>All Clear</strong> — no series outside the alert band '
        'in the selected lookback window. Adjust the threshold or lookback to explore historical events.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p class="section-header">Alerts — Last {lookback} Months</p>',
        unsafe_allow_html=True,
    )
    for alert in active:
        css = "alert-extreme" if alert.severity == "extreme" else "alert-moderate"
        icon = "🔴" if alert.severity == "extreme" else "🟡"
        direction = "above" if alert.direction == "high" else "below"
        pct_dev = abs((alert.value - alert.rolling_mean) / alert.rolling_mean * 100) if alert.rolling_mean else 0
        st.markdown(
            f'<div class="alert-card {css}">'
            f'{icon} <strong>{alert.series_name}</strong> &nbsp;|&nbsp; '
            f'{alert.date.strftime("%B %Y")} &nbsp;|&nbsp; '
            f'Z = {alert.z_score:+.2f} ({abs(alert.z_score):.1f}σ {direction} 2-year mean) &nbsp;|&nbsp; '
            f'Value: <strong>{alert.value:.2f}</strong> '
            f'(mean {alert.rolling_mean:.2f}, ±1σ = {alert.rolling_std:.2f}, '
            f'{pct_dev:.1f}% deviation)'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Full alert log table ──────────────────────────────────────────────────────
if all_alerts:
    with st.expander(f"📋 Full alert log ({len(all_alerts)} events in lookback window)"):
        rows = [
            {
                "Date": a.date.strftime("%Y-%m"),
                "Series": a.series_name,
                "Value": round(a.value, 2),
                "Z-Score": round(a.z_score, 3),
                "Direction": a.direction.title(),
                "Severity": a.severity.title(),
                "Rolling Mean": round(a.rolling_mean, 2),
                "Rolling Std": round(a.rolling_std, 2),
            }
            for a in all_alerts
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Per-series anomaly charts ─────────────────────────────────────────────────
st.markdown('<p class="section-header">Anomaly Detection Charts — All Series</p>', unsafe_allow_html=True)

flagged_keys = list({a.series_key for a in all_alerts})
unflagged_keys = [k for k in df.columns if k in SERIES_CONFIG and k not in flagged_keys]

def render_anomaly_charts(keys: list[str], label: str) -> None:
    if not keys:
        return
    st.markdown(f"**{label}**")
    cols_per_row = 2
    for i in range(0, len(keys), cols_per_row):
        chunk = keys[i : i + cols_per_row]
        cols = st.columns(len(chunk))
        for col_widget, key in zip(cols, chunk):
            cfg = SERIES_CONFIG.get(key, {})
            with col_widget:
                fig = anomaly_chart(
                    df, key,
                    series_name=names.get(key, key),
                    unit=cfg.get("unit", ""),
                    window=roll_window,
                    threshold=threshold,
                )
                st.plotly_chart(fig, use_container_width=True)

render_anomaly_charts(flagged_keys, f"🔔 Flagged series ({len(flagged_keys)})")
with st.expander(f"Other series — no active alerts ({len(unflagged_keys)})"):
    render_anomaly_charts(unflagged_keys, "")
