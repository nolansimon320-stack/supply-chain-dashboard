import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Anomaly Alerts — Supply Chain Analytics", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2rem; }
.section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280; margin: 2rem 0 0.75rem 0;
}
.alert-row {
    padding: 0.6rem 0.9rem;
    border-left: 3px solid #dc2626;
    background: #fef2f2;
    border-radius: 0 4px 4px 0;
    font-size: 0.875rem;
    margin-bottom: 0.4rem;
    line-height: 1.6;
}
.alert-row.moderate { border-color: #ca8a04; background: #fefce8; }
.all-clear {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    padding: 0.9rem 1rem; border-radius: 4px; font-size: 0.9rem;
}
.stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: #111827; }
</style>
""", unsafe_allow_html=True)

from src.config import render_sidebar_key_input
from src.data_loader import load_all_data, SERIES_CONFIG, series_names, last_updated
from src.analysis import detect_anomalies
from src.viz import anomaly_chart

with st.sidebar:
    st.markdown("**Settings**")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start year", list(range(2010, 2024)), index=5)
    start_date = f"{start_year}-01-01"
    threshold = st.slider("Alert threshold (s)", 1.5, 4.0, 2.0, 0.25)
    roll_window = st.slider("Rolling window (months)", 12, 48, 24)
    lookback = st.slider("Lookback (months)", 1, 12, 6)

if not api_key:
    st.warning("Add your FRED API key in the sidebar to load data.")
    st.stop()

with st.spinner("Loading data..."):
    df, _ = load_all_data(api_key, start_date)

names = series_names()
all_alerts = detect_anomalies(
    df, window=roll_window, threshold=threshold,
    series_names=names, lookback_months=lookback,
)
active = [a for a in all_alerts if a.date >= df.index[-3]]

st.markdown("# Anomaly Alerts")
st.markdown(
    "Flags any series that falls outside its normal range based on the past "
    f"{roll_window} months of data. Threshold: {threshold}s. "
    f"Latest data: **{last_updated(df)}**"
)

# ── Summary stats ─────────────────────────────────────────────────────────────
extreme = [a for a in active if a.severity == "extreme"]
moderate = [a for a in active if a.severity == "moderate"]
series_flagged = len({a.series_key for a in active})

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f'<p class="stat-label">Active alerts</p><p class="stat-value">{len(active)}</p>', unsafe_allow_html=True)
with s2:
    st.markdown(f'<p class="stat-label">Extreme (&gt;3s)</p><p class="stat-value" style="color:#dc2626">{len(extreme)}</p>', unsafe_allow_html=True)
with s3:
    st.markdown(f'<p class="stat-label">Moderate (2–3s)</p><p class="stat-value" style="color:#ca8a04">{len(moderate)}</p>', unsafe_allow_html=True)
with s4:
    st.markdown(f'<p class="stat-label">Series flagged</p><p class="stat-value">{series_flagged} / {len(df.columns)}</p>', unsafe_allow_html=True)

st.divider()

# ── Alert list ────────────────────────────────────────────────────────────────
if not active:
    st.markdown(
        '<div class="all-clear">No alerts in the selected window. '
        'Adjust the threshold or lookback period to explore historical events.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<p class="section-label">Current alerts</p>', unsafe_allow_html=True)
    for alert in active:
        css = "alert-row" if alert.severity == "extreme" else "alert-row moderate"
        direction_word = "above" if alert.direction == "high" else "below"
        pct_dev = abs((alert.value - alert.rolling_mean) / alert.rolling_mean * 100) if alert.rolling_mean else 0
        st.markdown(
            f'<div class="{css}">'
            f'<strong>{alert.series_name}</strong> — '
            f'{alert.date.strftime("%B %Y")} — '
            f'{abs(alert.z_score):.1f}s {direction_word} 2-year mean — '
            f'Value: {alert.value:.2f} (mean {alert.rolling_mean:.2f}, '
            f'std {alert.rolling_std:.2f}, deviation {pct_dev:.1f}%)'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Full log ──────────────────────────────────────────────────────────────────
if all_alerts:
    with st.expander(f"Full alert log ({len(all_alerts)} events)"):
        rows = [
            {
                "Date": a.date.strftime("%Y-%m"),
                "Series": a.series_name,
                "Value": round(a.value, 2),
                "Z-score": round(a.z_score, 3),
                "Direction": a.direction.title(),
                "Severity": a.severity.title(),
                "Rolling mean": round(a.rolling_mean, 2),
                "Rolling std": round(a.rolling_std, 2),
            }
            for a in all_alerts
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Charts ────────────────────────────────────────────────────────────────────
flagged_keys = list({a.series_key for a in all_alerts})
unflagged_keys = [k for k in df.columns if k in SERIES_CONFIG and k not in flagged_keys]

def render_charts(keys: list[str]) -> None:
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

if flagged_keys:
    st.markdown(f'<p class="section-label">Flagged series ({len(flagged_keys)})</p>', unsafe_allow_html=True)
    render_charts(flagged_keys)

if unflagged_keys:
    with st.expander(f"No active alerts ({len(unflagged_keys)} series)"):
        render_charts(unflagged_keys)
