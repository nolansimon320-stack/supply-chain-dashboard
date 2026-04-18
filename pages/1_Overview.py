"""
Overview page — full trend analysis for every loaded series.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(page_title="Overview · Supply Chain", page_icon="📊", layout="wide")

from src.config import render_sidebar_key_input
from src.data_loader import load_all_data, SERIES_CONFIG, series_names, last_updated
from src.analysis import add_moving_averages, compute_all_trends, yoy_pct
from src.viz import trend_chart, multi_series_chart, SERIES_PALETTE

st.markdown("""
<style>
.section-header {
    font-size:1.05rem;font-weight:600;color:#374151;
    border-left:3px solid #2563eb;padding-left:0.6rem;margin:1.5rem 0 0.75rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start Year", list(range(2010, 2024)), index=5, key="ov_yr")
    start_date = f"{start_year}-01-01"
    trend_window = st.slider("Trend window (months)", 12, 48, 24, key="ov_tw")
    st.divider()
    category_filter = st.multiselect(
        "Filter by category",
        ["upstream", "downstream", "logistics"],
        default=["upstream", "downstream", "logistics"],
    )

if not api_key:
    st.warning("Enter your FRED API key in the sidebar to load data.")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df, errors = load_all_data(api_key, start_date)

df_ma = add_moving_averages(df)
trends = compute_all_trends(df, window_months=trend_window)
names = series_names()

filtered_keys = [
    k for k, v in SERIES_CONFIG.items()
    if k in df.columns and v["category"] in category_filter
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Overview — Trend Analysis")
st.markdown(
    f"Moving averages and OLS trend lines for all tracked series. "
    f"Latest data: **{last_updated(df)}**"
)

# ── Upstream vs downstream comparison ────────────────────────────────────────
st.markdown('<p class="section-header">Upstream Inputs vs Downstream Costs (Normalized)</p>', unsafe_allow_html=True)

upstream = [k for k in filtered_keys if SERIES_CONFIG[k]["category"] == "upstream"]
downstream = [k for k in filtered_keys if SERIES_CONFIG[k]["category"] in ("downstream", "logistics")]

col_l, col_r = st.columns(2)
with col_l:
    fig = multi_series_chart(df_ma, upstream[:4], labels=names, normalize=True, title="Upstream Input Costs")
    st.plotly_chart(fig, use_container_width=True)
with col_r:
    fig = multi_series_chart(df_ma, downstream, labels=names, normalize=True, title="Downstream Cost Pass-Through")
    st.plotly_chart(fig, use_container_width=True)

# ── Trend summary table ───────────────────────────────────────────────────────
st.markdown('<p class="section-header">Trend Summary Table</p>', unsafe_allow_html=True)

rows = []
for key in filtered_keys:
    t = trends.get(key)
    if not t:
        continue
    cfg = SERIES_CONFIG[key]
    arrow = {"up": "↑ Rising", "down": "↓ Falling", "flat": "→ Flat"}[t.direction]
    rows.append({
        "Series": names[key],
        "Category": cfg["category"].title(),
        "Current": f"{t.current_value:.2f}",
        "3-Mo MA": f"{t.ma_3:.2f}",
        "12-Mo MA": f"{t.ma_12:.2f}",
        "Trend": arrow,
        "Rate (ann.)": f"{t.annualized_pct:+.1f}%/yr",
        "R²": f"{t.r_squared:.3f}",
        "p-value": f"{t.p_value:.4f}",
    })

if rows:
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Per-series detail charts ──────────────────────────────────────────────────
st.markdown('<p class="section-header">Individual Series — Trend Charts</p>', unsafe_allow_html=True)

cols_per_row = 2
for i in range(0, len(filtered_keys), cols_per_row):
    row_keys = filtered_keys[i : i + cols_per_row]
    cols = st.columns(len(row_keys))
    for col_widget, key in zip(cols, row_keys):
        cfg = SERIES_CONFIG[key]
        with col_widget:
            fig = trend_chart(
                df_ma, key,
                series_name=names[key],
                unit=cfg.get("unit", ""),
                add_regression=True,
                window_months=trend_window,
            )
            st.plotly_chart(fig, use_container_width=True)
