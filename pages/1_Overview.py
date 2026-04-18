import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Overview — Supply Chain Analytics", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2rem; }
.section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280; margin: 2rem 0 0.75rem 0;
}
</style>
""", unsafe_allow_html=True)

from src.config import render_sidebar_key_input
from src.data_loader import load_all_data, SERIES_CONFIG, series_names, last_updated
from src.analysis import add_moving_averages, compute_all_trends
from src.viz import trend_chart, multi_series_chart

with st.sidebar:
    st.markdown("**Settings**")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start year", list(range(2010, 2024)), index=5)
    start_date = f"{start_year}-01-01"
    trend_window = st.slider("Trend window (months)", 12, 48, 24)
    category_filter = st.multiselect(
        "Category",
        ["upstream", "downstream", "logistics"],
        default=["upstream", "downstream", "logistics"],
    )

if not api_key:
    st.warning("Add your FRED API key in the sidebar to load data.")
    st.stop()

with st.spinner("Loading data..."):
    df, _ = load_all_data(api_key, start_date)

df_ma = add_moving_averages(df)
trends = compute_all_trends(df, window_months=trend_window)
names = series_names()

filtered_keys = [
    k for k, v in SERIES_CONFIG.items()
    if k in df.columns and v["category"] in category_filter
]

st.markdown("# Trend Overview")
st.markdown(f"Moving averages and regression lines for all tracked series. Latest data: **{last_updated(df)}**")

# ── Upstream vs downstream side by side ──────────────────────────────────────
st.markdown('<p class="section-label">Upstream inputs vs downstream costs — indexed to 100</p>', unsafe_allow_html=True)

upstream = [k for k in filtered_keys if SERIES_CONFIG[k]["category"] == "upstream"]
downstream = [k for k in filtered_keys if SERIES_CONFIG[k]["category"] in ("downstream", "logistics")]

col_l, col_r = st.columns(2)
with col_l:
    fig = multi_series_chart(df_ma, upstream[:4], labels=names, normalize=True, title="Upstream inputs")
    st.plotly_chart(fig, use_container_width=True)
with col_r:
    fig = multi_series_chart(df_ma, downstream, labels=names, normalize=True, title="Downstream costs")
    st.plotly_chart(fig, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Trend summary</p>', unsafe_allow_html=True)

rows = []
for key in filtered_keys:
    t = trends.get(key)
    if not t:
        continue
    cfg = SERIES_CONFIG[key]
    direction_label = {"up": "Rising", "down": "Falling", "flat": "Flat"}[t.direction]
    rows.append({
        "Series": names[key],
        "Category": cfg["category"].title(),
        "Current": f"{t.current_value:.2f}",
        "3-mo avg": f"{t.ma_3:.2f}",
        "12-mo avg": f"{t.ma_12:.2f}",
        "Trend": direction_label,
        "Rate (ann.)": f"{t.annualized_pct:+.1f}%/yr",
        "R²": f"{t.r_squared:.3f}",
        "p-value": f"{t.p_value:.4f}",
    })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Individual charts ─────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Individual series</p>', unsafe_allow_html=True)

cols_per_row = 2
for i in range(0, len(filtered_keys), cols_per_row):
    chunk = filtered_keys[i : i + cols_per_row]
    cols = st.columns(len(chunk))
    for col_widget, key in zip(cols, chunk):
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
