import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 2rem; }

h1 { font-size: 1.75rem; font-weight: 700; color: #111827; letter-spacing: -0.02em; }
h2 { font-size: 1.1rem; font-weight: 600; color: #374151; margin-top: 2rem; }

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 2rem 0 0.75rem 0;
}

.kpi-note { font-size: 0.75rem; color: #9ca3af; margin-top: 0.2rem; }

.alert-row {
    padding: 0.6rem 0.9rem;
    border-left: 3px solid #dc2626;
    background: #fef2f2;
    border-radius: 0 4px 4px 0;
    font-size: 0.875rem;
    margin-bottom: 0.4rem;
    line-height: 1.5;
}
.alert-row.moderate {
    border-color: #ca8a04;
    background: #fefce8;
}

[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

from src.config import render_sidebar_key_input
from src.data_loader import (
    load_all_data, SERIES_CONFIG, series_names, series_short_names, last_updated,
)
from src.analysis import add_moving_averages, compute_all_trends, detect_anomalies
from src.viz import multi_series_chart, trend_chart, yoy_bar_chart, CATEGORY_COLOR


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Settings**")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start year", list(range(2010, 2024)), index=5)
    start_date = f"{start_year}-01-01"
    trend_window = st.slider("Trend window (months)", 12, 48, 24)
    anomaly_threshold = st.slider("Anomaly threshold (s)", 1.5, 3.5, 2.0, 0.5)
    st.divider()
    st.caption(f"Data from FRED — refreshes hourly")
    st.caption(f"Rendered {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Supply Chain Analytics")
st.markdown(
    "Commodity prices, logistics costs, and downstream inflation — "
    "data from the Federal Reserve Economic Data (FRED) API."
)

if not api_key:
    st.info(
        "Add your FRED API key in the sidebar to load data.  \n"
        "Free registration at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    st.divider()
    st.markdown("**What this tracks**")
    st.markdown("""
| Layer | Series |
|---|---|
| Upstream inputs | PPI All Commodities, PPI Crude Materials, Crude Oil (WTI), Natural Gas, Copper, Aluminum, Wheat |
| Logistics | CPI: Transportation |
| Downstream | PPI Finished Goods, CPI All Items |
""")
    st.markdown("**Methods**")
    st.markdown("""
- 3-month and 12-month moving averages per series
- OLS trend regression with R² and p-value
- Pearson correlation matrix with lagged cross-correlations (0–6 months)
- Rolling Z-score anomaly detection
""")
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data from FRED..."):
    try:
        df, load_errors = load_all_data(api_key, start_date)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

if load_errors:
    with st.expander(f"{len(load_errors)} series failed to load"):
        for err in load_errors:
            st.caption(err)

df_ma = add_moving_averages(df)
trends = compute_all_trends(df, window_months=trend_window)
names = series_names()
shorts = series_short_names()
alerts = detect_anomalies(
    df, window=24, threshold=anomaly_threshold,
    series_names=names, lookback_months=6,
)
active_alerts = [a for a in alerts if a.date >= df.index[-3]]


# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Key metrics — most recent month</p>', unsafe_allow_html=True)

kpi_keys = ["ppi_all", "crude_oil", "ppi_finished", "cpi"]
kpi_cols = st.columns(len(kpi_keys))

for col_widget, key in zip(kpi_cols, kpi_keys):
    if key not in df.columns:
        continue
    s = df[key].dropna()
    cur = s.iloc[-1]
    yoy = ((cur / s.iloc[-13]) - 1) * 100 if len(s) >= 13 else None
    delta_str = f"{yoy:+.1f}% YoY" if yoy is not None else "N/A"
    delta_color = "inverse" if key in ("crude_oil", "ppi_all", "ppi_crude") else "normal"
    with col_widget:
        st.metric(label=names.get(key, key), value=f"{cur:.1f}", delta=delta_str, delta_color=delta_color)
        trend = trends.get(key)
        if trend:
            arrow = {"up": "up", "down": "down", "flat": "flat"}[trend.direction]
            st.markdown(
                f'<p class="kpi-note">{arrow} {trend.annualized_pct:+.1f}%/yr — R²={trend.r_squared:.2f}</p>',
                unsafe_allow_html=True,
            )


# ── Alert banner ──────────────────────────────────────────────────────────────
if active_alerts:
    st.markdown(
        f'<p class="section-label">{len(active_alerts)} alert(s) — last 3 months</p>',
        unsafe_allow_html=True,
    )
    for alert in active_alerts[:6]:
        css = "alert-row" if alert.severity == "extreme" else "alert-row moderate"
        direction_word = "above" if alert.direction == "high" else "below"
        st.markdown(
            f'<div class="{css}">'
            f'<strong>{alert.series_name}</strong> — '
            f'{abs(alert.z_score):.1f}s {direction_word} 2-year average ({alert.date.strftime("%b %Y")}). '
            f'Value: {alert.value:.2f}, mean: {alert.rolling_mean:.2f}'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption("See the Anomaly Alerts page for full charts.")


# ── Overview chart ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">All series indexed to 100 at start date</p>', unsafe_allow_html=True)

display_cols = [k for k in ["ppi_all", "ppi_finished", "crude_oil", "cpi", "cpi_transport"] if k in df.columns]
cat_map = {k: SERIES_CONFIG[k]["category"] for k in display_cols}

fig_multi = multi_series_chart(df_ma, display_cols, labels=shorts, category_map=cat_map, normalize=True)
st.plotly_chart(fig_multi, use_container_width=True)


# ── YoY bar chart ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Year-over-year change by series</p>', unsafe_allow_html=True)

all_cols = [k for k in df.columns if k in SERIES_CONFIG]
fig_yoy = yoy_bar_chart(df, all_cols, labels=names)
st.plotly_chart(fig_yoy, use_container_width=True)


# ── Trend detail tabs ─────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Trend detail</p>', unsafe_allow_html=True)

tab_keys = [k for k in display_cols if k in df.columns][:5]
tabs = st.tabs([shorts.get(k, k) for k in tab_keys])

for tab, key in zip(tabs, tab_keys):
    with tab:
        cfg = SERIES_CONFIG.get(key, {})
        fig = trend_chart(
            df_ma, key,
            series_name=names.get(key, key),
            unit=cfg.get("unit", ""),
            add_regression=True,
            window_months=trend_window,
        )
        st.plotly_chart(fig, use_container_width=True)

        trend = trends.get(key)
        if trend:
            c1, c2, c3, c4 = st.columns(4)
            direction_label = {"up": "Rising", "down": "Falling", "flat": "Flat"}[trend.direction]
            c1.metric("Direction", direction_label)
            c2.metric("Annualized rate", f"{trend.annualized_pct:+.1f}%/yr")
            c3.metric("R²", f"{trend.r_squared:.3f}")
            c4.metric("12-month average", f"{trend.ma_12:.2f}")

        if cfg.get("description"):
            st.caption(cfg["description"])

st.markdown(
    f"---\nData: [FRED](https://fred.stlouisfed.org) — last data point: **{last_updated(df)}**"
)
