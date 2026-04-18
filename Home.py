import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
.block-container { padding-top: 1.5rem; }
.section-header {
    font-size: 1.05rem; font-weight: 600; color: #374151;
    border-left: 3px solid #2563eb; padding-left: 0.6rem;
    margin: 1.5rem 0 0.75rem 0;
}
.alert-card {
    padding: 0.7rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
    font-size: 0.9rem; line-height: 1.5;
}
.alert-extreme { background:#fef2f2; border-left: 4px solid #dc2626; }
.alert-moderate { background:#fffbeb; border-left: 4px solid #d97706; }
.kpi-footer { font-size: 0.75rem; color: #9ca3af; margin-top: 0.25rem; }
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
    st.markdown("## ⚙️ Settings")
    api_key = render_sidebar_key_input()

    st.divider()
    st.markdown("**Data Range**")
    start_year = st.selectbox("Start Year", list(range(2010, 2024)), index=5, key="home_yr")
    start_date = f"{start_year}-01-01"

    st.divider()
    st.markdown("**Analysis Parameters**")
    trend_window = st.slider("Trend window (months)", 12, 48, 24, key="home_tw")
    anomaly_threshold = st.slider("Anomaly threshold (σ)", 1.5, 3.5, 2.0, 0.5, key="home_at")

    st.divider()
    st.caption(f"Data: FRED · Refreshes hourly")
    st.caption(f"Last render: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Pages: Overview · Cost Drivers · Anomaly Alerts · Methodology")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📦 Supply Chain Analytics")
st.markdown(
    "Real-time cost signals and disruption detection — "
    "powered by FRED (Federal Reserve Bank of St. Louis)"
)

if not api_key:
    st.info(
        "**Get started:** Enter your free FRED API key in the sidebar.  \n"
        "Register at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    st.divider()
    st.markdown("""
    ### What this dashboard tracks

    | Layer | Indicators |
    |---|---|
    | **Upstream inputs** | PPI All Commodities, PPI Crude Materials, Crude Oil (WTI), Natural Gas, Copper, Aluminum, Wheat |
    | **Logistics** | CPI: Transportation (freight cost proxy) |
    | **Downstream** | PPI Finished Goods, CPI All Items |

    ### Analytical methods
    - **Trend analysis** — 3-month and 12-month moving averages + OLS regression with R² scoring
    - **Cost driver correlation** — Pearson correlation matrix with lagged cross-correlations (0–6 months)
    - **Anomaly detection** — rolling Z-score flags when any series moves >{threshold}σ outside its 24-month window
    """.format(threshold="N"))
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading supply chain data from FRED…"):
    try:
        df, load_errors = load_all_data(api_key, start_date)
    except Exception as exc:
        st.error(f"Data load failed: {exc}")
        st.stop()

if load_errors:
    with st.expander(f"⚠️ {len(load_errors)} series failed to load"):
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
st.markdown('<p class="section-header">Key Metrics — Most Recent Month</p>', unsafe_allow_html=True)

kpi_keys = ["ppi_all", "crude_oil", "ppi_finished", "cpi"]
kpi_cols = st.columns(len(kpi_keys))

for col_widget, key in zip(kpi_cols, kpi_keys):
    if key not in df.columns:
        continue
    s = df[key].dropna()
    cur = s.iloc[-1]
    yoy = ((cur / s.iloc[-13]) - 1) * 100 if len(s) >= 13 else None
    delta_str = f"{yoy:+.1f}% YoY" if yoy is not None else "N/A"
    # For input costs, rising prices are bad (inverse)
    delta_color = "inverse" if key in ("crude_oil", "ppi_all", "ppi_crude") else "normal"
    with col_widget:
        st.metric(
            label=names.get(key, key),
            value=f"{cur:.1f}",
            delta=delta_str,
            delta_color=delta_color,
        )
        trend = trends.get(key)
        if trend:
            arrow = {"up": "↑", "down": "↓", "flat": "→"}[trend.direction]
            st.markdown(
                f'<p class="kpi-footer">{arrow} {trend.annualized_pct:+.1f}%/yr · '
                f'R²={trend.r_squared:.2f}</p>',
                unsafe_allow_html=True,
            )


# ── Alert banner ──────────────────────────────────────────────────────────────
if active_alerts:
    st.markdown(
        f'<p class="section-header">🚨 {len(active_alerts)} Active Anomaly Alert(s)</p>',
        unsafe_allow_html=True,
    )
    for alert in active_alerts[:6]:
        css_cls = "alert-extreme" if alert.severity == "extreme" else "alert-moderate"
        icon = "🔴" if alert.severity == "extreme" else "🟡"
        direction_word = "above" if alert.direction == "high" else "below"
        st.markdown(
            f'<div class="alert-card {css_cls}">'
            f'{icon} <strong>{alert.series_name}</strong> — '
            f'{abs(alert.z_score):.1f}σ {direction_word} its 2-year rolling average '
            f'({alert.date.strftime("%b %Y")}). '
            f'Value: <strong>{alert.value:.2f}</strong> vs mean {alert.rolling_mean:.2f} '
            f'(±{alert.rolling_std:.2f})'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption("→ See the **Anomaly Alerts** page for full charts and historical context.")


# ── Overview chart ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Normalized Index Comparison (Base = 100 at Start)</p>', unsafe_allow_html=True)

display_cols = [k for k in ["ppi_all", "ppi_finished", "crude_oil", "cpi", "cpi_transport"] if k in df.columns]
cat_map = {k: SERIES_CONFIG[k]["category"] for k in display_cols}

fig_multi = multi_series_chart(
    df_ma, display_cols, labels=shorts, category_map=cat_map,
    normalize=True, title="",
)
st.plotly_chart(fig_multi, use_container_width=True)


# ── YoY bar chart ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Year-over-Year Change by Series</p>', unsafe_allow_html=True)

all_cols = [k for k in df.columns if k in SERIES_CONFIG]
fig_yoy = yoy_bar_chart(df, all_cols, labels=names, title="")
st.plotly_chart(fig_yoy, use_container_width=True)


# ── Trend detail tabs ─────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Trend Detail</p>', unsafe_allow_html=True)

tab_keys = [k for k in display_cols if k in df.columns][:5]
tab_labels = [shorts.get(k, k) for k in tab_keys]
tabs = st.tabs(tab_labels)

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
            arrow = {"up": "↑ Rising", "down": "↓ Falling", "flat": "→ Flat"}[trend.direction]
            c1.metric("Direction", arrow)
            c2.metric("Annualized Rate", f"{trend.annualized_pct:+.1f}%/yr")
            c3.metric("R² (fit quality)", f"{trend.r_squared:.3f}")
            c4.metric("12-Month Avg", f"{trend.ma_12:.2f}")

        if cfg.get("description"):
            st.caption(cfg["description"])

st.markdown(
    f"---\nData sourced from [FRED](https://fred.stlouisfed.org). "
    f"Last data point: **{last_updated(df)}**."
)
