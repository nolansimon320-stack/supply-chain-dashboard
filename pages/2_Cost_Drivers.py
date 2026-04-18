"""
Cost Driver Analysis — correlation matrix and lagged cross-correlations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Cost Drivers · Supply Chain", page_icon="🔗", layout="wide")

from src.config import render_sidebar_key_input
from src.data_loader import load_all_data, SERIES_CONFIG, series_names, series_short_names, last_updated
from src.analysis import correlation_matrix, lagged_correlations
from src.viz import correlation_heatmap, lagged_correlation_chart

st.markdown("""
<style>
.section-header {
    font-size:1.05rem;font-weight:600;color:#374151;
    border-left:3px solid #7c3aed;padding-left:0.6rem;margin:1.5rem 0 0.75rem 0;
}
.insight-box {
    background:#f5f3ff;border-left:4px solid #7c3aed;
    padding:0.75rem 1rem;border-radius:0 6px 6px 0;
    font-size:0.9rem;margin-bottom:0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start Year", list(range(2010, 2024)), index=5, key="cd_yr")
    start_date = f"{start_year}-01-01"
    st.divider()
    max_lag = st.slider("Max lag months", 1, 12, 6, key="cd_lag")
    st.markdown("**Target series** (for lagged analysis)")
    available_targets = ["ppi_finished", "cpi", "ppi_all"]
    target_key = st.selectbox(
        "Target",
        available_targets,
        format_func=lambda k: SERIES_CONFIG.get(k, {}).get("name", k),
        key="cd_target",
    )

if not api_key:
    st.warning("Enter your FRED API key in the sidebar to load data.")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df, _ = load_all_data(api_key, start_date)

names = series_names()
shorts = series_short_names()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔗 Cost Driver Analysis")
st.markdown(
    "Which upstream inputs are most tightly correlated with downstream price changes? "
    f"Latest data: **{last_updated(df)}**"
)

# ── Correlation matrix ────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Full Correlation Matrix</p>', unsafe_allow_html=True)

available_cols = [k for k in SERIES_CONFIG if k in df.columns]
corr = correlation_matrix(df[available_cols])
fig_heatmap = correlation_heatmap(corr, labels=shorts)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("""
<div class="insight-box">
<strong>How to read this:</strong> Blue = strong positive correlation, Red = strong negative.
Values near ±1 indicate two series move closely together.
Values near 0 indicate little linear relationship. Upstream inputs typically cluster together
in the top-left block; downstream outputs in the bottom-right.
</div>
""", unsafe_allow_html=True)

# ── Top correlators table ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">Top Correlators with Selected Target</p>', unsafe_allow_html=True)

if target_key in corr.columns:
    target_corr = (
        corr[target_key]
        .drop(index=target_key, errors="ignore")
        .abs()
        .sort_values(ascending=False)
    )
    rows = []
    for k, abs_r in target_corr.items():
        raw_r = corr.loc[k, target_key]
        rows.append({
            "Series": names.get(k, k),
            "Category": SERIES_CONFIG.get(k, {}).get("category", "").title(),
            "Pearson r": f"{raw_r:+.3f}",
            "|r|": f"{abs_r:.3f}",
            "Relationship": "↑ Co-moves" if raw_r > 0 else "↓ Inverse",
            "Strength": "Strong" if abs_r > 0.7 else ("Moderate" if abs_r > 0.4 else "Weak"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Lagged correlations ───────────────────────────────────────────────────────
st.markdown(
    f'<p class="section-header">Lagged Correlations → {names.get(target_key, target_key)}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "At **lag = N**, the source series is shifted back N months. A peak at lag > 0 means "
    "the source **leads** the target — useful for early-warning signals."
)

upstream_keys = [k for k, v in SERIES_CONFIG.items() if v["category"] == "upstream" and k in df.columns]
lag_df = lagged_correlations(df, target_cols=[target_key], source_cols=upstream_keys, max_lag=max_lag)

if not lag_df.empty:
    fig_lag = lagged_correlation_chart(lag_df, target_name=names.get(target_key, target_key), source_labels=shorts)
    st.plotly_chart(fig_lag, use_container_width=True)

    # Best lag table
    best_lags = (
        lag_df.loc[lag_df.groupby("source")["correlation"].apply(lambda s: s.abs().idxmax())]
        .copy()
    )
    best_lags["series"] = best_lags["source"].map(names)
    best_lags["correlation"] = best_lags["correlation"].map(lambda x: f"{x:+.3f}")
    best_lags = best_lags.rename(columns={"lag_months": "Best Lag (mo)", "correlation": "r at Best Lag"})
    st.markdown("**Best lag per upstream series:**")
    st.dataframe(
        best_lags[["series", "Best Lag (mo)", "r at Best Lag"]].rename(columns={"series": "Series"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Scatter matrix ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Pairwise Scatter: Upstream vs. Target</p>', unsafe_allow_html=True)

import plotly.express as px

top_upstream = list(target_corr.head(3).index) if target_key in corr.columns else upstream_keys[:3]
scatter_cols = top_upstream + ([target_key] if target_key in df.columns else [])
scatter_df = df[scatter_cols].dropna()
scatter_df.columns = [shorts.get(c, c) for c in scatter_df.columns]

if len(scatter_df.columns) >= 2:
    fig_scatter = px.scatter_matrix(
        scatter_df,
        dimensions=scatter_df.columns.tolist(),
        title="Pairwise Scatter Matrix — Top Upstream Drivers vs Target",
        color_discrete_sequence=["#2563eb"],
        opacity=0.5,
    )
    fig_scatter.update_traces(marker=dict(size=4))
    fig_scatter.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=10),
        height=500,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
