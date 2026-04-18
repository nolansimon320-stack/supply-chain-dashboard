import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Cost Drivers — Supply Chain Analytics", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2rem; }
.section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280; margin: 2rem 0 0.75rem 0;
}
.note-box {
    background: #f9fafb; border: 1px solid #e5e7eb;
    padding: 0.75rem 1rem; border-radius: 4px;
    font-size: 0.875rem; line-height: 1.6;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

from src.config import render_sidebar_key_input
from src.data_loader import load_all_data, SERIES_CONFIG, series_names, series_short_names, last_updated
from src.analysis import correlation_matrix, lagged_correlations
from src.viz import correlation_heatmap, lagged_correlation_chart

with st.sidebar:
    st.markdown("**Settings**")
    api_key = render_sidebar_key_input()
    st.divider()
    start_year = st.selectbox("Start year", list(range(2010, 2024)), index=5)
    start_date = f"{start_year}-01-01"
    max_lag = st.slider("Max lag (months)", 1, 12, 6)
    st.markdown("**Target series**")
    available_targets = ["ppi_finished", "cpi", "ppi_all"]
    target_key = st.selectbox(
        "Target",
        available_targets,
        format_func=lambda k: SERIES_CONFIG.get(k, {}).get("name", k),
    )

if not api_key:
    st.warning("Add your FRED API key in the sidebar to load data.")
    st.stop()

with st.spinner("Loading data..."):
    df, _ = load_all_data(api_key, start_date)

names = series_names()
shorts = series_short_names()

st.markdown("# Cost Driver Analysis")
st.markdown(
    f"Which input costs are most closely correlated with the selected target series? "
    f"Latest data: **{last_updated(df)}**"
)

# ── Correlation matrix ────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Correlation matrix — all series</p>', unsafe_allow_html=True)

available_cols = [k for k in SERIES_CONFIG if k in df.columns]
corr = correlation_matrix(df[available_cols])
fig_heatmap = correlation_heatmap(corr, labels=shorts)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown(
    '<div class="note-box">'
    'Blue = positive correlation, red = negative, white = no linear relationship. '
    'A value near 1.0 means two series tend to rise and fall together. '
    'Upstream inputs typically show strong positive correlation with each other '
    'and with downstream outputs, though with a lag.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Top correlators ───────────────────────────────────────────────────────────
st.markdown(
    f'<p class="section-label">Correlations with {names.get(target_key, target_key)}</p>',
    unsafe_allow_html=True,
)

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
            "Direction": "Co-moves" if raw_r > 0 else "Inverse",
            "Strength": "Strong" if abs_r > 0.7 else ("Moderate" if abs_r > 0.4 else "Weak"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Lagged correlations ───────────────────────────────────────────────────────
st.markdown(
    f'<p class="section-label">Lagged correlations — upstream inputs vs {names.get(target_key, target_key)}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "At lag N, the source series is shifted back N months before computing the correlation. "
    "A peak at lag > 0 means that input cost tends to move before the target does — "
    "a potential early-warning signal."
)

upstream_keys = [k for k, v in SERIES_CONFIG.items() if v["category"] == "upstream" and k in df.columns]
lag_df = lagged_correlations(df, target_cols=[target_key], source_cols=upstream_keys, max_lag=max_lag)

if not lag_df.empty:
    fig_lag = lagged_correlation_chart(lag_df, target_name=names.get(target_key, target_key), source_labels=shorts)
    st.plotly_chart(fig_lag, use_container_width=True)

    best_lags = (
        lag_df.loc[lag_df.groupby("source")["correlation"].apply(lambda s: s.abs().idxmax())]
        .copy()
    )
    best_lags["Series"] = best_lags["source"].map(names)
    best_lags["r at best lag"] = best_lags["correlation"].map(lambda x: f"{x:+.3f}")
    st.markdown("**Best lag per series**")
    st.dataframe(
        best_lags[["Series", "lag_months", "r at best lag"]].rename(columns={"lag_months": "Best lag (months)"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Scatter matrix ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Pairwise scatter — top 3 inputs vs target</p>', unsafe_allow_html=True)

import plotly.express as px

top_upstream = list(target_corr.head(3).index) if target_key in corr.columns else upstream_keys[:3]
scatter_cols = top_upstream + ([target_key] if target_key in df.columns else [])
scatter_df = df[scatter_cols].dropna()
scatter_df.columns = [shorts.get(c, c) for c in scatter_df.columns]

if len(scatter_df.columns) >= 2:
    fig_scatter = px.scatter_matrix(
        scatter_df,
        dimensions=scatter_df.columns.tolist(),
        color_discrete_sequence=["#1d4ed8"],
        opacity=0.45,
    )
    fig_scatter.update_traces(marker=dict(size=4))
    fig_scatter.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, system-ui, sans-serif", size=10),
        height=480,
        title=dict(text="Pairwise scatter — top upstream drivers vs target", font=dict(size=14), x=0),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
