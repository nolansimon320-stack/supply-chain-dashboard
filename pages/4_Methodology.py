import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Methodology — Supply Chain Analytics", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2rem; }
.method-block {
    background: #f9fafb; border: 1px solid #e5e7eb;
    padding: 1.25rem 1.5rem; border-radius: 4px; margin-bottom: 1rem;
}
.code-block {
    background: #1e293b; color: #e2e8f0;
    font-family: 'Menlo', 'Consolas', monospace;
    padding: 0.75rem 1rem; border-radius: 4px;
    font-size: 0.875rem; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Methodology")
st.markdown("Data sources, analytical methods, and known limitations.")

# ── Data sources ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Data sources")
st.markdown(
    "All data is fetched from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) API, "
    "maintained by the Federal Reserve Bank of St. Louis. "
    "API responses are cached for one hour using Streamlit's `@st.cache_data` decorator."
)

data_table = pd.DataFrame([
    ["PPI: All Commodities",        "PPIACO",     "Monthly",   "BLS",  "Upstream",   "Broadest measure of producer price change"],
    ["PPI: Crude Materials",        "PPICRM",     "Monthly",   "BLS",  "Upstream",   "Raw materials entering the production process"],
    ["PPI: Finished Goods",         "PPIFGS",     "Monthly",   "BLS",  "Downstream", "Prices at the wholesale level for finished products"],
    ["Crude Oil (WTI)",             "DCOILWTICO", "Daily→Mo",  "EIA",  "Upstream",   "Daily spot price, resampled to monthly mean"],
    ["Natural Gas (Henry Hub)",     "DHHNGSP",    "Daily→Mo",  "EIA",  "Upstream",   "Daily spot price, resampled to monthly mean"],
    ["Copper",                      "PCOPPUSDM",  "Monthly",   "IMF",  "Upstream",   "Used as a proxy for industrial demand"],
    ["Aluminum",                    "PALUMUSDM",  "Monthly",   "IMF",  "Upstream",   "Key input for packaging and manufacturing"],
    ["Wheat",                       "PWHEAMTUSDM","Monthly",   "IMF",  "Upstream",   "Agricultural supply chain proxy"],
    ["CPI: All Items",              "CPIAUCSL",   "Monthly",   "BLS",  "Downstream", "Consumer-level measure of cost pass-through"],
    ["CPI: Transportation",         "CPITRNSL",   "Monthly",   "BLS",  "Logistics",  "Proxy for freight and logistics costs"],
], columns=["Series", "FRED ID", "Frequency", "Source", "Layer", "Notes"])
st.dataframe(data_table, use_container_width=True, hide_index=True)

st.markdown(
    "**Baltic Dry Index (BDI):** The BDI is the standard benchmark for ocean freight costs "
    "and is not freely available through FRED. The CPI Transportation index is used as a "
    "domestic logistics proxy. BDI integration would require a paid data subscription "
    "(e.g., Nasdaq Data Link)."
)

# ── Trend analysis ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Trend analysis")

st.markdown('<div class="method-block">', unsafe_allow_html=True)
st.markdown("**Moving averages**")
st.markdown(
    "A 3-month and 12-month rolling mean are computed for each series. "
    "The 3-month MA responds quickly to recent changes; the 12-month MA filters out seasonal noise. "
    "When the 3-month crosses above the 12-month, it indicates accelerating price pressure."
)
st.markdown('<div class="code-block">MA(n, t) = (1/n) × sum of X(t-i) for i = 0 to n-1</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="method-block">', unsafe_allow_html=True)
st.markdown("**OLS regression**")
st.markdown(
    "A linear regression is fit to the most recent N months (configurable, default 24) to quantify direction and rate of change."
)
st.markdown('<div class="code-block">X(t) = b0 + b1 * t + error</div>', unsafe_allow_html=True)
st.markdown("""
- **Slope (b1)** — monthly rate of change
- **Annualized rate** — b1 × 12 / mean(X) × 100, expressed as %/year
- **R²** — how well the line fits the data; R² above 0.7 indicates a consistent trend
- **p-value** — statistical significance of the slope; below 0.05 is considered significant
- **Direction** — classified as Rising (>+1%/yr), Falling (<−1%/yr), or Flat
""")
st.markdown('</div>', unsafe_allow_html=True)

# ── Correlation analysis ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Cost driver correlation")

st.markdown('<div class="method-block">', unsafe_allow_html=True)
st.markdown("**Pearson correlation matrix**")
st.markdown(
    "Measures the linear relationship between all pairs of series. "
    "Values range from −1 (perfect inverse) to +1 (perfect positive)."
)
st.markdown('<div class="code-block">r(X, Y) = Cov(X, Y) / (std(X) × std(Y))</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="method-block">', unsafe_allow_html=True)
st.markdown("**Lagged correlations**")
st.markdown(
    "Each upstream series is shifted back 0 to 6 months before computing its correlation with the target. "
    "A peak at lag k means the upstream input tends to move k months before the target — "
    "a potential leading indicator."
)
st.markdown('<div class="code-block">r_lag(k) = Corr( downstream(t), upstream(t - k) )</div>', unsafe_allow_html=True)
st.markdown(
    "Example: if crude oil correlates most strongly with PPI Finished Goods at lag 2, "
    "an energy price spike today suggests finished goods inflation in roughly 2 months."
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Anomaly detection ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Anomaly detection")

st.markdown('<div class="method-block">', unsafe_allow_html=True)
st.markdown("**Rolling Z-score**")
st.markdown(
    "A rolling mean and standard deviation are computed over the past N months (default 24). "
    "A reading is flagged when its Z-score exceeds the set threshold (default 2.0)."
)
st.markdown('<div class="code-block">Z(t) = ( X(t) - rolling_mean(t) ) / rolling_std(t)</div>', unsafe_allow_html=True)

severity_table = pd.DataFrame([
    ["Moderate", "2s to 3s", "Unusual reading — worth monitoring"],
    ["Extreme",  ">3s",      "Highly atypical — likely a disruption signal"],
], columns=["Level", "Condition", "Interpretation"])
st.dataframe(severity_table, use_container_width=True, hide_index=True)

st.markdown(
    "A rolling baseline is used instead of a global one because supply chain indices shift over time "
    "(e.g., post-COVID normalization, 2022 energy shock). A rolling window adapts to the current regime "
    "rather than treating old shocks as the reference point."
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Limitations")

st.markdown("""
- **Correlation is not causation.** Strong statistical relationships do not confirm that one series drives another. Operational context is required.
- **Monthly frequency.** All data is monthly. Intra-month disruptions will not appear until the next data release.
- **No Baltic Dry Index.** The BDI requires a paid data subscription and is not included. CPI Transportation is used as a proxy.
- **Revision risk.** BLS and IMF figures are subject to retroactive revision. Values shown reflect the latest FRED release at the time of fetch.
- **Spurious correlations.** With 10 series, some correlations will appear by chance. Lagged correlation results should be interpreted alongside domain knowledge.
""")

# ── Stack ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Technical stack")

stack_table = pd.DataFrame([
    ["Streamlit",    "1.32+", "Web app framework and deployment"],
    ["FRED API",     "—",     "Real-time economic data (via fredapi Python client)"],
    ["pandas",       "2.2+",  "Time-series manipulation and resampling"],
    ["NumPy",        "1.24+", "Numerical operations"],
    ["SciPy",        "1.11+", "OLS regression, statistical functions"],
    ["Plotly",       "5.18+", "Interactive charts"],
], columns=["Library", "Version", "Role"])
st.dataframe(stack_table, use_container_width=True, hide_index=True)

st.markdown("---\nData from [FRED](https://fred.stlouisfed.org). Built with [Streamlit](https://streamlit.io).")
