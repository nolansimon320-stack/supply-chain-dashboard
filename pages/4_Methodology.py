"""
Methodology — explains the data sources, models, and design decisions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Methodology · Supply Chain", page_icon="📚", layout="wide")

st.markdown("""
<style>
.method-card {
    background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
    padding:1.25rem 1.5rem;margin-bottom:1rem;
}
.formula {
    background:#1e293b;color:#e2e8f0;font-family:monospace;
    padding:0.75rem 1rem;border-radius:6px;font-size:0.88rem;
    margin:0.5rem 0;
}
h3 { color:#1e40af; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📚 Methodology")
st.markdown("How this dashboard collects data, builds analyses, and surfaces insights.")

# ── Data ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 1. Data Sources")

st.markdown("""
All data is pulled in real time from the **Federal Reserve Economic Data (FRED)** API,
maintained by the Federal Reserve Bank of St. Louis. FRED aggregates data from dozens of
primary sources including the U.S. Bureau of Labor Statistics and the IMF.
""")

data_table = pd.DataFrame([
    ["PPI: All Commodities",        "PPIACO",     "Monthly", "BLS",      "Upstream",   "Broadest measure of producer-price pressure"],
    ["PPI: Crude Materials",        "PPICRM",     "Monthly", "BLS",      "Upstream",   "Raw material prices entering production"],
    ["PPI: Finished Goods",         "PPIFGS",     "Monthly", "BLS",      "Downstream", "Cost pass-through to wholesale buyers"],
    ["Crude Oil — WTI",             "DCOILWTICO", "Daily→Mo","EIA",       "Upstream",   "Spot price; resampled to monthly mean"],
    ["Natural Gas — Henry Hub",     "DHHNGSP",    "Daily→Mo","EIA",       "Upstream",   "Key industrial & heating input"],
    ["Copper Price",                "PCOPPUSDM",  "Monthly", "IMF",      "Upstream",   "Leading industrial demand indicator"],
    ["Aluminum Price",              "PALUMUSDM",  "Monthly", "IMF",      "Upstream",   "Packaging & manufacturing input"],
    ["Wheat Price",                 "PWHEAMTUSDM","Monthly", "IMF",      "Upstream",   "Agricultural supply chain proxy"],
    ["CPI: All Items",              "CPIAUCSL",   "Monthly", "BLS",      "Downstream", "Consumer-level cost pass-through"],
    ["CPI: Transportation",         "CPITRNSL",   "Monthly", "BLS",      "Logistics",  "Freight & logistics cost proxy"],
], columns=["Series", "FRED ID", "Frequency", "Source", "Layer", "Notes"])
st.dataframe(data_table, use_container_width=True, hide_index=True)

st.markdown("""
**Baltic Dry Index (BDI):** The BDI is a widely used benchmark for global dry-bulk shipping costs.
It is not freely available via FRED. Future versions of this dashboard can integrate it via a
paid data provider (e.g., Nasdaq Data Link / Quandl). The CPI Transportation index serves as a
domestic logistics proxy in the interim.

**Caching:** All FRED API responses are cached for 1 hour using Streamlit's `@st.cache_data` decorator,
ensuring the dashboard reflects near-real-time data while minimising API calls.
""")

# ── Trend Analysis ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Demand & Cost Trend Analysis")

st.markdown('<div class="method-card">', unsafe_allow_html=True)
st.markdown("""
### Moving Averages
Two rolling means are computed for each series:

- **3-month MA** — short-term signal, responsive to recent shifts
- **12-month MA** — longer-term trend, filters seasonal noise

A 3-month MA crossing above the 12-month MA suggests accelerating price pressure (a *golden cross*
pattern in commodity analysis).
""")
st.markdown('<div class="formula">MA(n, t) = (1/n) × Σ X(t-i) for i = 0..n-1</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="method-card">', unsafe_allow_html=True)
st.markdown("""
### Ordinary Least Squares (OLS) Regression
A linear regression is fit to the last *N* months (configurable, default 24) of each series
to quantify the directional trend:
""")
st.markdown('<div class="formula">X(t) = β₀ + β₁·t + ε</div>', unsafe_allow_html=True)
st.markdown("""
- **β₁ (slope)** — monthly rate of change
- **Annualised rate** — β₁ × 12 / mean(X) × 100, expressed as %/year
- **R²** — goodness-of-fit; R² > 0.7 indicates a strong, consistent trend
- **p-value** — statistical significance of the slope (p < 0.05 = significant)
- **Direction** — classified as *Rising* (>+1%/yr), *Falling* (<−1%/yr), or *Flat*

The trend line is overlaid on each chart so analysts can visually confirm the fit.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ── Correlation ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3. Cost Driver Correlation Analysis")

st.markdown('<div class="method-card">', unsafe_allow_html=True)
st.markdown("""
### Pearson Correlation Matrix
The full correlation matrix between all series quantifies co-movement:
""")
st.markdown('<div class="formula">r(X, Y) = Cov(X, Y) / (σ_X × σ_Y)</div>', unsafe_allow_html=True)
st.markdown("""
Values range from −1 (perfect inverse relationship) to +1 (perfect positive relationship).
Supply chains typically show strong positive correlation between upstream inputs and downstream
outputs, with some lag.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="method-card">', unsafe_allow_html=True)
st.markdown("""
### Lagged Cross-Correlations
To quantify leading/lagging relationships, each upstream series is shifted back 0–6 months
before computing its correlation with the target:
""")
st.markdown('<div class="formula">r_lag(k) = Corr( downstream(t), upstream(t−k) )</div>', unsafe_allow_html=True)
st.markdown("""
A peak correlation at **lag k > 0** means upstream prices lead the target by *k* months —
a potentially actionable early-warning signal.

**Example:** If crude oil prices show peak correlation with PPI Finished Goods at lag 2,
a sharp crude oil move today suggests finished goods inflation in approximately 2 months.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ── Anomaly Detection ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 4. Anomaly Detection")

st.markdown('<div class="method-card">', unsafe_allow_html=True)
st.markdown("""
### Rolling Z-Score
For each series, a rolling mean and standard deviation are computed over a configurable
window (default 24 months):
""")
st.markdown('<div class="formula">Z(t) = ( X(t) − μ_roll(t) ) / σ_roll(t)</div>', unsafe_allow_html=True)
st.markdown("""
An observation is flagged when **|Z(t)| > threshold** (default 2.0σ). Two severity levels:

| Severity | Condition | Interpretation |
|---|---|---|
| **Moderate** | 2σ ≤ |Z| < 3σ | Unusual — warrants monitoring |
| **Extreme** | |Z| ≥ 3σ | Highly atypical — likely a disruption signal |

**Why rolling vs. global statistics?** Supply chain indices exhibit regime shifts
(e.g., COVID shock, 2022 energy crisis). A rolling window adapts to the recent baseline,
preventing old shocks from masking new ones — the same approach used by ops-risk teams
in practice.

Charts show the rolling ±Nσ band as a shaded confidence envelope, with anomalous
data points marked as diamonds.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 5. Limitations & Known Constraints")

st.markdown("""
- **Correlation ≠ causation.** Strong statistical relationships do not prove one series causes
  another to move. Operational context is required to interpret co-movement.
- **Monthly frequency.** All series are resampled or natively reported monthly. Intra-month
  shocks (e.g., a one-week port shutdown) will not appear until the following data release.
- **No BDI.** The Baltic Dry Index — the industry-standard shipping cost benchmark — requires
  a paid data subscription. The CPI Transportation proxy captures related dynamics but is
  not a direct substitute for ocean freight costs.
- **Revision risk.** BLS and IMF indices are subject to revision. Values shown reflect the
  latest FRED release at the time of last data fetch.
- **Spurious correlations.** With 10 series, random correlations are expected at the 5%
  significance level. The lagged correlation analysis should be interpreted alongside
  domain knowledge, not used mechanically.
""")

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 6. Technical Stack")

tech_table = pd.DataFrame([
    ["Streamlit",    "≥ 1.32", "Web app framework"],
    ["FRED API",     "—",      "Real-time economic data (fredapi Python client)"],
    ["pandas",       "≥ 2.2",  "Time-series data manipulation and resampling"],
    ["NumPy",        "≥ 1.24", "Numerical operations"],
    ["SciPy",        "≥ 1.11", "OLS regression (linregress), statistical utilities"],
    ["scikit-learn", "≥ 1.3",  "Supporting ML utilities"],
    ["Plotly",       "≥ 5.18", "Interactive, publication-quality charts"],
], columns=["Library", "Version", "Role"])
st.dataframe(tech_table, use_container_width=True, hide_index=True)

st.markdown("""
---
*Built with [Streamlit](https://streamlit.io) · Data from [FRED](https://fred.stlouisfed.org)*
""")
