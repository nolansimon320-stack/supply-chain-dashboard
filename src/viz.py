from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#2563eb",
    "secondary":  "#7c3aed",
    "logistics":  "#0891b2",
    "success":    "#059669",
    "warning":    "#d97706",
    "danger":     "#dc2626",
    "neutral":    "#9ca3af",
    "grid":       "#e5e7eb",
    "bg":         "#ffffff",
    "bg2":        "#f8fafc",
}

CATEGORY_COLOR = {
    "upstream":   COLORS["primary"],
    "downstream": COLORS["secondary"],
    "logistics":  COLORS["logistics"],
}

SERIES_PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["logistics"],
    COLORS["warning"],
    COLORS["success"],
    "#ec4899",
    "#6366f1",
    "#14b8a6",
]

_BASE_LAYOUT = dict(
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1f2937"),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=COLORS["grid"],
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
    yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
)


def _base_layout(**overrides) -> dict:
    layout = dict(_BASE_LAYOUT)
    layout.update(overrides)
    return layout


# ── Individual trend chart ────────────────────────────────────────────────────
def trend_chart(
    df: pd.DataFrame,
    col: str,
    series_name: str,
    unit: str = "",
    add_regression: bool = True,
    window_months: int = 24,
) -> go.Figure:
    series = df[col].dropna()
    fig = go.Figure()

    # Raw series — light dashed backdrop
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=series_name,
        line=dict(color=COLORS["primary"], width=1.5, dash="dot"),
        opacity=0.45,
    ))

    # 3-month MA
    ma3 = df.get(f"{col}_ma3")
    if ma3 is not None:
        ma3 = ma3.dropna()
        fig.add_trace(go.Scatter(
            x=ma3.index, y=ma3.values,
            mode="lines", name="3-Month MA",
            line=dict(color=COLORS["primary"], width=2),
        ))

    # 12-month MA
    ma12 = df.get(f"{col}_ma12")
    if ma12 is not None:
        ma12 = ma12.dropna()
        fig.add_trace(go.Scatter(
            x=ma12.index, y=ma12.values,
            mode="lines", name="12-Month MA",
            line=dict(color=COLORS["secondary"], width=2.5),
        ))

    # OLS regression over last `window_months`
    if add_regression and len(series) >= 12:
        recent = series.tail(window_months)
        x_num = np.arange(len(recent), dtype=float)
        slope, intercept, r_value, *_ = stats.linregress(x_num, recent.to_numpy(dtype=float))
        trend_vals = slope * x_num + intercept
        fig.add_trace(go.Scatter(
            x=recent.index, y=trend_vals,
            mode="lines",
            name=f"OLS Trend ({window_months}mo, R²={r_value**2:.2f})",
            line=dict(color=COLORS["warning"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=dict(text=series_name, font=dict(size=15, color="#111827"), x=0),
        yaxis_title=unit,
        **_base_layout(),
    )
    return fig


# ── Multi-series comparison chart ────────────────────────────────────────────
def multi_series_chart(
    df: pd.DataFrame,
    cols: list[str],
    labels: dict[str, str] | None = None,
    category_map: dict[str, str] | None = None,
    normalize: bool = True,
    title: str = "Supply Chain Index Comparison",
) -> go.Figure:
    fig = go.Figure()
    labels = labels or {}
    category_map = category_map or {}

    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if normalize:
            base = float(series.iloc[0])
            if base != 0:
                series = series / base * 100

        cat = category_map.get(col, "")
        color = CATEGORY_COLOR.get(cat, SERIES_PALETTE[i % len(SERIES_PALETTE)])
        label = labels.get(col, col)

        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
        ))

    if normalize:
        fig.add_hline(y=100, line_dash="dash", line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0),
        yaxis_title="Normalized (start = 100)" if normalize else "",
        **_base_layout(),
    )
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────
def correlation_heatmap(
    corr: pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    display = [labels.get(c, c) if labels else c for c in corr.columns]
    vals = corr.values.round(2)

    fig = go.Figure(data=go.Heatmap(
        z=vals,
        x=display,
        y=display,
        colorscale=[[0, "#dc2626"], [0.5, "#ffffff"], [1, "#2563eb"]],
        zmid=0, zmin=-1, zmax=1,
        text=vals,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r", thickness=12),
    ))

    layout = _base_layout()
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout.pop("hovermode", None)
    fig.update_layout(
        title=dict(text="Correlation Matrix — Supply Chain Inputs vs Outputs", font=dict(size=15), x=0),
        height=480,
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        **{k: v for k, v in layout.items() if k not in ("xaxis", "yaxis", "hovermode")},
    )
    return fig


# ── Lagged correlation chart ──────────────────────────────────────────────────
def lagged_correlation_chart(
    lag_df: pd.DataFrame,
    target_name: str,
    source_labels: dict[str, str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    source_labels = source_labels or {}

    for i, source in enumerate(lag_df["source"].unique()):
        subset = lag_df[lag_df["source"] == source].sort_values("lag_months")
        label = source_labels.get(source, source)
        color = SERIES_PALETTE[i % len(SERIES_PALETTE)]

        fig.add_trace(go.Scatter(
            x=subset["lag_months"], y=subset["correlation"],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            hovertemplate=f"<b>{label}</b><br>Lag: %{{x}} mo<br>r = %{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(
        title=dict(text=f"Lagged Correlations → {target_name}", font=dict(size=15), x=0),
        xaxis_title="Lag (months) — positive = source leads target",
        yaxis_title="Pearson r",
        xaxis=dict(tickvals=list(range(7))),
        **_base_layout(),
    )
    return fig


# ── Anomaly detection chart ───────────────────────────────────────────────────
def anomaly_chart(
    df: pd.DataFrame,
    col: str,
    series_name: str,
    unit: str = "",
    window: int = 24,
    threshold: float = 2.0,
) -> go.Figure:
    series = df[col].dropna()
    min_p = window // 2
    roll_mean = series.rolling(window=window, min_periods=min_p).mean()
    roll_std = series.rolling(window=window, min_periods=min_p).std()
    upper = roll_mean + threshold * roll_std
    lower = roll_mean - threshold * roll_std

    z = (series - roll_mean) / roll_std.replace(0.0, np.nan)
    anomaly_mask = z.abs() > threshold
    anomalies = series[anomaly_mask]

    fig = go.Figure()

    # Confidence band
    combined_x = upper.index.tolist() + lower.index.tolist()[::-1]
    combined_y = upper.tolist() + lower.tolist()[::-1]
    fig.add_trace(go.Scatter(
        x=combined_x, y=combined_y,
        fill="toself",
        fillcolor="rgba(37,99,235,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±{threshold:.1f}σ Band",
        hoverinfo="skip",
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=roll_mean.index, y=roll_mean.values,
        mode="lines", name="Rolling Mean",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
    ))

    # Main series
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=series_name,
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Anomaly markers
    if len(anomalies) > 0:
        z_at = z[anomaly_mask]
        marker_colors = [COLORS["danger"] if v > 0 else COLORS["warning"] for v in z_at]
        fig.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies.values,
            mode="markers", name="Anomaly",
            marker=dict(color=marker_colors, size=12, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            text=[f"Z = {v:+.2f}" for v in z_at],
            hovertemplate="<b>Anomaly</b><br>%{x|%b %Y}<br>Value: %{y:.2f}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"Anomaly Detection — {series_name}", font=dict(size=15), x=0),
        yaxis_title=unit,
        **_base_layout(),
    )
    return fig


# ── YoY waterfall / bar chart ────────────────────────────────────────────────
def yoy_bar_chart(
    df: pd.DataFrame,
    cols: list[str],
    labels: dict[str, str] | None = None,
    title: str = "Year-over-Year % Change",
) -> go.Figure:
    labels = labels or {}
    records = []
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 13:
            continue
        yoy = (series.iloc[-1] / series.iloc[-13] - 1) * 100
        records.append({"label": labels.get(col, col), "yoy": yoy})

    if not records:
        return go.Figure()

    df_plot = pd.DataFrame(records).sort_values("yoy")
    colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in df_plot["yoy"]]

    fig = go.Figure(go.Bar(
        x=df_plot["yoy"], y=df_plot["label"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df_plot["yoy"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>YoY change: %{x:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1.5)

    layout = _base_layout()
    layout.pop("hovermode", None)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0),
        xaxis_title="YoY % Change",
        height=max(280, 40 * len(records) + 80),
        showlegend=False,
        **layout,
    )
    return fig
