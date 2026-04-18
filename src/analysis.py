from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class TrendResult:
    series_key: str
    current_value: float
    ma_3: float
    ma_12: float
    slope_monthly: float
    annualized_pct: float
    r_squared: float
    p_value: float
    direction: str       # "up" | "down" | "flat"
    window_months: int


@dataclass
class AnomalyAlert:
    series_key: str
    series_name: str
    date: pd.Timestamp
    value: float
    z_score: float
    rolling_mean: float
    rolling_std: float
    direction: str    # "high" | "low"
    severity: str     # "moderate" (2–3 σ) | "extreme" (> 3 σ)


# ── Moving averages ───────────────────────────────────────────────────────────
def add_moving_averages(df: pd.DataFrame, windows: list[int] = (3, 6, 12)) -> pd.DataFrame:
    result = df.copy()
    for col in df.columns:
        for w in windows:
            min_p = max(1, w // 2)
            result[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=min_p).mean()
    return result


# ── Trend regression ──────────────────────────────────────────────────────────
def compute_trend(series: pd.Series, window_months: int = 24) -> TrendResult | None:
    clean = series.dropna()
    if len(clean) < 6:
        return None

    recent = clean.tail(window_months)
    x = np.arange(len(recent), dtype=float)
    y = recent.to_numpy(dtype=float)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    mean_val = float(np.mean(y))
    annualized_pct = (slope * 12 / mean_val * 100) if mean_val != 0 else 0.0
    r_squared = r_value ** 2

    if abs(annualized_pct) < 1.0:
        direction = "flat"
    elif annualized_pct > 0:
        direction = "up"
    else:
        direction = "down"

    ma_3 = float(clean.tail(3).mean()) if len(clean) >= 3 else float(clean.iloc[-1])
    ma_12 = float(clean.tail(12).mean()) if len(clean) >= 12 else float(clean.mean())

    return TrendResult(
        series_key=str(series.name),
        current_value=float(clean.iloc[-1]),
        ma_3=ma_3,
        ma_12=ma_12,
        slope_monthly=float(slope),
        annualized_pct=float(annualized_pct),
        r_squared=float(r_squared),
        p_value=float(p_value),
        direction=direction,
        window_months=window_months,
    )


def compute_all_trends(df: pd.DataFrame, window_months: int = 24) -> dict[str, TrendResult]:
    results = {}
    for col in df.columns:
        result = compute_trend(df[col].rename(col), window_months)
        if result is not None:
            results[col] = result
    return results


# ── Correlation analysis ──────────────────────────────────────────────────────
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")


def lagged_correlations(
    df: pd.DataFrame,
    target_cols: list[str],
    source_cols: list[str],
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    For each (source, target) pair, compute Pearson correlation at lags 0..max_lag.
    A positive lag means the source is shifted back in time (i.e., source leads target).
    """
    records = []
    for target in target_cols:
        if target not in df.columns:
            continue
        for source in source_cols:
            if source not in df.columns or source == target:
                continue
            for lag in range(0, max_lag + 1):
                r = df[target].corr(df[source].shift(lag))
                records.append(
                    {"target": target, "source": source, "lag_months": lag, "correlation": r}
                )
    return pd.DataFrame(records)


# ── Anomaly detection ─────────────────────────────────────────────────────────
def detect_anomalies(
    df: pd.DataFrame,
    window: int = 24,
    threshold: float = 2.0,
    series_names: dict[str, str] | None = None,
    lookback_months: int = 6,
) -> list[AnomalyAlert]:
    """
    Rolling Z-score anomaly detection.
    Flags observations where |z| > threshold within the last `lookback_months` months.
    """
    alerts: list[AnomalyAlert] = []
    series_names = series_names or {}

    for col in df.columns:
        series = df[col].dropna()
        if len(series) < window:
            continue

        min_p = window // 2
        roll_mean = series.rolling(window=window, min_periods=min_p).mean()
        roll_std = series.rolling(window=window, min_periods=min_p).std()
        z = (series - roll_mean) / roll_std.replace(0.0, np.nan)

        recent = z.tail(lookback_months)
        for date, z_val in recent.items():
            if pd.isna(z_val) or abs(z_val) < threshold:
                continue
            alerts.append(
                AnomalyAlert(
                    series_key=col,
                    series_name=series_names.get(col, col),
                    date=date,
                    value=float(series.loc[date]),
                    z_score=float(z_val),
                    rolling_mean=float(roll_mean.loc[date]),
                    rolling_std=float(roll_std.loc[date]),
                    direction="high" if z_val > 0 else "low",
                    severity="extreme" if abs(z_val) > 3 else "moderate",
                )
            )

    return sorted(alerts, key=lambda a: abs(a.z_score), reverse=True)


# ── YoY / MoM change ─────────────────────────────────────────────────────────
def yoy_pct(series: pd.Series) -> pd.Series:
    return series.pct_change(periods=12) * 100


def mom_pct(series: pd.Series) -> pd.Series:
    return series.pct_change(periods=1) * 100
