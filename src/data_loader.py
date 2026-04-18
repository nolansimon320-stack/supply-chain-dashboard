import os
import pandas as pd
import numpy as np
import streamlit as st
from fredapi import Fred

# ── Series catalogue ──────────────────────────────────────────────────────────
SERIES_CONFIG: dict[str, dict] = {
    "ppi_all": {
        "id": "PPIACO",
        "name": "PPI: All Commodities",
        "short": "PPI (All)",
        "category": "upstream",
        "unit": "Index (1982=100)",
        "description": (
            "Measures average changes in prices received by domestic producers for their "
            "output — the broadest upstream cost signal in the supply chain."
        ),
    },
    "ppi_crude": {
        "id": "PPICRM",
        "name": "PPI: Crude Materials",
        "short": "PPI Crude",
        "category": "upstream",
        "unit": "Index (1982=100)",
        "description": (
            "Tracks prices of raw, unprocessed materials entering the supply chain. "
            "Typically leads finished-goods inflation by 1–3 months."
        ),
    },
    "ppi_finished": {
        "id": "PPIFGS",
        "name": "PPI: Finished Goods",
        "short": "PPI Finished",
        "category": "downstream",
        "unit": "Index (1982=100)",
        "description": (
            "Tracks price changes for goods ready for sale to end users — "
            "a key measure of cost pass-through from upstream inputs."
        ),
    },
    "crude_oil": {
        "id": "DCOILWTICO",
        "name": "Crude Oil — WTI",
        "short": "Crude Oil",
        "category": "upstream",
        "unit": "USD / barrel",
        "description": (
            "West Texas Intermediate spot price. Energy costs permeate every stage "
            "of production and logistics; WTI is the primary benchmark."
        ),
    },
    "natural_gas": {
        "id": "DHHNGSP",
        "name": "Natural Gas — Henry Hub",
        "short": "Natural Gas",
        "category": "upstream",
        "unit": "USD / MMBtu",
        "description": (
            "Henry Hub natural gas spot price. Critical input for chemical manufacturing, "
            "plastics, fertiliser, and industrial heating."
        ),
    },
    "copper": {
        "id": "PCOPPUSDM",
        "name": "Copper Price",
        "short": "Copper",
        "category": "upstream",
        "unit": "USD / metric ton",
        "description": (
            "Global copper price from the IMF. Often called 'Dr. Copper' because its "
            "demand mirrors industrial activity and construction pipelines."
        ),
    },
    "aluminum": {
        "id": "PALUMUSDM",
        "name": "Aluminum Price",
        "short": "Aluminum",
        "category": "upstream",
        "unit": "USD / metric ton",
        "description": (
            "Global aluminum price from the IMF. Core input for packaging, "
            "automotive, and aerospace supply chains."
        ),
    },
    "wheat": {
        "id": "PWHEAMTUSDM",
        "name": "Wheat Price",
        "short": "Wheat",
        "category": "upstream",
        "unit": "USD / metric ton",
        "description": (
            "Global wheat price. Indicator of agricultural commodity pressure "
            "relevant to food & beverage and packaging supply chains."
        ),
    },
    "cpi": {
        "id": "CPIAUCSL",
        "name": "CPI: All Items",
        "short": "CPI",
        "category": "downstream",
        "unit": "Index (1982–84=100)",
        "description": (
            "Consumer Price Index for all urban consumers — the primary downstream "
            "measure of how upstream cost pressure reaches end consumers."
        ),
    },
    "cpi_transport": {
        "id": "CPITRNSL",
        "name": "CPI: Transportation",
        "short": "Transport CPI",
        "category": "logistics",
        "unit": "Index (1982–84=100)",
        "description": (
            "CPI sub-index for transportation services. Used here as a logistics-cost "
            "proxy capturing fuel, freight, and shipping cost pass-through to consumers."
        ),
    },
}

DEFAULT_START = "2015-01-01"


# ── FRED client ───────────────────────────────────────────────────────────────
def _get_fred(api_key: str) -> Fred:
    if not api_key:
        raise ValueError(
            "FRED API key missing. Add it to .streamlit/secrets.toml "
            "or set the FRED_API_KEY environment variable."
        )
    return Fred(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_series(api_key: str, series_id: str, start_date: str) -> pd.Series:
    fred = _get_fred(api_key)
    raw = fred.get_series(series_id, observation_start=start_date)
    return raw.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(api_key: str, start_date: str = DEFAULT_START) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch all configured FRED series, resample to monthly frequency, and merge.
    Returns (DataFrame, list_of_error_strings).
    """
    monthly_frames: dict[str, pd.Series] = {}
    errors: list[str] = []

    for key, cfg in SERIES_CONFIG.items():
        try:
            raw = _fetch_series(api_key, cfg["id"], start_date)
            # Resample daily series to month-end mean; monthly series pass through unchanged
            monthly = raw.resample("ME").mean()
            monthly.name = key
            monthly_frames[key] = monthly
        except Exception as exc:
            errors.append(f"{cfg['name']} ({cfg['id']}): {exc}")

    if not monthly_frames:
        raise RuntimeError("No FRED series loaded successfully. Check your API key.")

    df = pd.concat(monthly_frames.values(), axis=1)
    df.columns = list(monthly_frames.keys())
    df = df.sort_index()

    # Forward-fill isolated gaps (≤ 2 months), then drop rows missing more than half
    df = df.ffill(limit=2)
    df = df.dropna(thresh=max(1, len(df.columns) // 2))

    return df, errors


# ── Convenience helpers ───────────────────────────────────────────────────────
def series_names() -> dict[str, str]:
    return {k: v["name"] for k, v in SERIES_CONFIG.items()}


def series_short_names() -> dict[str, str]:
    return {k: v["short"] for k, v in SERIES_CONFIG.items()}


def get_series_by_category(category: str) -> list[str]:
    return [k for k, v in SERIES_CONFIG.items() if v["category"] == category]


def last_updated(df: pd.DataFrame) -> str:
    return df.index[-1].strftime("%B %Y") if len(df) else "N/A"
