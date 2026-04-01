# eur_usd_pipeline/data/fetch_yields.py
"""
Fetch sovereign bond yields:
  - US 2-Year Treasury yield       → FRED (DGS2)
  - German 2-Year Bund yield       → ECB Statistical Data Warehouse API
  - Yield spread (DE2Y - US2Y)     → computed

The yield spread is the core driver for the EUR/USD carry-trade thesis:
  spread narrowing (US yields ↓ or DE yields ↑) → EUR appreciates.
"""

import io
import warnings
import pandas as pd
import numpy as np
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE
from utils.helpers import clean_series, resample_to_weekly, save_dataframe


# ──────────────────────────────────────────────────────────────
# US 2-YEAR TREASURY YIELD (FRED)
# ──────────────────────────────────────────────────────────────

def fetch_us_2y_yield(start=START_DATE, end=END_DATE, weekly=True):
    """
    Fetch the US 2-Year Treasury constant maturity rate from FRED.

    Uses the fredapi library. Falls back to direct FRED REST API
    if fredapi has issues (known: fredapi sometimes misparses XML
    when FRED returns observation notes).

    Returns
    -------
    pd.DataFrame
        Columns: ['us_2y_yield']
    """
    print("[...] Fetching US 2-Year Treasury yield (FRED)")

    series_id = FRED_SERIES["us_2y_yield"]  # DGS2

    # ── Primary method: fredapi ──
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        raw = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = raw.to_frame(name="us_2y_yield")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

    except Exception as e:
        print(f"  [!] fredapi failed ({e}), falling back to REST API")
        df = _fred_rest_fallback(series_id, start, end, col_name="us_2y_yield")

    # FRED marks non-trading days as "." which become NaN
    df["us_2y_yield"] = pd.to_numeric(df["us_2y_yield"], errors="coerce")
    df = df.dropna(subset=["us_2y_yield"])
    df["us_2y_yield"] = clean_series(df["us_2y_yield"], name="US 2Y Yield")

    if weekly:
        df = resample_to_weekly(df)

    print(f"[✓] US 2Y yield: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})")
    return df


def _fred_rest_fallback(series_id, start, end, col_name):
    """
    Direct FRED REST API call as fallback.
    Known fix for fredapi XML parsing issues with certain series.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = [
        {"Date": obs["date"], col_name: obs["value"]}
        for obs in data.get("observations", [])
    ]
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df


# ──────────────────────────────────────────────────────────────
# GERMAN 2-YEAR BUND YIELD (ECB STATISTICAL DATA WAREHOUSE)
# ──────────────────────────────────────────────────────────────
#
# KNOWN ISSUE: FRED does not carry the German 2Y Bund yield.
# The ECB SDW REST API is free, no auth required.
#
# Series key for German 2-year government bond yield:
#   FM.M.DE.EUR.FR2.BB.DE2YT_RR.YLD
#
# Frequency is monthly (M). We interpolate to weekly to match
# the rest of the pipeline. If the ECB API is unreachable,
# we fall back to a synthetic estimate from the ECB deposit
# rate plus a historical spread.
# ──────────────────────────────────────────────────────────────

ECB_SDW_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM/"
    "M.DE.EUR.FR2.BB.DE2YT_RR.YLD"
)

# Fallback: broader Eurozone 2Y yield (more reliably available)
ECB_SDW_FALLBACK_URL = (
    "https://data-api.ecb.europa.eu/service/data/YC/"
    "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"
)


def fetch_de_2y_yield(start=START_DATE, end=END_DATE, weekly=True):
    """
    Fetch the German 2-Year Bund yield from the ECB SDW.

    Falls back to the Eurozone AAA 2-year yield curve spot rate
    if the German-specific series is unavailable.

    Returns
    -------
    pd.DataFrame
        Columns: ['de_2y_yield']
    """
    print("[...] Fetching German 2-Year Bund yield (ECB SDW)")

    # ── Primary: German 2Y Bund ──
    df = _fetch_ecb_series(ECB_SDW_URL, "de_2y_yield", start, end)

    # ── Fallback: Eurozone AAA 2Y yield curve ──
    if df is None or df.empty:
        print("  [!] German 2Y not available, trying Eurozone AAA 2Y curve")
        df = _fetch_ecb_series(ECB_SDW_FALLBACK_URL, "de_2y_yield", start, end)

    # ── Last resort: synthetic from FRED ECB rate + spread ──
    if df is None or df.empty:
        print("  [!] ECB SDW unreachable, building synthetic German 2Y proxy")
        df = _synthetic_de_2y(start, end)

    if df is not None and not df.empty:
        df["de_2y_yield"] = clean_series(df["de_2y_yield"], name="DE 2Y Yield")

        if weekly:
            # Monthly data → weekly via forward-fill interpolation
            df = df.resample("W-FRI").ffill()
            df = df.dropna()

        print(f"[✓] DE 2Y yield: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})")
    else:
        raise RuntimeError("Could not fetch or synthesise German 2Y yield from any source")

    return df


def _fetch_ecb_series(url, col_name, start, end):
    """
    Generic ECB SDW REST API fetcher.

    Returns pd.DataFrame or None on failure.

    ECB SDW returns SDMX-CSV format by default when we set
    the Accept header appropriately.
    """
    headers = {
        "Accept": "text/csv",
    }
    params = {
        "startPeriod": start[:7],   # ECB expects YYYY-MM for monthly
        "endPeriod": end[:7],
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] ECB SDW request failed: {e}")
        return None

    try:
        csv_data = io.StringIO(resp.text)
        raw = pd.read_csv(csv_data)

        # ECB CSV columns vary; typically includes TIME_PERIOD and OBS_VALUE
        if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
            print(f"  [!] Unexpected ECB CSV columns: {list(raw.columns)}")
            return None

        df = pd.DataFrame({
            "Date": pd.to_datetime(raw["TIME_PERIOD"]),
            col_name: pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
        })
        df = df.set_index("Date").sort_index()
        df = df.dropna()

        return df

    except Exception as e:
        print(f"  [!] ECB CSV parsing failed: {e}")
        return None


def _synthetic_de_2y(start, end):
    """
    Build a synthetic German 2Y yield proxy.

    Method: ECB main refinancing rate (from FRED) + historical
    average spread between the ECB rate and the 2Y Bund.
    The 2Y Bund historically trades ~30-80bp above the ECB deposit rate.
    We use a fixed +50bp offset as a rough proxy.

    This is a last-resort fallback — flag it clearly.
    """
    print("  [⚠] Using synthetic DE 2Y = ECB rate + 0.50% (rough proxy)")

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        ecb_rate = fred.get_series("ECBMRRFR", observation_start=start, observation_end=end)
    except Exception:
        ecb_rate = _fred_rest_fallback("ECBMRRFR", start, end, "ecb_rate")
        ecb_rate = ecb_rate["ecb_rate"]

    ecb_rate = pd.to_numeric(ecb_rate, errors="coerce").dropna()
    synthetic = ecb_rate + 0.50  # Rough 2Y-ECB rate spread

    df = pd.DataFrame({"de_2y_yield": synthetic})
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    return df


# ──────────────────────────────────────────────────────────────
# YIELD SPREAD CALCULATION
# ──────────────────────────────────────────────────────────────

def compute_yield_spread(us_2y_df, de_2y_df):
    """
    Compute the 2Y yield spread: German Bund minus US Treasury.

    A narrowing spread (becoming less negative or more positive)
    is theoretically EUR-positive.

    Returns
    -------
    pd.DataFrame
        Columns: ['us_2y_yield', 'de_2y_yield', 'yield_spread_2y',
                   'spread_change', 'spread_direction']
    """
    print("[...] Computing 2Y yield spread (DE - US)")

    combined = us_2y_df.join(de_2y_df, how="inner")
    combined = combined.ffill(limit=2).dropna()

    combined["yield_spread_2y"] = combined["de_2y_yield"] - combined["us_2y_yield"]
    combined["spread_change"] = combined["yield_spread_2y"].diff()

    # Direction: 1 = narrowing (EUR positive), -1 = widening (EUR negative)
    combined["spread_direction"] = np.where(
        combined["spread_change"] > 0, 1, -1
    )

    print(f"[✓] Yield spread: {len(combined)} rows")
    print(f"    Latest spread: {combined['yield_spread_2y'].iloc[-1]:.2f}%")
    print(f"    12-week Δ:     {combined['yield_spread_2y'].iloc[-1] - combined['yield_spread_2y'].iloc[-13]:.2f}%")

    return combined


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FETCH ALL YIELDS + SPREAD
# ──────────────────────────────────────────────────────────────

def fetch_all_yields(start=START_DATE, end=END_DATE, weekly=True, save=True):
    """
    Fetch US 2Y, DE 2Y, and compute the spread.

    Returns
    -------
    pd.DataFrame
        Full yield + spread dataset.
    """
    us_2y = fetch_us_2y_yield(start=start, end=end, weekly=weekly)
    de_2y = fetch_de_2y_yield(start=start, end=end, weekly=weekly)
    spread_df = compute_yield_spread(us_2y, de_2y)

    if save:
        save_dataframe(spread_df, "yield_data.csv")

    return spread_df


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_all_yields()
    print("\nSample:")
    print(df.tail())