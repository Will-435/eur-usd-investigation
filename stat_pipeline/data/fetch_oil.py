# eur_usd_pipeline/data/fetch_oil.py
"""
Fetch Brent Crude oil prices and construct Terms of Trade indicators.

Core thesis: If oil prices drop, the Eurozone (a net energy importer)
benefits from improved terms of trade, which historically supports
EUR appreciation. We invert Brent to show this inverse relationship.
"""

import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKERS, START_DATE, END_DATE
from utils.helpers import clean_series, resample_to_weekly, save_dataframe


# ──────────────────────────────────────────────────────────────
# KNOWN ISSUE FIX: yfinance Brent ticker
# BZ=F is the ICE Brent front-month future. Occasionally yfinance
# returns incomplete data for BZ=F before 2019. If that happens
# we fall back to CL=F (WTI) which is highly correlated (~0.95)
# and available further back. We note the substitution clearly.
# ──────────────────────────────────────────────────────────────


def _flatten_columns(df):
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_brent(start=START_DATE, end=END_DATE, weekly=True):
    """
    Fetch Brent Crude front-month futures price.

    Falls back to WTI (CL=F) if Brent data is insufficient.

    Returns
    -------
    pd.DataFrame
        Columns: ['brent_close', 'brent_return', 'brent_inverted']
    """
    print("[...] Fetching Brent Crude oil prices")

    df = _try_download(TICKERS["brent"], "brent", start, end)

    # Fallback to WTI if Brent has < 100 rows
    if df is None or len(df) < 100:
        print("  [!] Brent data insufficient, falling back to WTI (CL=F)")
        df = _try_download("CL=F", "brent", start, end)
        if df is not None:
            print("  [⚠] Using WTI as Brent proxy — highly correlated (~0.95)")

    if df is None or df.empty:
        raise RuntimeError("Could not fetch oil prices from any source")

    # Build output columns
    out = pd.DataFrame(index=df.index)
    out["brent_close"] = clean_series(df["Close"], name="Brent Close")
    out["brent_return"] = out["brent_close"].pct_change()

    # Inverted Brent: higher value = lower oil = EUR positive
    # Normalised as 1/price * 100 for readability
    out["brent_inverted"] = (1.0 / out["brent_close"]) * 100

    if weekly:
        out = resample_to_weekly(out)

    out = out.dropna(subset=["brent_close"])
    print(f"[✓] Brent Crude: {len(out)} rows ({out.index.min().date()} → {out.index.max().date()})")

    return out


def _try_download(ticker, name, start, end):
    """Attempt yfinance download with error handling."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(ticker, start=start, end=end, progress=False)
        if raw.empty:
            return None
        raw = _flatten_columns(raw)
        return raw
    except Exception as e:
        print(f"  [!] Failed to download {ticker}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# TERMS OF TRADE PROXY
# ──────────────────────────────────────────────────────────────

def compute_terms_of_trade(brent_df, eurusd_df=None):
    """
    Construct a Terms of Trade (ToT) proxy for the Eurozone.

    The Eurozone is a major net energy importer. When energy prices
    fall, import costs decline relative to export revenue, improving
    the ToT. This is captured by inverting Brent crude.

    Optionally, if EUR/USD is provided, computes the rolling
    correlation between inverted oil and EUR/USD to validate the
    historical relationship.

    Parameters
    ----------
    brent_df : pd.DataFrame
        Must contain 'brent_inverted' column.
    eurusd_df : pd.DataFrame, optional
        Must contain 'eurusd_close' column.

    Returns
    -------
    pd.DataFrame
        ToT proxy with optional rolling correlation.
    """
    print("[...] Building Terms of Trade proxy")

    tot = brent_df[["brent_close", "brent_inverted"]].copy()

    # Z-score normalise inverted Brent for comparability
    tot["tot_proxy_zscore"] = (
        (tot["brent_inverted"] - tot["brent_inverted"].mean())
        / tot["brent_inverted"].std()
    )

    # Rolling 26-week (6-month) rate of change in oil
    tot["brent_roc_26w"] = tot["brent_close"].pct_change(periods=26)

    # Oil regime flag: bull (rising) vs bear (falling)
    tot["oil_regime"] = np.where(
        tot["brent_close"].rolling(52).mean() > tot["brent_close"].rolling(13).mean(),
        "falling",  # 52w MA above 13w MA → price declining
        "rising"
    )

    if eurusd_df is not None and "eurusd_close" in eurusd_df.columns:
        # Merge for correlation calculation
        merged = tot.join(eurusd_df[["eurusd_close"]], how="inner")

        # Rolling 52-week correlation: inverted Brent vs EUR/USD
        merged["tot_eur_corr_52w"] = (
            merged["brent_inverted"]
            .rolling(52, min_periods=26)
            .corr(merged["eurusd_close"])
        )

        # Copy correlation back to tot frame
        tot = tot.join(merged[["tot_eur_corr_52w"]], how="left")

        avg_corr = merged["tot_eur_corr_52w"].mean()
        latest_corr = merged["tot_eur_corr_52w"].iloc[-1]
        print(f"    Avg 52w correlation (inverted oil vs EUR/USD): {avg_corr:.3f}")
        print(f"    Latest 52w correlation: {latest_corr:.3f}")

    print(f"[✓] Terms of Trade proxy: {len(tot)} rows")

    return tot


# ──────────────────────────────────────────────────────────────
# ENERGY IMPORT DEPENDENCY INDICATOR
# ──────────────────────────────────────────────────────────────

def compute_energy_pressure_index(brent_df, lookback_weeks=52):
    """
    Create an energy pressure index for the Eurozone.

    Combines:
    - Oil price level (z-scored)
    - Oil price momentum (rate of change)
    - Oil volatility (rolling std of returns)

    Higher index = more pressure on Eurozone ToT (EUR negative).
    Lower index = relief on energy imports (EUR positive).

    Returns
    -------
    pd.DataFrame
        Column: ['energy_pressure_index']
    """
    print("[...] Computing energy pressure index")

    df = brent_df.copy()

    # Component 1: z-scored price level
    price_z = (
        (df["brent_close"] - df["brent_close"].rolling(lookback_weeks).mean())
        / df["brent_close"].rolling(lookback_weeks).std()
    )

    # Component 2: 13-week momentum
    momentum = df["brent_close"].pct_change(13)
    momentum_z = (momentum - momentum.rolling(lookback_weeks).mean()) / momentum.rolling(lookback_weeks).std()

    # Component 3: realised volatility (rolling std of weekly returns)
    vol = df["brent_return"].rolling(13).std() * np.sqrt(52)  # Annualised
    vol_z = (vol - vol.rolling(lookback_weeks).mean()) / vol.rolling(lookback_weeks).std()

    # Composite: equal-weighted
    df["energy_pressure_index"] = (price_z + momentum_z + vol_z) / 3.0

    df = df[["energy_pressure_index"]].dropna()
    print(f"[✓] Energy pressure index: {len(df)} rows")

    return df


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FETCH ALL OIL DATA
# ──────────────────────────────────────────────────────────────

def fetch_all_oil(start=START_DATE, end=END_DATE, weekly=True, save=True):
    """
    Fetch Brent, build ToT proxy and energy pressure index.

    Returns
    -------
    pd.DataFrame
        Combined oil/ToT dataset.
    """
    brent = fetch_brent(start=start, end=end, weekly=weekly)
    tot = compute_terms_of_trade(brent)
    energy = compute_energy_pressure_index(brent)

    combined = brent.join(
        tot[[c for c in tot.columns if c not in brent.columns]],
        how="left"
    )
    combined = combined.join(energy, how="left")

    if save:
        save_dataframe(combined, "oil_data.csv")

    return combined


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_all_oil()
    print("\nSample:")
    print(df.tail())