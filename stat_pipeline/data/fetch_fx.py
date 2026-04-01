# eur_usd_pipeline/data/fetch_fx.py
"""
Fetch EUR/USD spot rate and Dollar Index (DXY) from yfinance.
Returns clean, weekly-resampled DataFrames.
"""

import pandas as pd
import yfinance as yf
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKERS, START_DATE, END_DATE
from utils.helpers import clean_series, resample_to_weekly, save_dataframe


# ──────────────────────────────────────────────────────────────
# KNOWN ISSUE FIX: yfinance >= 0.2.31
# yfinance switched to a new backend and sometimes returns
# MultiIndex columns when downloading single tickers.
# We flatten them explicitly to avoid KeyError on 'Close'.
# Also, auto_adjust=True is now default — 'Adj Close' is gone,
# 'Close' already contains adjusted values.
# ──────────────────────────────────────────────────────────────


def _flatten_columns(df):
    """
    yfinance can return MultiIndex columns like ('Close', 'EURUSD=X').
    Flatten to just the first level.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_eurusd(start=START_DATE, end=END_DATE, weekly=True):
    """
    Fetch EUR/USD spot rate.

    Parameters
    ----------
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    weekly : bool
        If True, resample to weekly (Friday close).

    Returns
    -------
    pd.DataFrame
        Columns: ['eurusd_close', 'eurusd_return']
    """
    print("[...] Fetching EUR/USD spot rate")

    ticker = TICKERS["eurusd"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    raw = _flatten_columns(raw)

    df = pd.DataFrame(index=raw.index)
    df["eurusd_close"] = clean_series(raw["Close"], name="EUR/USD Close")
    df["eurusd_return"] = df["eurusd_close"].pct_change()

    if weekly:
        df = resample_to_weekly(df)

    df = df.dropna(subset=["eurusd_close"])
    print(f"[✓] EUR/USD: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})")

    return df


def fetch_dxy(start=START_DATE, end=END_DATE, weekly=True):
    """
    Fetch the US Dollar Index (DXY).
    Useful as an inverse proxy for Euro strength.

    Returns
    -------
    pd.DataFrame
        Columns: ['dxy_close', 'dxy_return']
    """
    print("[...] Fetching DXY (Dollar Index)")

    ticker = TICKERS["dxy"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    raw = _flatten_columns(raw)

    df = pd.DataFrame(index=raw.index)
    df["dxy_close"] = clean_series(raw["Close"], name="DXY Close")
    df["dxy_return"] = df["dxy_close"].pct_change()

    if weekly:
        df = resample_to_weekly(df)

    df = df.dropna(subset=["dxy_close"])
    print(f"[✓] DXY: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})")

    return df


def fetch_vix(start=START_DATE, end=END_DATE, weekly=True):
    """
    Fetch CBOE VIX — used as a global risk sentiment proxy.

    Returns
    -------
    pd.DataFrame
        Columns: ['vix_close']
    """
    print("[...] Fetching VIX (risk sentiment proxy)")

    ticker = TICKERS["vix"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    raw = _flatten_columns(raw)

    df = pd.DataFrame(index=raw.index)
    df["vix_close"] = clean_series(raw["Close"], name="VIX Close")

    if weekly:
        df = resample_to_weekly(df)

    df = df.dropna(subset=["vix_close"])
    print(f"[✓] VIX: {len(df)} rows ({df.index.min().date()} → {df.index.max().date()})")

    return df


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FETCH ALL FX DATA AT ONCE
# ──────────────────────────────────────────────────────────────

def fetch_all_fx(start=START_DATE, end=END_DATE, weekly=True, save=True):
    """
    Fetch EUR/USD, DXY, and VIX together.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all FX / risk columns.
    """
    eurusd = fetch_eurusd(start=start, end=end, weekly=weekly)
    dxy = fetch_dxy(start=start, end=end, weekly=weekly)
    vix = fetch_vix(start=start, end=end, weekly=weekly)

    combined = eurusd.join(dxy, how="outer").join(vix, how="outer")
    combined = combined.ffill(limit=2).dropna()

    print(f"[✓] Combined FX data: {len(combined)} rows, {list(combined.columns)}")

    if save:
        save_dataframe(combined, "fx_data.csv")

    return combined


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_all_fx()
    print("\nSample:")
    print(df.tail())