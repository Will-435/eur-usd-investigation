# eur_usd_pipeline/features/technical.py
"""
Technical indicators for EUR/USD and supporting series.

Computes classic momentum, trend, and volatility indicators
that serve as features in the VAR and GLM models.

Indicators included:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands (width and %B position)
  - Rate of Change (ROC) at multiple horizons
  - Average True Range (ATR) — adapted for FX (no gaps)
  - Stochastic Oscillator
  - On-Balance momentum proxy
  - Hurst exponent (mean-reversion vs trend detection)

All indicators are computed from scratch (no TA-Lib dependency)
to keep the environment lightweight and avoid the notoriously
painful TA-Lib C library installation.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# RSI (RELATIVE STRENGTH INDEX)
# ──────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """
    Compute RSI using the Wilder smoothing method (exponential).

    Parameters
    ----------
    series : pd.Series
        Price series (typically close prices).
    period : int
        Lookback window (default 14).

    Returns
    -------
    pd.Series
        RSI values [0, 100].
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ──────────────────────────────────────────────────────────────
# MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)
# ──────────────────────────────────────────────────────────────

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Compute MACD line, signal line, and histogram.

    Parameters
    ----------
    series : pd.Series
        Price series.
    fast, slow, signal : int
        EMA periods.

    Returns
    -------
    pd.DataFrame
        Columns: ['macd_line', 'macd_signal', 'macd_histogram']
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    }, index=series.index)


# ──────────────────────────────────────────────────────────────
# BOLLINGER BANDS
# ──────────────────────────────────────────────────────────────

def compute_bollinger_bands(series, period=20, num_std=2.0):
    """
    Compute Bollinger Bands, band width, and %B position.

    Parameters
    ----------
    series : pd.Series
        Price series.
    period : int
        SMA lookback.
    num_std : float
        Number of standard deviations for bands.

    Returns
    -------
    pd.DataFrame
        Columns: ['bb_middle', 'bb_upper', 'bb_lower',
                   'bb_width', 'bb_pct_b']
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    # Band width: normalised by middle band
    width = (upper - lower) / middle

    # %B: where price sits relative to bands (0 = lower, 1 = upper)
    pct_b = (series - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    }, index=series.index)


# ──────────────────────────────────────────────────────────────
# RATE OF CHANGE (ROC) — MULTI-HORIZON
# ──────────────────────────────────────────────────────────────

def compute_roc(series, periods=None):
    """
    Compute rate of change at multiple horizons.

    Parameters
    ----------
    series : pd.Series
        Price series.
    periods : list of int
        Lookback periods (default: [4, 13, 26, 52] weeks).

    Returns
    -------
    pd.DataFrame
        One column per period: 'roc_{n}w'.
    """
    if periods is None:
        periods = [4, 13, 26, 52]

    result = pd.DataFrame(index=series.index)
    for p in periods:
        result[f"roc_{p}w"] = series.pct_change(periods=p) * 100

    return result


# ──────────────────────────────────────────────────────────────
# STOCHASTIC OSCILLATOR
# ──────────────────────────────────────────────────────────────

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Compute Stochastic Oscillator (%K and %D).

    For weekly FX data where we may only have close prices,
    we approximate high/low from rolling max/min of close.

    Parameters
    ----------
    high, low, close : pd.Series
        Price series. If high/low are None, derived from close.
    k_period : int
        %K lookback.
    d_period : int
        %D smoothing period.

    Returns
    -------
    pd.DataFrame
        Columns: ['stoch_k', 'stoch_d']
    """
    if high is None:
        high = close.rolling(k_period).max()
    if low is None:
        low = close.rolling(k_period).min()

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)

    stoch_k = ((close - lowest_low) / denom) * 100
    stoch_d = stoch_k.rolling(d_period).mean()

    return pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    }, index=close.index)


# ──────────────────────────────────────────────────────────────
# AVERAGE TRUE RANGE (ATR) — ADAPTED FOR FX
# ──────────────────────────────────────────────────────────────

def compute_atr(close, period=14):
    """
    Compute ATR from close-only data (no intraday high/low).

    Approximates true range using close-to-close differences
    scaled by a factor of 1.25 (empirical FX adjustment for
    the missing intraday range component).

    Parameters
    ----------
    close : pd.Series
    period : int

    Returns
    -------
    pd.Series
        ATR values.
    """
    # Close-to-close range as TR proxy
    tr_proxy = close.diff().abs() * 1.25

    # Wilder smoothing
    atr = tr_proxy.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return atr


# ──────────────────────────────────────────────────────────────
# HURST EXPONENT (TREND VS MEAN-REVERSION)
# ──────────────────────────────────────────────────────────────

def compute_hurst(series, max_lag=100):
    """
    Estimate the Hurst exponent using rescaled range (R/S) analysis.

    H > 0.5 → trending (momentum)
    H = 0.5 → random walk
    H < 0.5 → mean-reverting

    Computed on a rolling basis using expanding windows.

    Parameters
    ----------
    series : pd.Series
        Price returns or log returns.
    max_lag : int
        Maximum lag for R/S calculation.

    Returns
    -------
    float
        Hurst exponent estimate.
    """
    clean = series.dropna().values
    if len(clean) < max_lag:
        return np.nan

    lags = range(2, min(max_lag, len(clean) // 2))
    rs_values = []

    for lag in lags:
        # Split into non-overlapping sub-series
        n_subseries = len(clean) // lag
        if n_subseries < 1:
            continue

        rs_lag = []
        for i in range(n_subseries):
            subseries = clean[i * lag: (i + 1) * lag]
            mean_sub = np.mean(subseries)
            deviate = np.cumsum(subseries - mean_sub)
            r = np.max(deviate) - np.min(deviate)
            s = np.std(subseries, ddof=1)
            if s > 0:
                rs_lag.append(r / s)

        if rs_lag:
            rs_values.append((lag, np.mean(rs_lag)))

    if len(rs_values) < 5:
        return np.nan

    lags_arr = np.array([x[0] for x in rs_values])
    rs_arr = np.array([x[1] for x in rs_values])

    # Log-log regression
    log_lags = np.log(lags_arr)
    log_rs = np.log(rs_arr)

    # Simple OLS
    slope = np.polyfit(log_lags, log_rs, 1)[0]

    return slope


def compute_rolling_hurst(series, window=104, step=4):
    """
    Compute rolling Hurst exponent.

    Parameters
    ----------
    series : pd.Series
        Returns series.
    window : int
        Rolling window size (default: 104 weeks = 2 years).
    step : int
        Step size to reduce computation (default: every 4 weeks).

    Returns
    -------
    pd.Series
        Rolling Hurst exponent.
    """
    result = pd.Series(index=series.index, dtype=float)

    indices = range(window, len(series), step)
    for i in indices:
        window_data = series.iloc[i - window:i]
        h = compute_hurst(window_data)
        result.iloc[i] = h

    # Forward-fill the gaps from stepping
    result = result.ffill()

    return result


# ──────────────────────────────────────────────────────────────
# TREND STRENGTH (ADX-LIKE)
# ──────────────────────────────────────────────────────────────

def compute_trend_strength(close, period=14):
    """
    Compute a directional movement / trend strength indicator
    from close-only data.

    Approximates ADX concept using directional changes in close.

    Returns
    -------
    pd.DataFrame
        Columns: ['trend_strength', 'trend_direction']
        trend_strength: 0–100 scale (higher = stronger trend)
        trend_direction: +1 (uptrend) or -1 (downtrend)
    """
    diff = close.diff()
    abs_diff = diff.abs()

    # Smoothed directional movement
    pos_dm = diff.where(diff > 0, 0.0)
    neg_dm = (-diff).where(diff < 0, 0.0)

    smooth_pos = pos_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smooth_neg = neg_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smooth_range = abs_diff.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    smooth_range = smooth_range.replace(0, np.nan)

    di_pos = (smooth_pos / smooth_range) * 100
    di_neg = (smooth_neg / smooth_range) * 100

    di_sum = di_pos + di_neg
    di_sum = di_sum.replace(0, np.nan)

    dx = (abs(di_pos - di_neg) / di_sum) * 100
    adx_proxy = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    direction = np.where(di_pos > di_neg, 1, -1)

    return pd.DataFrame({
        "trend_strength": adx_proxy,
        "trend_direction": direction,
    }, index=close.index)


# ──────────────────────────────────────────────────────────────
# MASTER TECHNICAL FEATURE BUILDER
# ──────────────────────────────────────────────────────────────

def compute_all_technicals(price_df, price_col="eurusd_close", prefix="eur"):
    """
    Compute all technical indicators for a given price series.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain the column specified by `price_col`.
    price_col : str
        Column name of the close price.
    prefix : str
        Prefix for output column names (e.g. 'eur', 'brent').

    Returns
    -------
    pd.DataFrame
        All technical indicators with prefixed column names.
    """
    print(f"[...] Computing technical indicators for '{price_col}'")

    if price_col not in price_df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")

    close = price_df[price_col]
    result = pd.DataFrame(index=price_df.index)

    # ── RSI ──
    result[f"{prefix}_rsi_14"] = compute_rsi(close, period=14)

    # ── MACD ──
    macd = compute_macd(close)
    for col in macd.columns:
        result[f"{prefix}_{col}"] = macd[col]

    # ── Bollinger Bands ──
    bb = compute_bollinger_bands(close, period=20)
    result[f"{prefix}_bb_width"] = bb["bb_width"]
    result[f"{prefix}_bb_pct_b"] = bb["bb_pct_b"]

    # ── Rate of Change ──
    roc = compute_roc(close, periods=[4, 13, 26, 52])
    for col in roc.columns:
        result[f"{prefix}_{col}"] = roc[col]

    # ── Stochastic Oscillator ──
    stoch = compute_stochastic(high=None, low=None, close=close)
    result[f"{prefix}_stoch_k"] = stoch["stoch_k"]
    result[f"{prefix}_stoch_d"] = stoch["stoch_d"]

    # ── ATR ──
    result[f"{prefix}_atr_14"] = compute_atr(close, period=14)

    # ── Trend Strength ──
    trend = compute_trend_strength(close)
    result[f"{prefix}_trend_strength"] = trend["trend_strength"]
    result[f"{prefix}_trend_direction"] = trend["trend_direction"]

    # ── Moving averages (for crossover signals) ──
    result[f"{prefix}_sma_13"] = close.rolling(13).mean()
    result[f"{prefix}_sma_26"] = close.rolling(26).mean()
    result[f"{prefix}_sma_52"] = close.rolling(52).mean()

    # MA crossover signal: 13w vs 52w
    result[f"{prefix}_ma_cross_signal"] = np.where(
        result[f"{prefix}_sma_13"] > result[f"{prefix}_sma_52"], 1, -1
    )

    # ── Distance from 52-week extremes ──
    high_52 = close.rolling(52).max()
    low_52 = close.rolling(52).min()
    range_52 = high_52 - low_52
    range_52 = range_52.replace(0, np.nan)
    result[f"{prefix}_pct_from_52w_high"] = (close - high_52) / high_52 * 100
    result[f"{prefix}_position_in_52w_range"] = (close - low_52) / range_52

    # ── Hurst exponent (rolling) ──
    if "eurusd_return" in price_df.columns:
        returns = price_df["eurusd_return"]
    else:
        returns = close.pct_change()

    result[f"{prefix}_hurst"] = compute_rolling_hurst(returns, window=104, step=4)

    n_indicators = len(result.columns)
    result = result.dropna(how="all")
    print(f"[✓] {n_indicators} technical indicators computed for '{prefix}'")

    return result


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    from utils.helpers import resample_to_weekly

    print("Fetching test data...")
    raw = yf.download("EURUSD=X", start="2020-01-01", progress=False)
    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({"eurusd_close": raw["Close"]})
    df["eurusd_return"] = df["eurusd_close"].pct_change()
    df = resample_to_weekly(df)

    technicals = compute_all_technicals(df, price_col="eurusd_close", prefix="eur")
    print(f"\nShape: {technicals.shape}")
    print(f"Columns: {list(technicals.columns)}")
    print(f"\nSample:")
    print(technicals.tail())

    # Compute Hurst for full series
    h = compute_hurst(df["eurusd_return"].dropna())
    print(f"\nFull-sample Hurst exponent: {h:.3f}")
    if h > 0.55:
        print("  → Trending regime (momentum strategies favoured)")
    elif h < 0.45:
        print("  → Mean-reverting regime (contrarian strategies favoured)")
    else:
        print("  → Near random walk")