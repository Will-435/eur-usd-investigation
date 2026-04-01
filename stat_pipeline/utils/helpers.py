# eur_usd_pipeline/utils/helpers.py
"""
Shared utility functions: date alignment, stationarity testing,
I/O helpers, and data cleaning used across the pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ──────────────────────────────────────────────────────────────

def ensure_directories():
    """Create all output directories if they don't exist."""
    from config import OUTPUT_DIR, PLOT_DIR, DATA_DIR, MODEL_DIR
    for d in [OUTPUT_DIR, PLOT_DIR, DATA_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"[✓] Output directories verified under '{OUTPUT_DIR}/'")


# ──────────────────────────────────────────────────────────────
# DATE ALIGNMENT
# ──────────────────────────────────────────────────────────────

def align_series(*dataframes, how="inner", freq="W-FRI"):
    """
    Align multiple DataFrames/Series to a common date index.

    Parameters
    ----------
    *dataframes : pd.DataFrame or pd.Series
        Any number of time-indexed pandas objects.
    how : str
        Join type — 'inner' (default, safest) or 'outer'.
    freq : str
        Resample frequency. 'W-FRI' aligns to weekly on Fridays,
        which matches COT release schedule. Use 'B' for business daily.

    Returns
    -------
    pd.DataFrame
        Single DataFrame with all columns aligned.
    """
    combined = pd.DataFrame()
    for i, df in enumerate(dataframes):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        # Resample to target frequency using last available value
        df_resampled = df.resample(freq).last()
        if combined.empty:
            combined = df_resampled
        else:
            combined = combined.join(df_resampled, how=how)

    # Forward-fill small gaps (max 2 periods), then drop remaining NaNs
    combined = combined.ffill(limit=2)
    initial_len = len(combined)
    combined = combined.dropna()
    dropped = initial_len - len(combined)
    if dropped > 0:
        print(f"[i] Alignment dropped {dropped} rows with residual NaNs")

    return combined


def resample_to_weekly(df, date_col=None):
    """
    Resample a DataFrame to weekly frequency (Friday close).
    If date_col is specified, sets it as index first.
    """
    if date_col and date_col in df.columns:
        df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    return df.resample("W-FRI").last().dropna(how="all")


# ──────────────────────────────────────────────────────────────
# STATIONARITY TESTING
# ──────────────────────────────────────────────────────────────

def test_stationarity(series, name="series", alpha=0.05, verbose=True):
    """
    Run both ADF and KPSS stationarity tests.

    Returns
    -------
    dict with keys: 'is_stationary', 'adf_pvalue', 'kpss_pvalue',
                    'differencing_needed'
    """
    result = {
        "name": name,
        "is_stationary": False,
        "adf_pvalue": None,
        "kpss_pvalue": None,
        "differencing_needed": 0,
    }

    clean = series.dropna()
    if len(clean) < 20:
        warnings.warn(f"[!] {name}: too few observations ({len(clean)}) for stationarity test")
        return result

    # ADF test — H0: unit root (non-stationary)
    adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
    result["adf_pvalue"] = adf_p

    # KPSS test — H0: stationary (opposite null!)
    # Suppress the interpolation warning from statsmodels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")
    result["kpss_pvalue"] = kpss_p

    # Stationary if: ADF rejects (p < alpha) AND KPSS fails to reject (p > alpha)
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha

    if adf_stationary and kpss_stationary:
        result["is_stationary"] = True
        result["differencing_needed"] = 0
    else:
        result["is_stationary"] = False
        result["differencing_needed"] = 1  # Suggest first differencing

    if verbose:
        status = "✓ Stationary" if result["is_stationary"] else "✗ Non-stationary"
        print(f"  [{status}] {name}: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}")

    return result


def make_stationary(df, verbose=True):
    """
    Test each column for stationarity and difference as needed.

    Returns
    -------
    df_stationary : pd.DataFrame
        Stationary version of the data.
    diff_record : dict
        Maps column name → number of differences applied.
    """
    df_out = df.copy()
    diff_record = {}

    if verbose:
        print("\n── Stationarity Tests ──")

    for col in df_out.columns:
        result = test_stationarity(df_out[col], name=col, verbose=verbose)
        d = result["differencing_needed"]
        diff_record[col] = d
        if d > 0:
            for _ in range(d):
                df_out[col] = df_out[col].diff()

    df_out = df_out.dropna()

    if verbose:
        diffed = [c for c, d in diff_record.items() if d > 0]
        if diffed:
            print(f"  [i] Differenced columns: {diffed}")
        else:
            print(f"  [i] All columns already stationary")

    return df_out, diff_record


# ──────────────────────────────────────────────────────────────
# DATA CLEANING
# ──────────────────────────────────────────────────────────────

def clean_series(series, name="series", remove_outliers=True, z_thresh=4.0):
    """
    Clean a time series: handle NaNs, optionally cap outliers.

    Parameters
    ----------
    series : pd.Series
    name : str
    remove_outliers : bool
        If True, replace values beyond z_thresh std devs with NaN,
        then forward-fill.
    z_thresh : float
        Z-score threshold for outlier detection.

    Returns
    -------
    pd.Series
    """
    s = series.copy()
    initial_nans = s.isna().sum()

    if remove_outliers and len(s.dropna()) > 30:
        z = np.abs(stats.zscore(s.dropna()))
        # Map z-scores back to original index
        z_series = pd.Series(z, index=s.dropna().index)
        outlier_mask = z_series > z_thresh
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            s.loc[z_series[outlier_mask].index] = np.nan
            print(f"  [i] {name}: capped {n_outliers} outliers (|z| > {z_thresh})")

    # Forward-fill, then back-fill any remaining at start
    s = s.ffill().bfill()

    return s


def winsorize_dataframe(df, lower=0.01, upper=0.99):
    """Winsorize all numeric columns to [lower, upper] quantiles."""
    df_out = df.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        lo = df_out[col].quantile(lower)
        hi = df_out[col].quantile(upper)
        df_out[col] = df_out[col].clip(lo, hi)
    return df_out


# ──────────────────────────────────────────────────────────────
# SCALING
# ──────────────────────────────────────────────────────────────

def scale_features(df, exclude_cols=None):
    """
    Z-score standardise all columns except those in exclude_cols.

    Returns
    -------
    df_scaled : pd.DataFrame
    scaler : StandardScaler (fitted, so you can inverse_transform)
    """
    exclude_cols = exclude_cols or []
    cols_to_scale = [c for c in df.columns if c not in exclude_cols]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df_scaled, scaler


# ──────────────────────────────────────────────────────────────
# I/O HELPERS
# ──────────────────────────────────────────────────────────────

def save_dataframe(df, filename, subdir="data"):
    """Save a DataFrame to CSV in the output directory."""
    from config import OUTPUT_DIR
    path = os.path.join(OUTPUT_DIR, subdir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"[✓] Saved: {path}")
    return path


def load_dataframe(filename, subdir="data", parse_dates=True):
    """Load a DataFrame from CSV in the output directory."""
    from config import OUTPUT_DIR
    path = os.path.join(OUTPUT_DIR, subdir, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=parse_dates)
    print(f"[✓] Loaded: {path}")
    return df


# ──────────────────────────────────────────────────────────────
# QUICK SUMMARY STATS
# ──────────────────────────────────────────────────────────────

def summary_stats(df, name="Dataset"):
    """Print a concise summary of a DataFrame."""
    print(f"\n── {name} Summary ──")
    print(f"  Shape     : {df.shape}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  NaN counts:")
    nans = df.isna().sum()
    for col in nans[nans > 0].index:
        print(f"    {col}: {nans[col]}")
    if nans.sum() == 0:
        print(f"    (none)")
    print(f"  Columns   : {list(df.columns)}")
    return df.describe()