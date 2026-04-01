# eur_usd_pipeline/features/engineer.py
"""
Master feature engineering module.

Orchestrates all feature construction, merges everything into a single
analysis-ready DataFrame, and handles:

  1. Data fetching (calls all data/ modules)
  2. Technical indicator computation
  3. Spread and macro feature construction
  4. Sentiment integration
  5. Interaction features and non-linear transforms
  6. Lag structure for VAR / GLM models
  7. Final alignment, cleaning, and train/test split

This is the single entry point that produces the modelling dataset.
"""

import warnings
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import START_DATE, END_DATE, MODEL_CONFIG
from utils.helpers import (
    ensure_directories, align_series, make_stationary,
    clean_series, winsorize_dataframe, scale_features,
    save_dataframe, summary_stats,
)

# Data fetchers
from data.fetch_fx import fetch_all_fx
from data.fetch_yields import fetch_all_yields
from data.fetch_oil import fetch_all_oil
from data.fetch_cot import fetch_cot_data, print_positioning_summary
from data.fetch_risk_reversals import fetch_all_risk_reversals, print_rr_summary
from data.fetch_macro import fetch_all_macro, print_macro_summary

# Feature builders
from features.sentiment import fetch_all_sentiment
from features.technical import compute_all_technicals
from features.spreads import (
    build_yield_spread_features,
    build_tot_spread_features,
    build_carry_index,
    build_positioning_features,
    build_volatility_features,
    build_composite_macro_score,
)


# ──────────────────────────────────────────────────────────────
# INTERACTION & NON-LINEAR FEATURES
# ──────────────────────────────────────────────────────────────

def build_interaction_features(df):
    """
    Create interaction terms and non-linear transforms that
    capture relationships the GLM can exploit beyond what
    the linear VAR sees.

    Parameters
    ----------
    df : pd.DataFrame
        Merged feature DataFrame.

    Returns
    -------
    pd.DataFrame
        Original columns + new interaction columns.
    """
    print("[...] Building interaction and non-linear features")

    out = df.copy()
    n_before = len(out.columns)

    # ──────────────────────────────────────────────────────
    # 1. YIELD SPREAD × POSITIONING
    # When yield spread is narrowing AND specs are short,
    # the EUR rally potential is amplified (short squeeze
    # fuelled by fundamental shift).
    # ──────────────────────────────────────────────────────
    if "yld_spread_zscore" in out.columns and "pos_zscore" in out.columns:
        out["interact_yield_x_pos"] = (
            out["yld_spread_zscore"] * (-out["pos_zscore"])
        )

    # ──────────────────────────────────────────────────────
    # 2. OIL SHOCK × CARRY
    # Oil shocks that coincide with carry trade unwinds
    # can amplify EUR moves.
    # ──────────────────────────────────────────────────────
    if "oil_shock_cumulative" in out.columns and "carry_index" in out.columns:
        out["interact_oil_shock_x_carry"] = (
            out["oil_shock_cumulative"] * out["carry_index"]
        )

    # ──────────────────────────────────────────────────────
    # 3. RISK REVERSAL × MOMENTUM
    # When options market tilts bullish (RR > 0) and price
    # momentum confirms, the signal strengthens.
    # ──────────────────────────────────────────────────────
    if "rr_level" in out.columns and "eur_ma_cross_signal" in out.columns:
        out["interact_rr_x_momentum"] = (
            out["rr_level"] * out["eur_ma_cross_signal"]
        )

    # ──────────────────────────────────────────────────────
    # 4. SENTIMENT × POSITIONING
    # Bullish sentiment + short positioning = squeeze catalyst.
    # ──────────────────────────────────────────────────────
    if "proxy_sentiment" in out.columns and "pos_zscore" in out.columns:
        out["interact_sentiment_x_pos"] = (
            out["proxy_sentiment"] * (-out["pos_zscore"])
        )

    # ──────────────────────────────────────────────────────
    # 5. VOLATILITY REGIME INTERACTIONS
    # Some signals only matter in certain vol regimes.
    # ──────────────────────────────────────────────────────
    if "eur_rvol_13w" in out.columns:
        vol = out["eur_rvol_13w"]
        vol_median = vol.rolling(52, min_periods=26).median()

        # High vol flag (binary)
        high_vol = (vol > vol_median * 1.3).astype(float)

        if "yld_spread_chg_4w" in out.columns:
            out["interact_yield_chg_x_highvol"] = (
                out["yld_spread_chg_4w"] * high_vol
            )

        if "pos_contrarian_signal" in out.columns:
            out["interact_contrarian_x_highvol"] = (
                out["pos_contrarian_signal"] * high_vol
            )

    # ──────────────────────────────────────────────────────
    # 6. NON-LINEAR TRANSFORMS
    # Squared and cubed terms for key drivers to let the
    # GLM capture diminishing / accelerating effects.
    # ──────────────────────────────────────────────────────
    nonlinear_candidates = [
        "yld_spread_level", "carry_index", "pos_zscore",
        "proxy_sentiment", "rr_level",
    ]
    for col in nonlinear_candidates:
        if col in out.columns:
            out[f"{col}_sq"] = out[col] ** 2
            out[f"{col}_cb"] = out[col] ** 3

    # ──────────────────────────────────────────────────────
    # 7. CROSS-ASSET MOMENTUM CONSENSUS
    # Count how many momentum signals agree on EUR direction.
    # ──────────────────────────────────────────────────────
    momentum_cols = [c for c in out.columns if "momentum" in c.lower() and out[c].dtype in [np.float64, np.float32]]
    if len(momentum_cols) >= 3:
        # Binarise: positive = EUR bullish
        momentum_binary = out[momentum_cols].apply(lambda x: (x > 0).astype(float))
        out["momentum_consensus"] = momentum_binary.mean(axis=1)
        out["momentum_consensus_count"] = momentum_binary.sum(axis=1)

    n_after = len(out.columns)
    n_new = n_after - n_before
    print(f"[✓] Interaction features: {n_new} new columns added")

    return out


# ──────────────────────────────────────────────────────────────
# LAG FEATURE CONSTRUCTION
# ──────────────────────────────────────────────────────────────

def build_lag_features(df, target_col="eurusd_close", lags=None):
    """
    Create lagged versions of key features for the VAR/GLM models.

    Also creates forward returns (the prediction target).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        The EUR/USD price column.
    lags : list of int
        Lag periods in weeks (default: [1, 2, 4, 8, 13]).

    Returns
    -------
    pd.DataFrame
        With lag columns and forward return targets.
    """
    print("[...] Building lag features and forward targets")

    if lags is None:
        lags = [1, 2, 4, 8, 13]

    out = df.copy()
    n_before = len(out.columns)

    # ── Forward returns (TARGETS) ──
    if target_col in out.columns:
        price = out[target_col]
        out["target_return_4w"] = price.pct_change(4).shift(-4)    # 1-month forward
        out["target_return_13w"] = price.pct_change(13).shift(-13)  # 3-month forward
        out["target_return_26w"] = price.pct_change(26).shift(-26)  # 6-month forward
        out["target_return_52w"] = price.pct_change(52).shift(-52)  # 12-month forward

        # Binary direction targets
        out["target_direction_4w"] = np.where(out["target_return_4w"] > 0, 1, 0)
        out["target_direction_13w"] = np.where(out["target_return_13w"] > 0, 1, 0)

    # ── Lag key features ──
    lag_candidates = [
        "yld_spread_level", "yld_spread_chg_4w", "carry_index",
        "pos_zscore", "proxy_sentiment", "rr_level",
        "eur_rsi_14", "eur_macd_histogram", "tot_divergence",
        "macro_composite_score",
    ]

    for col in lag_candidates:
        if col in out.columns:
            for lag in lags:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)

    n_after = len(out.columns)
    n_new = n_after - n_before
    print(f"[✓] Lag features: {n_new} new columns ({len(lags)} lags × {len([c for c in lag_candidates if c in out.columns])} features + targets)")

    return out


# ──────────────────────────────────────────────────────────────
# FEATURE SELECTION / REDUCTION
# ──────────────────────────────────────────────────────────────

def select_features(df, min_coverage=0.7, max_correlation=0.95):
    """
    Automated feature selection:
      1. Drop columns with too many NaNs
      2. Drop constant / near-constant columns
      3. Drop highly correlated features (keep the first)

    Parameters
    ----------
    df : pd.DataFrame
    min_coverage : float
        Minimum fraction of non-NaN values required (0–1).
    max_correlation : float
        Maximum pairwise correlation allowed.

    Returns
    -------
    pd.DataFrame
        Reduced feature set.
    list
        Names of dropped columns.
    """
    print("[...] Running feature selection")

    dropped = []
    out = df.copy()

    # ── Step 1: Drop low-coverage columns ──
    coverage = out.notna().mean()
    low_cov = coverage[coverage < min_coverage].index.tolist()
    if low_cov:
        out = out.drop(columns=low_cov)
        dropped.extend(low_cov)
        print(f"  [·] Dropped {len(low_cov)} columns with < {min_coverage:.0%} coverage")

    # ── Step 2: Drop constant columns ──
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    constant = [c for c in numeric_cols if out[c].nunique() <= 1]
    if constant:
        out = out.drop(columns=constant)
        dropped.extend(constant)
        print(f"  [·] Dropped {len(constant)} constant columns")

    # ── Step 3: Drop highly correlated features ──
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        corr_matrix = out[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > max_correlation)]
        if to_drop:
            out = out.drop(columns=to_drop)
            dropped.extend(to_drop)
            print(f"  [·] Dropped {len(to_drop)} highly correlated columns (r > {max_correlation})")

    print(f"[✓] Feature selection: {len(df.columns)} → {len(out.columns)} columns ({len(dropped)} dropped)")

    return out, dropped


# ──────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────

def temporal_train_test_split(df, train_ratio=None):
    """
    Split time series data chronologically (no shuffling).

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    train_ratio : float
        Fraction for training (default from MODEL_CONFIG).

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    if train_ratio is None:
        train_ratio = MODEL_CONFIG.get("train_test_split", 0.85)

    n = len(df)
    split_idx = int(n * train_ratio)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"[✓] Train/Test split: {len(train)} / {len(test)} rows "
          f"(split at {df.index[split_idx].date()})")

    return train, test


# ──────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ──────────────────────────────────────────────────────────────

def run_feature_engineering(start=START_DATE, end=END_DATE, save=True):
    """
    Execute the full feature engineering pipeline.

    Steps:
      1. Fetch all raw data
      2. Build technical indicators
      3. Build spread / macro features
      4. Build sentiment features
      5. Merge everything
      6. Create interaction features
      7. Create lag features
      8. Feature selection
      9. Train/test split
      10. Save outputs

    Returns
    -------
    dict with keys:
        'full_df'       - Complete feature DataFrame (before split)
        'train_df'      - Training set
        'test_df'       - Test set
        'raw_data'      - Dict of raw data DataFrames
        'dropped_cols'  - Columns removed during selection
        'diff_record'   - Differencing record for stationarity
    """
    ensure_directories()

    print("\n" + "=" * 60)
    print("   EUR/USD FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print(f"   Period: {start} → {end}")
    print("=" * 60 + "\n")

    # ──────────────────────────────────────────────────────
    # STEP 1: FETCH RAW DATA
    # ──────────────────────────────────────────────────────
    print("\n── STEP 1: Fetching Raw Data ──\n")

    fx_data = fetch_all_fx(start=start, end=end, save=save)
    yield_data = fetch_all_yields(start=start, end=end, save=save)
    oil_data = fetch_all_oil(start=start, end=end, save=save)

    # COT data (may fail if CFTC is down)
    try:
        cot_data = fetch_cot_data(start=start, end=end, save=save)
        print_positioning_summary(cot_data)
    except Exception as e:
        print(f"  [!] COT data fetch failed: {e}")
        cot_data = pd.DataFrame()

    # Risk reversals (synthetic)
    try:
        rr_data = fetch_all_risk_reversals(start=start, end=end, save=save)
        print_rr_summary(rr_data)
    except Exception as e:
        print(f"  [!] Risk reversal construction failed: {e}")
        rr_data = pd.DataFrame()

    # Macro data
    try:
        macro_result = fetch_all_macro(start=start, end=end, save=save)
        print_macro_summary(macro_result)
        macro_data = macro_result.get("combined", pd.DataFrame())
    except Exception as e:
        print(f"  [!] Macro data fetch failed: {e}")
        macro_result = {}
        macro_data = pd.DataFrame()

    raw_data = {
        "fx": fx_data,
        "yields": yield_data,
        "oil": oil_data,
        "cot": cot_data,
        "risk_reversals": rr_data,
        "macro": macro_data,
    }

    # ──────────────────────────────────────────────────────
    # STEP 2: TECHNICAL INDICATORS
    # ──────────────────────────────────────────────────────
    print("\n── STEP 2: Technical Indicators ──\n")

    eur_technicals = compute_all_technicals(
        fx_data, price_col="eurusd_close", prefix="eur"
    )

    brent_technicals = pd.DataFrame()
    if "brent_close" in oil_data.columns:
        oil_for_tech = oil_data.copy()
        oil_for_tech["brent_return"] = oil_for_tech["brent_close"].pct_change()
        brent_technicals = compute_all_technicals(
            oil_for_tech, price_col="brent_close", prefix="brent"
        )

    # ──────────────────────────────────────────────────────
    # STEP 3: SPREAD & MACRO FEATURES
    # ──────────────────────────────────────────────────────
    print("\n── STEP 3: Spread & Macro Features ──\n")

    yield_features = build_yield_spread_features(yield_data)
    tot_features = build_tot_spread_features(oil_data, fx_data)
    carry_features = build_carry_index(yield_data, macro_data if not macro_data.empty else None)
    pos_features = build_positioning_features(cot_data)

    # VIX DataFrame for volatility features
    vix_df = fx_data[["vix_close"]] if "vix_close" in fx_data.columns else None
    vol_features = build_volatility_features(fx_data, vix_df=vix_df, rr_df=rr_data)

    composite = build_composite_macro_score(
        yield_features=yield_features,
        tot_features=tot_features,
        carry_features=carry_features,
        pos_features=pos_features,
        vol_features=vol_features,
    )

    # ──────────────────────────────────────────────────────
    # STEP 4: SENTIMENT FEATURES
    # ──────────────────────────────────────────────────────
    print("\n── STEP 4: Sentiment Features ──\n")

    try:
        sentiment_result = fetch_all_sentiment(
            eurusd_df=fx_data,
            vix_df=vix_df,
            save=save,
        )
        sentiment_data = sentiment_result.get("combined", pd.DataFrame())
    except Exception as e:
        print(f"  [!] Sentiment pipeline failed: {e}")
        sentiment_data = pd.DataFrame()

    # ──────────────────────────────────────────────────────
    # STEP 5: MERGE EVERYTHING
    # ──────────────────────────────────────────────────────
    print("\n── STEP 5: Merging All Features ──\n")

    # Start with FX as the base (most complete time series)
    merged = fx_data.copy()

    # Join each feature set
    feature_sets = {
        "yields": yield_data,
        "oil": oil_data[["brent_close", "brent_inverted", "brent_return"]],
        "eur_technicals": eur_technicals,
        "brent_technicals": brent_technicals,
        "yield_features": yield_features,
        "tot_features": tot_features,
        "carry_features": carry_features,
        "pos_features": pos_features,
        "vol_features": vol_features,
        "composite": composite,
        "sentiment": sentiment_data,
    }

    # Add macro data (monthly → weekly via ffill)
    if not macro_data.empty:
        macro_weekly = macro_data.resample("W-FRI").ffill()
        feature_sets["macro"] = macro_weekly

    # Add COT (already weekly, but dates may not align perfectly)
    if not cot_data.empty:
        cot_cols = [c for c in cot_data.columns if c not in merged.columns]
        if cot_cols:
            feature_sets["cot_extra"] = cot_data[cot_cols]

    # Add risk reversals
    if not rr_data.empty:
        rr_cols = [c for c in rr_data.select_dtypes(include=[np.number]).columns
                   if c not in merged.columns]
        if rr_cols:
            feature_sets["rr_extra"] = rr_data[rr_cols]

    for name, feat_df in feature_sets.items():
        if feat_df is not None and not feat_df.empty:
            # Avoid duplicate columns
            new_cols = [c for c in feat_df.columns if c not in merged.columns]
            if new_cols:
                merged = merged.join(feat_df[new_cols], how="left")
                print(f"  [+] Merged '{name}': {len(new_cols)} columns")

    # Forward-fill gaps from different frequencies (max 4 weeks)
    merged = merged.ffill(limit=4)

    print(f"\n[✓] Merged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")

    # ──────────────────────────────────────────────────────
    # STEP 6: INTERACTION FEATURES
    # ──────────────────────────────────────────────────────
    print("\n── STEP 6: Interaction Features ──\n")

    merged = build_interaction_features(merged)

    # ──────────────────────────────────────────────────────
    # STEP 7: LAG FEATURES
    # ──────────────────────────────────────────────────────
    print("\n── STEP 7: Lag Features ──\n")

    merged = build_lag_features(merged, target_col="eurusd_close")

    # ──────────────────────────────────────────────────────
    # STEP 8: FEATURE SELECTION
    # ──────────────────────────────────────────────────────
    print("\n── STEP 8: Feature Selection ──\n")

    # Separate non-numeric columns before selection
    non_numeric_cols = merged.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric = merged[non_numeric_cols] if non_numeric_cols else pd.DataFrame()
    numeric = merged.select_dtypes(include=[np.number])

    numeric_selected, dropped_cols = select_features(
        numeric, min_coverage=0.6, max_correlation=0.95
    )

    # Winsorize to handle remaining outliers
    numeric_selected = winsorize_dataframe(numeric_selected, lower=0.01, upper=0.99)

    # Recombine with non-numeric
    if not non_numeric.empty:
        full_df = numeric_selected.join(non_numeric, how="left")
    else:
        full_df = numeric_selected

    # ──────────────────────────────────────────────────────
    # STEP 9: TRAIN / TEST SPLIT
    # ──────────────────────────────────────────────────────
    print("\n── STEP 9: Train/Test Split ──\n")

    # Drop rows where the primary target is NaN
    modelling_df = full_df.dropna(subset=["eurusd_close"])

    train_df, test_df = temporal_train_test_split(modelling_df)

    # ──────────────────────────────────────────────────────
    # STEP 10: SAVE
    # ──────────────────────────────────────────────────────
    print("\n── STEP 10: Saving Outputs ──\n")

    if save:
        save_dataframe(full_df, "full_features.csv")
        save_dataframe(train_df, "train_features.csv")
        save_dataframe(test_df, "test_features.csv")

    # Summary
    summary_stats(full_df, "Full Feature Dataset")

    print("\n" + "=" * 60)
    print("   FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"   Total features:  {len(numeric_selected.columns)}")
    print(f"   Training rows:   {len(train_df)}")
    print(f"   Test rows:       {len(test_df)}")
    print(f"   Date range:      {full_df.index.min().date()} → {full_df.index.max().date()}")
    print(f"   Dropped columns: {len(dropped_cols)}")
    print("=" * 60 + "\n")

    return {
        "full_df": full_df,
        "train_df": train_df,
        "test_df": test_df,
        "raw_data": raw_data,
        "dropped_cols": dropped_cols,
    }


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_feature_engineering()
    print("\nFull feature set columns:")
    for i, col in enumerate(result["full_df"].columns):
        print(f"  {i+1:3d}. {col}")