# eur_usd_pipeline/features/spreads.py
"""
Spread and relative-value feature construction.

Builds cross-asset spread features that capture the macro drivers
of EUR/USD from multiple angles:

  1. Yield spreads (2Y, real rate differentials)
  2. Terms of Trade spreads (oil-adjusted)
  3. Policy rate spreads (Fed vs ECB)
  4. Growth differentials (PMI-based)
  5. Positioning spreads (COT extremes)
  6. Volatility spreads (EUR vs USD implied/realised)
  7. Carry trade attractiveness index

These features are the primary inputs for the VAR and GLM models.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# YIELD SPREAD FEATURES
# ──────────────────────────────────────────────────────────────

def build_yield_spread_features(yield_df, window_short=13, window_long=52):
    """
    Construct yield-spread-derived features for modelling.

    Parameters
    ----------
    yield_df : pd.DataFrame
        Must contain 'yield_spread_2y' from fetch_yields.
    window_short : int
        Short rolling window (weeks).
    window_long : int
        Long rolling window (weeks).

    Returns
    -------
    pd.DataFrame
        Yield spread features.
    """
    print("[...] Building yield spread features")

    if "yield_spread_2y" not in yield_df.columns:
        print("  [!] 'yield_spread_2y' not found — skipping yield spread features")
        return pd.DataFrame()

    df = pd.DataFrame(index=yield_df.index)
    spread = yield_df["yield_spread_2y"]

    # ── Level and changes ──
    df["yld_spread_level"] = spread
    df["yld_spread_chg_1w"] = spread.diff(1)
    df["yld_spread_chg_4w"] = spread.diff(4)
    df["yld_spread_chg_13w"] = spread.diff(13)

    # ── Z-score (how extreme is the current spread?) ──
    roll_mean = spread.rolling(window_long, min_periods=window_short).mean()
    roll_std = spread.rolling(window_long, min_periods=window_short).std()
    df["yld_spread_zscore"] = (spread - roll_mean) / roll_std

    # ── Momentum (is the spread narrowing or widening?) ──
    # SMA crossover: short-term avg vs long-term avg
    sma_short = spread.rolling(window_short).mean()
    sma_long = spread.rolling(window_long).mean()
    df["yld_spread_momentum"] = sma_short - sma_long

    # Direction signal: narrowing = EUR positive
    df["yld_spread_narrowing"] = np.where(df["yld_spread_chg_4w"] > 0, 1, 0)

    # ── Rate of convergence ──
    # How fast is the spread moving toward zero?
    df["yld_spread_convergence_speed"] = spread.diff(4) / spread.shift(4).abs().replace(0, np.nan)

    # ── Regime detection ──
    # Is the spread in a historically wide or narrow range?
    pct_rank = spread.rolling(window_long * 2, min_periods=window_long).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["yld_spread_percentile"] = pct_rank * 100

    df["yld_spread_regime"] = pd.cut(
        df["yld_spread_percentile"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["VERY_NARROW", "NARROW", "NEUTRAL", "WIDE", "VERY_WIDE"],
        include_lowest=True,
    )

    # ── Acceleration (second derivative) ──
    df["yld_spread_acceleration"] = df["yld_spread_chg_1w"].diff()

    print(f"[✓] Yield spread features: {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────────────────────
# TERMS OF TRADE SPREAD FEATURES
# ──────────────────────────────────────────────────────────────

def build_tot_spread_features(oil_df, eurusd_df, window=52):
    """
    Build Terms of Trade features linking oil to EUR/USD.

    Core hypothesis: falling oil → improved Eurozone ToT → EUR positive.

    Parameters
    ----------
    oil_df : pd.DataFrame
        Must contain 'brent_close' and 'brent_inverted'.
    eurusd_df : pd.DataFrame
        Must contain 'eurusd_close'.
    window : int
        Rolling window for correlation and z-scores.

    Returns
    -------
    pd.DataFrame
    """
    print("[...] Building Terms of Trade spread features")

    merged = eurusd_df.join(oil_df, how="inner")
    df = pd.DataFrame(index=merged.index)

    if "brent_close" not in merged.columns or "eurusd_close" not in merged.columns:
        print("  [!] Missing required columns for ToT features")
        return pd.DataFrame()

    brent = merged["brent_close"]
    eur = merged["eurusd_close"]

    # ── Oil-EUR rolling correlation ──
    df["tot_corr_26w"] = brent.rolling(26, min_periods=13).corr(eur)
    df["tot_corr_52w"] = brent.rolling(52, min_periods=26).corr(eur)

    # ── Inverted oil z-score (higher = lower oil = EUR positive) ──
    if "brent_inverted" in merged.columns:
        inv = merged["brent_inverted"]
        df["tot_inv_zscore"] = (
            (inv - inv.rolling(window).mean()) / inv.rolling(window).std()
        )

    # ── Oil-FX divergence (when they decouple, reversion is likely) ──
    # Normalise both to [0, 1] range over rolling window
    brent_norm = (brent - brent.rolling(window).min()) / (
        brent.rolling(window).max() - brent.rolling(window).min()
    ).replace(0, np.nan)

    eur_norm = (eur - eur.rolling(window).min()) / (
        eur.rolling(window).max() - eur.rolling(window).min()
    ).replace(0, np.nan)

    # Divergence: if oil is high (norm ~1) and EUR is high (norm ~1),
    # something is off. We expect inverse relationship.
    # Score: (1 - brent_norm) - eur_norm → positive = EUR should rise
    df["tot_divergence"] = (1 - brent_norm) - eur_norm

    # ── Oil momentum relative to EUR momentum ──
    brent_mom = brent.pct_change(13)
    eur_mom = eur.pct_change(13)
    df["tot_relative_momentum"] = -brent_mom - eur_mom  # Negative oil + positive EUR = aligned

    # ── Energy shock detector ──
    brent_ret = brent.pct_change()
    brent_vol = brent_ret.rolling(52).std()
    df["oil_shock_flag"] = np.where(
        brent_ret.abs() > 2 * brent_vol, 1, 0
    )

    # Cumulative shock impact (decays over 13 weeks)
    df["oil_shock_cumulative"] = df["oil_shock_flag"].rolling(13).sum()

    print(f"[✓] ToT spread features: {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────────────────────
# CARRY TRADE ATTRACTIVENESS INDEX
# ──────────────────────────────────────────────────────────────

def build_carry_index(yield_df, macro_df=None, window=52):
    """
    Build a carry trade attractiveness index.

    Carry trade logic: borrow in low-yield currency, invest in high-yield.
    When US yields are higher, capital flows to USD → EUR negative.
    When the differential narrows, carry unwinds → EUR positive.

    This index combines:
    - Yield differential (level + direction)
    - Real rate differential (adjusted for inflation)
    - Risk-adjusted carry (yield spread / volatility)

    Parameters
    ----------
    yield_df : pd.DataFrame
        Must have 'us_2y_yield', 'de_2y_yield', 'yield_spread_2y'.
    macro_df : pd.DataFrame, optional
        If provided and has 'real_rate_diff', incorporates real rates.
    window : int
        Rolling window.

    Returns
    -------
    pd.DataFrame
        Columns: ['carry_index', 'carry_direction', 'carry_zscore']
    """
    print("[...] Building carry trade attractiveness index")

    df = pd.DataFrame(index=yield_df.index)

    if "yield_spread_2y" not in yield_df.columns:
        print("  [!] Yield spread not available for carry index")
        return pd.DataFrame()

    spread = yield_df["yield_spread_2y"]

    # ── Component 1: Nominal spread direction ──
    # Normalise to [-1, 1] range
    spread_z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    carry_nominal = spread_z.clip(-3, 3) / 3  # Scale to [-1, 1]

    # ── Component 2: Spread momentum ──
    spread_mom = spread.diff(13)
    mom_z = (spread_mom - spread_mom.rolling(window).mean()) / spread_mom.rolling(window).std()
    carry_momentum = mom_z.clip(-3, 3) / 3

    # ── Component 3: Real rate differential (if available) ──
    carry_real = pd.Series(0, index=yield_df.index)
    if macro_df is not None and "real_rate_diff" in macro_df.columns:
        # Resample macro to match yield frequency
        real_rate = macro_df["real_rate_diff"].reindex(yield_df.index, method="ffill")
        rr_z = (real_rate - real_rate.rolling(window, min_periods=26).mean()) / \
               real_rate.rolling(window, min_periods=26).std()
        carry_real = rr_z.clip(-3, 3) / 3

    # ── Composite carry index ──
    # Weights: 40% nominal level, 30% momentum, 30% real rate
    df["carry_index"] = (
        0.40 * carry_nominal.fillna(0)
        + 0.30 * carry_momentum.fillna(0)
        + 0.30 * carry_real.fillna(0)
    )

    # Direction: positive = carry favouring EUR, negative = favouring USD
    df["carry_direction"] = np.where(df["carry_index"] > 0.1, "EUR_FAVOURABLE",
                            np.where(df["carry_index"] < -0.1, "USD_FAVOURABLE", "NEUTRAL"))

    # Z-score of carry index itself
    ci_mean = df["carry_index"].rolling(window, min_periods=26).mean()
    ci_std = df["carry_index"].rolling(window, min_periods=26).std()
    df["carry_zscore"] = (df["carry_index"] - ci_mean) / ci_std

    print(f"[✓] Carry index features: {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────────────────────
# POSITIONING SPREAD FEATURES
# ──────────────────────────────────────────────────────────────

def build_positioning_features(cot_df, window=104):
    """
    Build positioning-derived features from COT data.

    Parameters
    ----------
    cot_df : pd.DataFrame
        From fetch_cot_data().
    window : int
        Rolling window for percentile / z-score (default 2 years).

    Returns
    -------
    pd.DataFrame
    """
    print("[...] Building positioning spread features")

    if cot_df.empty or "net_spec_position" not in cot_df.columns:
        print("  [!] COT data missing or incomplete")
        return pd.DataFrame()

    df = pd.DataFrame(index=cot_df.index)
    spec = cot_df["net_spec_position"]

    # ── Level and changes ──
    df["pos_net_spec"] = spec
    df["pos_change_1w"] = spec.diff(1)
    df["pos_change_4w"] = spec.diff(4)

    # ── Z-score ──
    roll_mean = spec.rolling(window, min_periods=52).mean()
    roll_std = spec.rolling(window, min_periods=52).std()
    df["pos_zscore"] = (spec - roll_mean) / roll_std

    # ── Percentile rank ──
    df["pos_percentile"] = spec.rolling(window, min_periods=52).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ) * 100

    # ── Extreme positioning flags ──
    df["pos_extreme_long"] = np.where(df["pos_zscore"] > 1.5, 1, 0)
    df["pos_extreme_short"] = np.where(df["pos_zscore"] < -1.5, 1, 0)

    # ── Positioning momentum (are specs adding or reducing?) ──
    df["pos_momentum_4w"] = spec.diff(4) / spec.shift(4).abs().replace(0, np.nan)

    # ── Short squeeze score ──
    # Combines how short the market is + whether covering has started
    if "crowding_zscore" in cot_df.columns:
        df["squeeze_score"] = -cot_df["crowding_zscore"] * np.where(
            spec.diff(1) > 0, 1.5, 1.0  # Amplify if already covering
        )
        df["squeeze_score"] = df["squeeze_score"].clip(-5, 5)

    # ── Contrarian signal ──
    # Extreme positioning often precedes reversals
    df["pos_contrarian_signal"] = -df["pos_zscore"]  # Fade the crowd

    # ── Dealer vs spec divergence ──
    if "net_dealer_position" in cot_df.columns:
        dealer = pd.to_numeric(cot_df["net_dealer_position"], errors="coerce")
        df["dealer_spec_divergence"] = dealer - spec
        # Large positive → dealers long while specs short → bullish signal
        div_z = (df["dealer_spec_divergence"] - df["dealer_spec_divergence"].rolling(window, min_periods=52).mean()) / \
                df["dealer_spec_divergence"].rolling(window, min_periods=52).std()
        df["dealer_spec_zscore"] = div_z

    print(f"[✓] Positioning features: {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────────────────────
# VOLATILITY SPREAD FEATURES
# ──────────────────────────────────────────────────────────────

def build_volatility_features(eurusd_df, vix_df=None, rr_df=None, window=52):
    """
    Build volatility-based features.

    Parameters
    ----------
    eurusd_df : pd.DataFrame
        Must have 'eurusd_return'.
    vix_df : pd.DataFrame, optional
        Must have 'vix_close'.
    rr_df : pd.DataFrame, optional
        Risk reversal data with 'rr_25d_proxy'.
    window : int
        Rolling window.

    Returns
    -------
    pd.DataFrame
    """
    print("[...] Building volatility spread features")

    df = pd.DataFrame(index=eurusd_df.index)

    if "eurusd_return" in eurusd_df.columns:
        ret = eurusd_df["eurusd_return"]

        # ── Realised vol (annualised) ──
        df["eur_rvol_13w"] = ret.rolling(13).std() * np.sqrt(52)
        df["eur_rvol_26w"] = ret.rolling(26).std() * np.sqrt(52)

        # ── Vol regime ──
        vol_median = df["eur_rvol_13w"].rolling(window, min_periods=26).median()
        df["vol_regime"] = np.where(
            df["eur_rvol_13w"] > vol_median * 1.5, "HIGH_VOL",
            np.where(df["eur_rvol_13w"] < vol_median * 0.6, "LOW_VOL", "NORMAL_VOL")
        )

        # ── Vol term structure (short vs long) ──
        df["vol_term_structure"] = df["eur_rvol_13w"] - df["eur_rvol_26w"]
        # Inverted = short vol > long vol = risk-off signal

    # ── VIX integration ──
    if vix_df is not None and "vix_close" in vix_df.columns:
        vix = vix_df["vix_close"].reindex(eurusd_df.index, method="ffill")
        df["vix_level"] = vix
        df["vix_zscore"] = (vix - vix.rolling(window).mean()) / vix.rolling(window).std()

        # VIX-EUR vol spread: when VIX spikes but EUR vol doesn't,
        # EUR may be about to catch up
        if "eur_rvol_13w" in df.columns:
            vix_scaled = vix / 100  # Scale VIX to comparable range
            df["vix_eur_vol_spread"] = vix_scaled - df["eur_rvol_13w"]

    # ── Risk reversal integration ──
    if rr_df is not None and "rr_25d_proxy" in rr_df.columns:
        rr = rr_df["rr_25d_proxy"].reindex(eurusd_df.index, method="ffill")
        df["rr_level"] = rr
        df["rr_zscore"] = (rr - rr.rolling(window, min_periods=26).mean()) / \
                          rr.rolling(window, min_periods=26).std()
        df["rr_momentum_4w"] = rr.diff(4)

    print(f"[✓] Volatility features: {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────────────────────
# COMPOSITE MACRO SCORE
# ──────────────────────────────────────────────────────────────

def build_composite_macro_score(yield_features=None, tot_features=None,
                                 carry_features=None, pos_features=None,
                                 vol_features=None):
    """
    Combine all spread features into a single composite macro score
    that summarises whether conditions favour EUR appreciation.

    Score > 0 → conditions favour EUR
    Score < 0 → conditions favour USD

    Each component is z-scored and equally weighted.

    Returns
    -------
    pd.DataFrame
        Columns: ['macro_composite_score', 'macro_signal', 'macro_confidence']
    """
    print("[...] Building composite macro score")

    components = {}

    # ── Yield spread component ──
    if yield_features is not None and "yld_spread_zscore" in yield_features.columns:
        # Positive z-score = spread narrowing = EUR positive
        components["yield"] = yield_features["yld_spread_zscore"]

    # ── ToT component ──
    if tot_features is not None and "tot_divergence" in tot_features.columns:
        components["tot"] = tot_features["tot_divergence"]

    # ── Carry component ──
    if carry_features is not None and "carry_index" in carry_features.columns:
        components["carry"] = carry_features["carry_index"]

    # ── Positioning (contrarian) component ──
    if pos_features is not None and "pos_contrarian_signal" in pos_features.columns:
        components["positioning"] = pos_features["pos_contrarian_signal"]

    # ── Vol / RR component ──
    if vol_features is not None and "rr_level" in vol_features.columns:
        components["vol_rr"] = vol_features["rr_level"]

    if not components:
        print("  [!] No components available for composite score")
        return pd.DataFrame()

    # ── Align all components ──
    comp_df = pd.DataFrame(components)
    comp_df = comp_df.ffill(limit=4).dropna(how="all")

    # Z-score each component
    for col in comp_df.columns:
        mean = comp_df[col].rolling(104, min_periods=52).mean()
        std = comp_df[col].rolling(104, min_periods=52).std()
        comp_df[col] = (comp_df[col] - mean) / std

    # ── Equal-weighted composite ──
    result = pd.DataFrame(index=comp_df.index)
    result["macro_composite_score"] = comp_df.mean(axis=1)

    # ── Signal ──
    result["macro_signal"] = np.where(
        result["macro_composite_score"] > 0.5, "STRONG_EUR_BUY",
        np.where(result["macro_composite_score"] > 0, "LEAN_EUR_BUY",
        np.where(result["macro_composite_score"] > -0.5, "LEAN_EUR_SELL",
        "STRONG_EUR_SELL"))
    )

    # ── Confidence (how many components agree?) ──
    n_positive = (comp_df > 0).sum(axis=1)
    n_total = comp_df.notna().sum(axis=1)
    result["macro_confidence"] = n_positive / n_total  # 0 to 1

    # ── Individual component contributions ──
    for col in comp_df.columns:
        result[f"component_{col}"] = comp_df[col]

    print(f"[✓] Composite macro score: {len(result)} rows, {len(components)} components")

    return result


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Spread features module loaded. Run from main.py for full pipeline.")
    print(f"Available builders:")
    print(f"  - build_yield_spread_features()")
    print(f"  - build_tot_spread_features()")
    print(f"  - build_carry_index()")
    print(f"  - build_positioning_features()")
    print(f"  - build_volatility_features()")
    print(f"  - build_composite_macro_score()")