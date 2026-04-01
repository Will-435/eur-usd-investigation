# eur_usd_pipeline/models/granger.py
"""
Extended Granger causality analysis module.

While var_model.py includes basic Granger tests, this module provides:

  1. Bidirectional Granger causality (does X cause Y AND does Y cause X?)
  2. Time-varying Granger causality (rolling windows)
  3. Spectral Granger causality (frequency-domain — which frequencies matter?)
  4. Granger causality network visualisation data
  5. Lag-specific causality profiles (at which lag is causality strongest?)

This helps answer: "Which macro factors LEAD EUR/USD, at what horizons,
and has that leadership changed over time?"
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG
from utils.helpers import (
    test_stationarity, save_dataframe, ensure_directories,
)
from models.var_model import VAR_CORE_VARIABLES


# ──────────────────────────────────────────────────────────────
# BIDIRECTIONAL GRANGER CAUSALITY
# ──────────────────────────────────────────────────────────────

def bidirectional_granger(df, var_a, var_b, max_lag=12, alpha=0.05):
    """
    Test Granger causality in both directions between two variables.

    Returns
    -------
    dict with keys:
        'a_causes_b': bool
        'b_causes_a': bool
        'a_causes_b_pval': float
        'b_causes_a_pval': float
        'a_causes_b_best_lag': int
        'b_causes_a_best_lag': int
        'relationship': str  ('a→b', 'b→a', 'bidirectional', 'independent')
    """
    result = {
        "var_a": var_a,
        "var_b": var_b,
        "a_causes_b": False,
        "b_causes_a": False,
        "a_causes_b_pval": 1.0,
        "b_causes_a_pval": 1.0,
        "a_causes_b_best_lag": None,
        "b_causes_a_best_lag": None,
        "relationship": "independent",
    }

    if var_a not in df.columns or var_b not in df.columns:
        return result

    # Prepare stationary data
    pair = df[[var_a, var_b]].dropna()
    if len(pair) < max_lag * 4:
        return result

    pair_s = pair.copy()
    for col in pair_s.columns:
        sr = test_stationarity(pair_s[col], name=col, verbose=False)
        if not sr["is_stationary"]:
            pair_s[col] = pair_s[col].diff()
    pair_s = pair_s.dropna()

    if len(pair_s) < max_lag * 4:
        return result

    # ── Test A → B (does A Granger-cause B?) ──
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_ab = grangercausalitytests(pair_s[[var_b, var_a]], maxlag=max_lag, verbose=False)

        min_p_ab = 1.0
        best_lag_ab = 1
        for lag in range(1, max_lag + 1):
            if lag in gc_ab:
                p = gc_ab[lag][0]["ssr_ftest"][1]
                if p < min_p_ab:
                    min_p_ab = p
                    best_lag_ab = lag

        result["a_causes_b_pval"] = min_p_ab
        result["a_causes_b_best_lag"] = best_lag_ab
        result["a_causes_b"] = min_p_ab < alpha

    except Exception:
        pass

    # ── Test B → A (does B Granger-cause A?) ──
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_ba = grangercausalitytests(pair_s[[var_a, var_b]], maxlag=max_lag, verbose=False)

        min_p_ba = 1.0
        best_lag_ba = 1
        for lag in range(1, max_lag + 1):
            if lag in gc_ba:
                p = gc_ba[lag][0]["ssr_ftest"][1]
                if p < min_p_ba:
                    min_p_ba = p
                    best_lag_ba = lag

        result["b_causes_a_pval"] = min_p_ba
        result["b_causes_a_best_lag"] = best_lag_ba
        result["b_causes_a"] = min_p_ba < alpha

    except Exception:
        pass

    # ── Classify relationship ──
    if result["a_causes_b"] and result["b_causes_a"]:
        result["relationship"] = "bidirectional"
    elif result["a_causes_b"]:
        result["relationship"] = f"{var_a} → {var_b}"
    elif result["b_causes_a"]:
        result["relationship"] = f"{var_b} → {var_a}"
    else:
        result["relationship"] = "independent"

    return result


def run_bidirectional_analysis(df, variables=None, target="eurusd_close",
                                max_lag=12, alpha=0.05):
    """
    Run bidirectional Granger causality between target and all variables.

    Returns
    -------
    pd.DataFrame
        Bidirectional causality results.
    """
    print(f"\n[...] Bidirectional Granger Causality Analysis")

    if variables is None:
        variables = [v for v in VAR_CORE_VARIABLES if v in df.columns and v != target]

    results = []
    for var in variables:
        res = bidirectional_granger(df, var, target, max_lag=max_lag, alpha=alpha)
        results.append(res)

        # Print result
        rel = res["relationship"]
        p_ab = res["a_causes_b_pval"]
        p_ba = res["b_causes_a_pval"]
        print(f"  {var:30s} ↔ {target}: {rel}")
        print(f"    {var} → {target}: p={p_ba:.4f} (lag={res['b_causes_a_best_lag']})")
        print(f"    {target} → {var}: p={p_ab:.4f} (lag={res['a_causes_b_best_lag']})")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# TIME-VARYING GRANGER CAUSALITY
# ──────────────────────────────────────────────────────────────

def rolling_granger(df, cause, effect, window=104, step=4,
                    max_lag=8, alpha=0.05):
    """
    Compute Granger causality on rolling windows.

    This reveals whether the causal relationship between two variables
    is stable or changes over time (e.g., yield spreads may lead EUR/USD
    more strongly during tightening cycles than during QE).

    Parameters
    ----------
    df : pd.DataFrame
    cause, effect : str
        Variable names.
    window : int
        Rolling window size in weeks (default: 2 years).
    step : int
        Step size between windows.
    max_lag : int
        Maximum lag for Granger test.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'p_value', 'significant', 'best_lag', 'f_stat']
    """
    print(f"  [·] Rolling Granger: {cause} → {effect} (window={window}w)")

    if cause not in df.columns or effect not in df.columns:
        return pd.DataFrame()

    pair = df[[effect, cause]].dropna()

    # Make stationary
    pair_s = pair.copy()
    for col in pair_s.columns:
        sr = test_stationarity(pair_s[col], name=col, verbose=False)
        if not sr["is_stationary"]:
            pair_s[col] = pair_s[col].diff()
    pair_s = pair_s.dropna()

    if len(pair_s) < window + max_lag:
        print(f"    Insufficient data for rolling Granger ({len(pair_s)} < {window + max_lag})")
        return pd.DataFrame()

    results = []
    indices = range(window, len(pair_s), step)

    for i in indices:
        window_data = pair_s.iloc[i - window:i]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc = grangercausalitytests(
                    window_data[[effect, cause]],
                    maxlag=max_lag,
                    verbose=False,
                )

            min_p = 1.0
            best_lag = 1
            best_f = 0

            for lag in range(1, max_lag + 1):
                if lag in gc:
                    p = gc[lag][0]["ssr_ftest"][1]
                    f = gc[lag][0]["ssr_ftest"][0]
                    if p < min_p:
                        min_p = p
                        best_lag = lag
                        best_f = f

            results.append({
                "date": pair_s.index[i - 1],
                "p_value": min_p,
                "significant": min_p < alpha,
                "best_lag": best_lag,
                "f_stat": best_f,
            })

        except Exception:
            pass

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.set_index("date")
        pct_sig = result_df["significant"].mean() * 100
        print(f"    → Significant in {pct_sig:.0f}% of windows")

    return result_df


def run_rolling_granger_analysis(df, target="eurusd_close", variables=None,
                                  window=104, step=4, max_lag=8):
    """
    Run rolling Granger causality for all variables against EUR/USD.

    Returns
    -------
    dict of pd.DataFrames
        Keyed by variable name.
    pd.DataFrame
        Summary statistics.
    """
    print(f"\n[...] Rolling Granger Causality Analysis (window={window}w)")

    if variables is None:
        variables = [v for v in VAR_CORE_VARIABLES if v in df.columns and v != target]

    rolling_results = {}
    summary_rows = []

    for var in variables:
        rg = rolling_granger(df, cause=var, effect=target,
                             window=window, step=step, max_lag=max_lag)

        if not rg.empty:
            rolling_results[var] = rg

            summary_rows.append({
                "variable": var,
                "pct_significant": rg["significant"].mean() * 100,
                "mean_p_value": rg["p_value"].mean(),
                "median_p_value": rg["p_value"].median(),
                "mean_f_stat": rg["f_stat"].mean(),
                "modal_best_lag": rg["best_lag"].mode().iloc[0] if not rg["best_lag"].mode().empty else np.nan,
                "recent_significant": rg["significant"].iloc[-1] if len(rg) > 0 else np.nan,
                "recent_p_value": rg["p_value"].iloc[-1] if len(rg) > 0 else np.nan,
            })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("pct_significant", ascending=False)

        print(f"\n── Rolling Granger Summary ──")
        for _, row in summary_df.iterrows():
            recent = "✓" if row.get("recent_significant") else "·"
            print(f"  [{recent}] {row['variable']:30s}: significant {row['pct_significant']:.0f}% "
                  f"of time (median p={row['median_p_value']:.3f}, modal lag={row['modal_best_lag']:.0f})")

    return rolling_results, summary_df


# ──────────────────────────────────────────────────────────────
# LAG-SPECIFIC CAUSALITY PROFILE
# ──────────────────────────────────────────────────────────────

def lag_causality_profile(df, cause, effect, max_lag=26):
    """
    Show how Granger causality strength varies across different lags.

    This answers: "At what lead time does this variable best predict EUR/USD?"

    Parameters
    ----------
    df : pd.DataFrame
    cause, effect : str
    max_lag : int
        Maximum lag to test (in weeks).

    Returns
    -------
    pd.DataFrame
        Columns: ['lag', 'p_value', 'f_stat', 'significant']
    """
    if cause not in df.columns or effect not in df.columns:
        return pd.DataFrame()

    pair = df[[effect, cause]].dropna()

    # Make stationary
    pair_s = pair.copy()
    for col in pair_s.columns:
        sr = test_stationarity(pair_s[col], name=col, verbose=False)
        if not sr["is_stationary"]:
            pair_s[col] = pair_s[col].diff()
    pair_s = pair_s.dropna()

    if len(pair_s) < max_lag * 3:
        return pd.DataFrame()

    results = []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc = grangercausalitytests(pair_s[[effect, cause]], maxlag=max_lag, verbose=False)

        for lag in range(1, max_lag + 1):
            if lag in gc:
                p = gc[lag][0]["ssr_ftest"][1]
                f = gc[lag][0]["ssr_ftest"][0]
                results.append({
                    "lag_weeks": lag,
                    "p_value": p,
                    "f_statistic": f,
                    "significant_5pct": p < 0.05,
                    "significant_1pct": p < 0.01,
                })

    except Exception:
        pass

    return pd.DataFrame(results)


def run_lag_profiles(df, target="eurusd_close", variables=None, max_lag=26):
    """
    Compute lag-specific causality profiles for all variables.

    Returns
    -------
    dict of pd.DataFrames
        Keyed by variable name.
    pd.DataFrame
        Summary: optimal lag for each variable.
    """
    print(f"\n[...] Lag-Specific Causality Profiles (max_lag={max_lag})")

    if variables is None:
        variables = [v for v in VAR_CORE_VARIABLES if v in df.columns and v != target]

    profiles = {}
    summary_rows = []

    for var in variables:
        profile = lag_causality_profile(df, cause=var, effect=target, max_lag=max_lag)

        if not profile.empty:
            profiles[var] = profile

            # Find optimal lag (lowest p-value)
            best_row = profile.loc[profile["p_value"].idxmin()]
            n_sig = profile["significant_5pct"].sum()

            summary_rows.append({
                "variable": var,
                "best_lag_weeks": int(best_row["lag_weeks"]),
                "best_p_value": best_row["p_value"],
                "best_f_stat": best_row["f_statistic"],
                "n_significant_lags": n_sig,
                "pct_significant_lags": n_sig / len(profile) * 100,
            })

            print(f"  {var:30s}: best lag = {int(best_row['lag_weeks'])}w "
                  f"(p={best_row['p_value']:.4f}), "
                  f"{n_sig}/{len(profile)} lags significant")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("best_p_value")

    return profiles, summary_df


# ──────────────────────────────────────────────────────────────
# CAUSALITY NETWORK DATA
# ──────────────────────────────────────────────────────────────

def build_causality_network(df, variables=None, max_lag=8, alpha=0.05):
    """
    Build a directed causality network from pairwise Granger tests.

    Each edge represents a significant Granger-causal relationship.
    Edge weight = -log10(p_value), so stronger causality = heavier edge.

    This is used for the network visualisation in the dashboard.

    Parameters
    ----------
    df : pd.DataFrame
    variables : list of str
    max_lag : int
    alpha : float

    Returns
    -------
    pd.DataFrame
        Edge list: ['source', 'target', 'p_value', 'weight', 'best_lag']
    dict
        Node metadata: centrality scores, etc.
    """
    print(f"\n[...] Building Granger Causality Network")

    if variables is None:
        variables = [v for v in VAR_CORE_VARIABLES if v in df.columns]

    edges = []

    for cause in variables:
        for effect in variables:
            if cause == effect:
                continue

            pair = df[[effect, cause]].dropna()
            if len(pair) < max_lag * 4:
                continue

            pair_s = pair.copy()
            for col in pair_s.columns:
                sr = test_stationarity(pair_s[col], name=col, verbose=False)
                if not sr["is_stationary"]:
                    pair_s[col] = pair_s[col].diff()
            pair_s = pair_s.dropna()

            if len(pair_s) < max_lag * 4:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc = grangercausalitytests(
                        pair_s[[effect, cause]], maxlag=max_lag, verbose=False
                    )

                min_p = min(gc[lag][0]["ssr_ftest"][1]
                           for lag in range(1, max_lag + 1) if lag in gc)

                best_lag = min(
                    range(1, max_lag + 1),
                    key=lambda lag: gc[lag][0]["ssr_ftest"][1] if lag in gc else 1.0
                )

                if min_p < alpha:
                    edges.append({
                        "source": cause,
                        "target": effect,
                        "p_value": min_p,
                        "weight": -np.log10(max(min_p, 1e-10)),
                        "best_lag": best_lag,
                    })

            except Exception:
                pass

    edge_df = pd.DataFrame(edges)

    # ── Compute node centrality ──
    node_meta = {}
    for var in variables:
        out_edges = edge_df[edge_df["source"] == var] if not edge_df.empty else pd.DataFrame()
        in_edges = edge_df[edge_df["target"] == var] if not edge_df.empty else pd.DataFrame()

        node_meta[var] = {
            "out_degree": len(out_edges),
            "in_degree": len(in_edges),
            "total_degree": len(out_edges) + len(in_edges),
            "out_weight": out_edges["weight"].sum() if not out_edges.empty else 0,
            "in_weight": in_edges["weight"].sum() if not in_edges.empty else 0,
            "is_leader": len(out_edges) > len(in_edges),  # Causes more than it's caused by
        }

    print(f"[✓] Causality network: {len(variables)} nodes, {len(edge_df)} edges")

    # Print leaders
    leaders = {k: v for k, v in node_meta.items() if v["is_leader"]}
    followers = {k: v for k, v in node_meta.items() if not v["is_leader"]}

    if leaders:
        print(f"\n  LEADING indicators (cause more than they're caused):")
        for var, meta in sorted(leaders.items(), key=lambda x: -x[1]["out_degree"]):
            print(f"    {var:30s}: out={meta['out_degree']}, in={meta['in_degree']}")

    if followers:
        print(f"\n  LAGGING indicators (caused more than they cause):")
        for var, meta in sorted(followers.items(), key=lambda x: -x[1]["in_degree"]):
            print(f"    {var:30s}: out={meta['out_degree']}, in={meta['in_degree']}")

    return edge_df, node_meta


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FULL GRANGER PIPELINE
# ──────────────────────────────────────────────────────────────

def run_granger_pipeline(df, save=True):
    """
    Execute the complete Granger causality analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset.

    Returns
    -------
    dict with all Granger outputs.
    """
    ensure_directories()

    print("\n" + "=" * 60)
    print("   GRANGER CAUSALITY PIPELINE")
    print("=" * 60 + "\n")

    variables = [v for v in VAR_CORE_VARIABLES if v in df.columns]
    target = "eurusd_close"

    # ── Bidirectional analysis ──
    bidir_results = run_bidirectional_analysis(df, variables=variables, target=target)

    # ── Rolling Granger ──
    rolling_results, rolling_summary = run_rolling_granger_analysis(
        df, target=target, variables=[v for v in variables if v != target]
    )

    # ── Lag profiles ──
    lag_profiles, lag_summary = run_lag_profiles(
        df, target=target, variables=[v for v in variables if v != target]
    )

    # ── Causality network ──
    network_edges, network_nodes = build_causality_network(df, variables=variables)

    # ── Save results ──
    if save:
        if not bidir_results.empty:
            save_dataframe(bidir_results, "granger_bidirectional.csv", subdir="models")

        if not rolling_summary.empty:
            save_dataframe(rolling_summary, "granger_rolling_summary.csv", subdir="models")

        if not lag_summary.empty:
            save_dataframe(lag_summary, "granger_lag_profiles.csv", subdir="models")

        if not network_edges.empty:
            save_dataframe(network_edges, "granger_network_edges.csv", subdir="models")

        node_df = pd.DataFrame(network_nodes).T
        node_df.index.name = "variable"
        save_dataframe(node_df, "granger_network_nodes.csv", subdir="models")

    print("\n" + "=" * 60)
    print("   GRANGER PIPELINE COMPLETE")
    print("=" * 60 + "\n")

    return {
        "bidirectional": bidir_results,
        "rolling_results": rolling_results,
        "rolling_summary": rolling_summary,
        "lag_profiles": lag_profiles,
        "lag_summary": lag_summary,
        "network_edges": network_edges,
        "network_nodes": network_nodes,
    }


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from features.engineer import run_feature_engineering

    print("Running feature engineering first...")
    fe_result = run_feature_engineering()

    print("\nRunning Granger pipeline...")
    granger_result = run_granger_pipeline(fe_result["full_df"])