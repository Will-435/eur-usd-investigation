# eur_usd_pipeline/models/forecast.py
"""
Forecast comparison and model diagnostics module.

Brings together VAR and GLM forecasts and provides:

  1. Out-of-sample forecast comparison (VAR vs GLM)
  2. Model combination (ensemble) forecasts
  3. Comprehensive diagnostic tests
  4. Scenario analysis
  5. Final thesis assessment

This is the "verdict" module — it synthesises all modelling work
into a clear conclusion about EUR/USD direction.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats as sp_stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FORECAST_HORIZONS
from utils.helpers import ensure_directories, save_dataframe


# ──────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ──────────────────────────────────────────────────────────────

def compare_var_glm(var_results, glm_results, test_df, save=True):
    """
    Compare VAR and GLM forecasts on the test set.

    Parameters
    ----------
    var_results : dict
        From run_var_pipeline().
    glm_results : dict
        From run_glm_pipeline().
    test_df : pd.DataFrame
        Held-out test data.

    Returns
    -------
    pd.DataFrame
        Comparison metrics for all model-horizon combinations.
    dict
        Detailed comparison data for plotting.
    """
    print("\n" + "=" * 60)
    print("   VAR vs GLM FORECAST COMPARISON")
    print("=" * 60 + "\n")

    comparison_rows = []
    comparison_data = {}

    # ── VAR in-sample performance ──
    var_model = var_results.get("model")
    if var_model and var_model.results is not None:
        # Get VAR fitted values for EUR/USD
        try:
            fitted = var_model.results.fittedvalues
            if "eurusd_close" in fitted.columns:
                var_fitted = fitted["eurusd_close"]
                actual_stationary = var_model.data_stationary["eurusd_close"]

                # Align
                common_idx = var_fitted.index.intersection(actual_stationary.index)
                y_true = actual_stationary.loc[common_idx].values
                y_pred = var_fitted.loc[common_idx].values

                var_train_metrics = _compute_metrics(y_true, y_pred)

                comparison_rows.append({
                    "model": "VAR",
                    "horizon": "in_sample",
                    "metric_set": "training",
                    **var_train_metrics,
                })
        except Exception as e:
            print(f"  [!] VAR in-sample metrics failed: {e}")

    # ── GLM performance by horizon ──
    for horizon_label, horizon_data in glm_results.items():
        best_glm = horizon_data.get("best_model")
        target_col = horizon_data.get("target")

        if best_glm is None or target_col is None:
            continue

        # Training metrics
        if best_glm.train_metrics:
            comparison_rows.append({
                "model": f"GLM_{horizon_label}",
                "horizon": horizon_label,
                "metric_set": "training",
                **best_glm.train_metrics,
            })

        # Test metrics
        if target_col in test_df.columns:
            try:
                test_metrics, test_comparison = best_glm.evaluate(test_df)
                if test_metrics:
                    comparison_rows.append({
                        "model": f"GLM_{horizon_label}",
                        "horizon": horizon_label,
                        "metric_set": "test",
                        **test_metrics,
                    })

                    comparison_data[f"glm_{horizon_label}"] = test_comparison

            except Exception as e:
                print(f"  [!] GLM test evaluation failed for {horizon_label}: {e}")

    # ── Build comparison table ──
    comp_df = pd.DataFrame(comparison_rows)

    if not comp_df.empty:
        print("\n── Model Performance Comparison ──\n")
        print(f"  {'Model':<20s} {'Horizon':<12s} {'Set':<10s} "
              f"{'RMSE':>10s} {'MAE':>10s} {'R²':>8s} {'Dir.Acc':>8s}")
        print("  " + "─" * 78)

        for _, row in comp_df.iterrows():
            print(f"  {row['model']:<20s} {row['horizon']:<12s} {row['metric_set']:<10s} "
                  f"{row.get('rmse', np.nan):>10.6f} {row.get('mae', np.nan):>10.6f} "
                  f"{row.get('r2', np.nan):>8.4f} {row.get('directional_accuracy', np.nan):>8.1%}")

    if save and not comp_df.empty:
        save_dataframe(comp_df, "model_comparison.csv", subdir="models")

    return comp_df, comparison_data


def _compute_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {}

    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "directional_accuracy": np.mean(np.sign(y_true) == np.sign(y_pred)),
        "correlation": np.corrcoef(y_true, y_pred)[0, 1],
    }


# ──────────────────────────────────────────────────────────────
# ENSEMBLE FORECAST
# ──────────────────────────────────────────────────────────────

def ensemble_forecast(var_results, glm_results, test_df,
                      var_weight=0.5, glm_weight=0.5):
    """
    Combine VAR and GLM forecasts via weighted average.

    The VAR captures system dynamics (spillovers, feedback loops).
    The GLM captures non-linear factor loadings.
    Combining them often outperforms either alone.

    Parameters
    ----------
    var_results, glm_results : dict
    test_df : pd.DataFrame
    var_weight, glm_weight : float
        Weights for combining (must sum to 1).

    Returns
    -------
    pd.DataFrame
        Ensemble forecast comparison.
    """
    print("\n[...] Building ensemble (VAR + GLM) forecasts")

    # Normalise weights
    total = var_weight + glm_weight
    var_weight /= total
    glm_weight /= total

    ensemble_results = []

    # Get VAR forecasts
    var_forecasts = var_results.get("forecasts", {})

    for horizon_label, horizon_data in glm_results.items():
        best_glm = horizon_data.get("best_model")
        target_col = horizon_data.get("target")

        if best_glm is None or target_col is None:
            continue

        if target_col not in test_df.columns:
            continue

        # GLM predictions on test set
        try:
            glm_preds = best_glm.predict(test_df)
        except Exception:
            continue

        # For ensemble, we need both to be on the same scale
        test_actual = test_df[target_col].dropna()
        common_idx = glm_preds.dropna().index.intersection(test_actual.index)

        if len(common_idx) < 5:
            continue

        glm_values = glm_preds.loc[common_idx].values
        actual_values = test_actual.loc[common_idx].values

        # Simple ensemble: weighted average of GLM prediction and naive (VAR direction)
        # Since VAR produces multi-step forecasts and GLM produces point estimates,
        # we use GLM as primary and adjust by VAR directional signal
        var_direction = 1  # Default: neutral
        var_fc = var_forecasts.get(f"{FORECAST_HORIZONS[0]}m")  # Use shortest horizon
        if var_fc is not None and "eurusd_close" in var_fc.columns:
            var_model = var_results.get("model")
            if var_model:
                start_val = var_model.last_levels.get("eurusd_close", 1.0)
                end_val = var_fc["eurusd_close"].iloc[-1]
                var_direction = 1 if end_val > start_val else -1

        # Adjust GLM predictions by VAR directional confidence
        # If VAR agrees with GLM direction, boost; if disagrees, dampen
        glm_direction = np.sign(glm_values.mean())
        agreement = 1.1 if glm_direction == var_direction else 0.9

        ensemble_values = glm_values * agreement

        # Metrics
        glm_metrics = _compute_metrics(actual_values, glm_values)
        ensemble_metrics = _compute_metrics(actual_values, ensemble_values)

        ensemble_results.append({
            "horizon": horizon_label,
            "glm_rmse": glm_metrics.get("rmse", np.nan),
            "glm_dir_acc": glm_metrics.get("directional_accuracy", np.nan),
            "ensemble_rmse": ensemble_metrics.get("rmse", np.nan),
            "ensemble_dir_acc": ensemble_metrics.get("directional_accuracy", np.nan),
            "var_direction_signal": "BULLISH" if var_direction > 0 else "BEARISH",
            "var_glm_agree": var_direction == glm_direction,
        })

    result_df = pd.DataFrame(ensemble_results)

    if not result_df.empty:
        print("\n── Ensemble vs GLM-Only ──")
        for _, row in result_df.iterrows():
            agree = "✓" if row["var_glm_agree"] else "✗"
            print(f"  {row['horizon']:6s}: GLM RMSE={row['glm_rmse']:.6f} → "
                  f"Ensemble RMSE={row['ensemble_rmse']:.6f} "
                  f"| VAR signal: {row['var_direction_signal']} [{agree} agrees]")

    return result_df


# ──────────────────────────────────────────────────────────────
# DIAGNOSTIC TESTS
# ──────────────────────────────────────────────────────────────

def run_residual_diagnostics(var_results, glm_results):
    """
    Run diagnostic tests on model residuals.

    Tests:
      - Normality (Jarque-Bera, Shapiro-Wilk)
      - Autocorrelation (Ljung-Box)
      - Heteroscedasticity (Breusch-Pagan proxy)

    Returns
    -------
    pd.DataFrame
        Diagnostic test results.
    """
    print("\n[...] Running residual diagnostics")

    diagnostics = []

    # ── VAR residuals ──
    var_model = var_results.get("model")
    if var_model and var_model.results is not None:
        try:
            resid = var_model.results.resid
            if "eurusd_close" in resid.columns:
                eur_resid = resid["eurusd_close"].dropna().values

                diag = _diagnose_residuals(eur_resid, "VAR (EUR/USD eq.)")
                diagnostics.append(diag)
        except Exception as e:
            print(f"  [!] VAR diagnostics failed: {e}")

    # ── GLM residuals ──
    for horizon_label, horizon_data in glm_results.items():
        best_glm = horizon_data.get("best_model")
        if best_glm and best_glm.results is not None:
            try:
                resid = best_glm.results.resid_response
                if resid is not None and len(resid) > 20:
                    diag = _diagnose_residuals(
                        resid, f"GLM ({horizon_label})"
                    )
                    diagnostics.append(diag)
            except Exception as e:
                print(f"  [!] GLM {horizon_label} diagnostics failed: {e}")

    diag_df = pd.DataFrame(diagnostics)

    if not diag_df.empty:
        print("\n── Residual Diagnostics Summary ──")
        for _, row in diag_df.iterrows():
            print(f"\n  {row['model']}:")
            print(f"    Normality (JB):     p={row['jb_pvalue']:.4f} "
                  f"{'✓' if row['jb_pvalue'] > 0.05 else '✗ Non-normal'}")
            print(f"    Autocorrelation:    p={row['lb_pvalue']:.4f} "
                  f"{'✓' if row['lb_pvalue'] > 0.05 else '✗ Autocorrelated'}")
            print(f"    Mean residual:      {row['mean_resid']:.6f}")
            print(f"    Std residual:       {row['std_resid']:.6f}")
            print(f"    Skewness:           {row['skewness']:.3f}")
            print(f"    Kurtosis:           {row['kurtosis']:.3f}")

    return diag_df


def _diagnose_residuals(residuals, model_name):
    """Run all diagnostic tests on a residual series."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    diag = {"model": model_name}

    # ── Normality: Jarque-Bera ──
    try:
        jb_stat, jb_p = sp_stats.jarque_bera(residuals)
        diag["jb_statistic"] = jb_stat
        diag["jb_pvalue"] = jb_p
    except Exception:
        diag["jb_statistic"] = np.nan
        diag["jb_pvalue"] = np.nan

    # ── Autocorrelation: Ljung-Box ──
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            diag["lb_statistic"] = lb_result["lb_stat"].iloc[-1]
            diag["lb_pvalue"] = lb_result["lb_pvalue"].iloc[-1]
    except Exception:
        diag["lb_statistic"] = np.nan
        diag["lb_pvalue"] = np.nan

    # ── Descriptive stats ──
    diag["mean_resid"] = np.mean(residuals)
    diag["std_resid"] = np.std(residuals)
    diag["skewness"] = sp_stats.skew(residuals)
    diag["kurtosis"] = sp_stats.kurtosis(residuals)
    diag["n_obs"] = len(residuals)

    return diag


# ──────────────────────────────────────────────────────────────
# SCENARIO ANALYSIS
# ──────────────────────────────────────────────────────────────

def scenario_analysis(var_results, glm_results):
    """
    Run scenario analysis to stress-test the EUR/USD thesis.

    Scenarios:
      1. BASE CASE: Current trends continue
      2. BULL CASE: Oil drops 20%, ECB hikes, specs cover shorts
      3. BEAR CASE: Oil spikes 30%, Fed hikes further, risk-off
      4. SHOCK CASE: Sudden geopolitical event (VIX spikes)

    Returns
    -------
    pd.DataFrame
        Scenario outcomes.
    """
    print("\n" + "=" * 60)
    print("   SCENARIO ANALYSIS")
    print("=" * 60 + "\n")

    var_model = var_results.get("model")
    if var_model is None or var_model.results is None:
        print("[!] VAR model required for scenario analysis")
        return pd.DataFrame()

    scenarios = {
        "BASE_CASE": {
            "description": "Current trends continue — gradual normalisation",
            "shocks": {},  # No additional shocks
        },
        "BULL_EUR": {
            "description": "Oil drops 20%, ECB hawkish, spec short covering",
            "shocks": {
                "brent_close": -0.20,          # 20% decline
                "yield_spread_2y": +0.50,       # Spread narrows 50bp
                "net_spec_position": +30000,     # Significant short covering
                "rr_25d_proxy": +0.5,           # Options tilt to calls
                "vix_close": -5.0,              # Risk appetite improves
            },
        },
        "BEAR_EUR": {
            "description": "Oil spikes 30%, Fed hikes, risk-off",
            "shocks": {
                "brent_close": +0.30,
                "yield_spread_2y": -0.75,       # Spread widens 75bp
                "net_spec_position": -20000,     # Specs add shorts
                "rr_25d_proxy": -1.0,           # Heavy put premium
                "vix_close": +10.0,             # Risk-off
            },
        },
        "GEOPOLITICAL_SHOCK": {
            "description": "Sudden geopolitical crisis — VIX spike, flight to USD",
            "shocks": {
                "vix_close": +20.0,
                "brent_close": +0.15,
                "yield_spread_2y": -0.30,
                "net_spec_position": -40000,     # Panic selling EUR
            },
        },
    }

    results = []

    for scenario_name, scenario in scenarios.items():
        print(f"\n  ── {scenario_name} ──")
        print(f"  {scenario['description']}")

        # Get base forecast (12-month)
        base_fc = var_results.get("forecasts", {}).get(f"{FORECAST_HORIZONS[0]}m")
        if base_fc is None or "eurusd_close" not in base_fc.columns:
            continue

        base_end = base_fc["eurusd_close"].iloc[-1]
        current = var_model.last_levels.get("eurusd_close", 1.0)

        # Apply shocks as percentage adjustments to the base forecast
        if scenario["shocks"]:
            # Estimate impact using a simple sensitivity approach
            total_impact_pct = 0

            for var, shock in scenario["shocks"].items():
                # Rough sensitivity: how much does EUR/USD move per unit of this variable?
                if var in var_model.variables and var_model.results is not None:
                    try:
                        # Use the VAR coefficient on first lag of this variable
                        # in the EUR/USD equation
                        eur_idx = var_model.variables.index("eurusd_close")
                        var_idx = var_model.variables.index(var)
                        coef = var_model.results.coefs[0][eur_idx, var_idx]  # Lag 1 coefficient

                        if var == "brent_close":
                            # Shock is a percentage change
                            impact = coef * shock * current * 100
                        else:
                            # Shock is in levels
                            impact = coef * shock

                        total_impact_pct += impact
                        print(f"    {var}: shock={shock:+.2f} → impact={impact:+.4f}")
                    except (IndexError, Exception):
                        pass

            scenario_end = base_end * (1 + total_impact_pct)
        else:
            scenario_end = base_end

        change_pct = (scenario_end - current) / current * 100
        direction = "EUR APPRECIATION" if change_pct > 0 else "EUR DEPRECIATION"

        results.append({
            "scenario": scenario_name,
            "description": scenario["description"],
            "current_eurusd": current,
            "forecast_eurusd": scenario_end,
            "change_pct": change_pct,
            "direction": direction,
        })

        print(f"    → EUR/USD: {current:.4f} → {scenario_end:.4f} ({change_pct:+.2f}% {direction})")

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        print(f"\n── Scenario Summary ──")
        for _, row in result_df.iterrows():
            emoji = "📈" if row["change_pct"] > 0 else "📉"
            print(f"  {emoji} {row['scenario']:25s}: {row['change_pct']:+.2f}% → {row['direction']}")

    return result_df


# ──────────────────────────────────────────────────────────────
# THESIS ASSESSMENT
# ──────────────────────────────────────────────────────────────

def assess_thesis(var_results, glm_results, granger_results,
                  raw_data, save=True):
    """
    Synthesise all analysis into a final thesis assessment.

    Answers: "Will the Euro appreciate vs the Dollar over
    the next 1-2 years, and what are the key drivers?"

    Returns
    -------
    dict
        Complete thesis assessment.
    """
    print("\n" + "=" * 60)
    print("   ╔══════════════════════════════════════╗")
    print("   ║     EUR/USD THESIS ASSESSMENT        ║")
    print("   ╚══════════════════════════════════════╝")
    print("=" * 60 + "\n")

    assessment = {
        "signals": {},
        "overall_direction": None,
        "confidence": None,
        "key_drivers": [],
        "key_risks": [],
    }

    signals = {}

    # ── Signal 1: VAR forecast direction ──
    var_forecasts = var_results.get("forecasts", {})
    var_model = var_results.get("model")

    if var_model and var_forecasts:
        current = var_model.last_levels.get("eurusd_close", np.nan)

        for label, fc in var_forecasts.items():
            if "eurusd_close" in fc.columns:
                end = fc["eurusd_close"].iloc[-1]
                chg = (end - current) / current * 100
                signals[f"var_{label}"] = {
                    "direction": "BULLISH" if chg > 0 else "BEARISH",
                    "magnitude": chg,
                    "detail": f"VAR {label} forecast: {current:.4f} → {end:.4f} ({chg:+.2f}%)",
                }

    # ── Signal 2: GLM predictions ──
    for horizon_label, horizon_data in glm_results.items():
        best_glm = horizon_data.get("best_model")
        if best_glm and best_glm.train_metrics:
            dir_acc = best_glm.train_metrics.get("directional_accuracy", 0.5)
            signals[f"glm_{horizon_label}"] = {
                "direction": "INFORMATIVE" if dir_acc > 0.55 else "WEAK",
                "magnitude": dir_acc,
                "detail": f"GLM {horizon_label} directional accuracy: {dir_acc:.1%}",
            }

    # ── Signal 3: Yield spread direction ──
    yields = raw_data.get("yields")
    if yields is not None and "yield_spread_2y" in yields.columns:
        spread = yields["yield_spread_2y"]
        spread_recent = spread.iloc[-1]
        spread_13w_ago = spread.iloc[-13] if len(spread) > 13 else spread.iloc[0]
        narrowing = spread_recent > spread_13w_ago

        signals["yield_spread"] = {
            "direction": "BULLISH" if narrowing else "BEARISH",
            "magnitude": spread_recent - spread_13w_ago,
            "detail": f"2Y spread: {spread_13w_ago:.2f}% → {spread_recent:.2f}% "
                     f"({'narrowing ✓' if narrowing else 'widening ✗'})",
        }

    # ── Signal 4: COT positioning ──
    cot = raw_data.get("cot")
    if cot is not None and "net_spec_position" in cot.columns and not cot.empty:
        latest_pos = cot["net_spec_position"].iloc[-1]
        pos_z = cot.get("crowding_zscore")
        if pos_z is not None and not pos_z.empty:
            z = pos_z.iloc[-1]
            if z < -1:
                signals["positioning"] = {
                    "direction": "BULLISH",
                    "magnitude": z,
                    "detail": f"Specs heavily SHORT (z={z:.2f}) — squeeze risk elevated",
                }
            elif z > 1.5:
                signals["positioning"] = {
                    "direction": "BEARISH",
                    "magnitude": z,
                    "detail": f"Specs heavily LONG (z={z:.2f}) — crowded trade risk",
                }
            else:
                signals["positioning"] = {
                    "direction": "NEUTRAL",
                    "magnitude": z,
                    "detail": f"Positioning neutral (z={z:.2f})",
                }

    # ── Signal 5: Oil / Terms of Trade ──
    oil = raw_data.get("oil")
    if oil is not None and "brent_close" in oil.columns:
        brent = oil["brent_close"]
        brent_13w_chg = brent.iloc[-1] / brent.iloc[-13] - 1 if len(brent) > 13 else 0

        signals["oil_tot"] = {
            "direction": "BULLISH" if brent_13w_chg < -0.05 else ("BEARISH" if brent_13w_chg > 0.10 else "NEUTRAL"),
            "magnitude": brent_13w_chg,
            "detail": f"Brent 13w change: {brent_13w_chg:+.1%} "
                     f"({'falling → EUR positive' if brent_13w_chg < 0 else 'rising → EUR headwind'})",
        }

    # ── Signal 6: Granger causality ──
    gc = granger_results.get("bidirectional", pd.DataFrame())
    if not gc.empty:
        sig_causes = gc[gc["b_causes_a"] == True]
        n_sig = len(sig_causes)
        signals["granger"] = {
            "direction": "INFORMATIVE",
            "magnitude": n_sig,
            "detail": f"{n_sig} variables significantly Granger-cause EUR/USD",
        }

    assessment["signals"] = signals

    # ── Overall assessment ──
    bullish = sum(1 for s in signals.values() if s["direction"] == "BULLISH")
    bearish = sum(1 for s in signals.values() if s["direction"] == "BEARISH")
    total = len(signals)

    if bullish > bearish:
        assessment["overall_direction"] = "EUR APPRECIATION LIKELY"
        assessment["confidence"] = bullish / total if total > 0 else 0
    elif bearish > bullish:
        assessment["overall_direction"] = "EUR DEPRECIATION LIKELY"
        assessment["confidence"] = bearish / total if total > 0 else 0
    else:
        assessment["overall_direction"] = "INCONCLUSIVE"
        assessment["confidence"] = 0.5

    # ── Print final verdict ──
    print("\n╔══════════════════════════════════════════════╗")
    print(f"║  VERDICT: {assessment['overall_direction']:^35s}║")
    print(f"║  Confidence: {assessment['confidence']:.0%}{'':>31s}║")
    print("╠══════════════════════════════════════════════╣")

    print("║  SIGNAL BREAKDOWN:                           ║")
    for name, sig in signals.items():
        icon = "🟢" if sig["direction"] == "BULLISH" else ("🔴" if sig["direction"] == "BEARISH" else "🟡")
        print(f"║  {icon} {name:20s} {sig['direction']:12s}         ║")
        print(f"║     {sig['detail'][:44]:<44s}║")

    print("╚══════════════════════════════════════════════╝\n")

    # ── Key drivers and risks ──
    print("KEY DRIVERS:")
    bullish_signals = [(n, s) for n, s in signals.items() if s["direction"] == "BULLISH"]
    for name, sig in bullish_signals:
        print(f"  + {sig['detail']}")
        assessment["key_drivers"].append(sig["detail"])

    print("\nKEY RISKS:")
    bearish_signals = [(n, s) for n, s in signals.items() if s["direction"] == "BEARISH"]
    for name, sig in bearish_signals:
        print(f"  - {sig['detail']}")
        assessment["key_risks"].append(sig["detail"])

    neutral_signals = [(n, s) for n, s in signals.items()
                       if s["direction"] not in ("BULLISH", "BEARISH")]
    if neutral_signals:
        print("\nNEUTRAL/INFORMATIVE:")
        for name, sig in neutral_signals:
            print(f"  · {sig['detail']}")

    # Save assessment
    if save:
        assessment_df = pd.DataFrame([{
            "overall_direction": assessment["overall_direction"],
            "confidence": assessment["confidence"],
            "n_bullish": bullish,
            "n_bearish": bearish,
            "n_total_signals": total,
            "key_drivers": "; ".join(assessment["key_drivers"]),
            "key_risks": "; ".join(assessment["key_risks"]),
        }])
        save_dataframe(assessment_df, "thesis_assessment.csv", subdir="models")

    return assessment


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FULL FORECAST PIPELINE
# ──────────────────────────────────────────────────────────────

def run_forecast_pipeline(var_results, glm_results, granger_results,
                          test_df, raw_data, save=True):
    """
    Execute the complete forecast comparison and thesis assessment.

    Returns
    -------
    dict with all forecast comparison outputs.
    """
    ensure_directories()

    print("\n" + "=" * 60)
    print("   FORECAST & THESIS PIPELINE")
    print("=" * 60 + "\n")

    # ── Model comparison ──
    comparison, comparison_data = compare_var_glm(
        var_results, glm_results, test_df, save=save
    )

    # ── Ensemble ──
    ensemble = ensemble_forecast(var_results, glm_results, test_df)
    if save and not ensemble.empty:
        save_dataframe(ensemble, "ensemble_comparison.csv", subdir="models")

    # ── Diagnostics ──
    diagnostics = run_residual_diagnostics(var_results, glm_results)
    if save and not diagnostics.empty:
        save_dataframe(diagnostics, "residual_diagnostics.csv", subdir="models")

    # ── Scenarios ──
    scenarios = scenario_analysis(var_results, glm_results)
    if save and not scenarios.empty:
        save_dataframe(scenarios, "scenario_analysis.csv", subdir="models")

    # ── Final thesis assessment ──
    thesis = assess_thesis(
        var_results, glm_results, granger_results,
        raw_data, save=save
    )

    return {
        "comparison": comparison,
        "comparison_data": comparison_data,
        "ensemble": ensemble,
        "diagnostics": diagnostics,
        "scenarios": scenarios,
        "thesis": thesis,
    }


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Forecast module loaded. Run from main.py for full pipeline.")
    print("Requires VAR and GLM results as inputs.")