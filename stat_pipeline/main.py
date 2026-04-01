# eur_usd_pipeline/main.py
"""
EUR/USD Macro Analysis Pipeline — Master Orchestrator

This script runs the complete pipeline end-to-end:

  1. Feature Engineering (data fetch + indicators + spreads + sentiment)
  2. VAR Model (lag selection, fitting, IRFs, FEVD, forecasts)
  3. GLM Model (multi-specification comparison, feature importance)
  4. Granger Causality Analysis (bidirectional, rolling, lag profiles)
  5. Forecast Comparison & Thesis Assessment
  6. Visualisation Suite (all charts + dashboard)

Usage:
    python main.py                  # Full pipeline
    python main.py --skip-plots     # Skip visualisation step
    python main.py --data-only      # Only run data + feature engineering

Estimated runtime: 5-15 minutes depending on internet speed and
data availability. The CFTC COT download is usually the slowest step.
"""

import argparse
import time
import warnings
import traceback
import sys
import os

# Suppress noisy warnings from yfinance, statsmodels, etc.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import START_DATE, END_DATE, FORECAST_HORIZONS
from utils.helpers import ensure_directories


# ──────────────────────────────────────────────────────────────
# STEP RUNNERS
# ──────────────────────────────────────────────────────────────

def step_feature_engineering():
    """Step 1: Data collection and feature engineering."""
    from features.engineer import run_feature_engineering
    return run_feature_engineering(start=START_DATE, end=END_DATE, save=True)


def step_var_model(fe_result):
    """Step 2: VAR model pipeline."""
    from models.var_model import run_var_pipeline
    return run_var_pipeline(fe_result["full_df"], save=True)


def step_glm_model(fe_result):
    """Step 3: GLM model pipeline."""
    from models.glm_model import run_glm_pipeline
    return run_glm_pipeline(
        fe_result["train_df"],
        fe_result["test_df"],
        save=True,
    )


def step_granger_analysis(fe_result):
    """Step 4: Granger causality analysis."""
    from models.granger import run_granger_pipeline
    return run_granger_pipeline(fe_result["full_df"], save=True)


def step_forecast_comparison(var_results, glm_results, granger_results,
                              fe_result):
    """Step 5: Forecast comparison and thesis assessment."""
    from models.forecast import run_forecast_pipeline
    return run_forecast_pipeline(
        var_results=var_results,
        glm_results=glm_results,
        granger_results=granger_results,
        test_df=fe_result["test_df"],
        raw_data=fe_result["raw_data"],
        save=True,
    )


def step_visualisations(fe_result, var_results, glm_results,
                         granger_results, forecast_results):
    """Step 6: Generate all visualisations."""
    from visualisations.yield_spread_plot import (
        plot_yield_spread_vs_eurusd, plot_yield_spread_correlation,
    )
    from visualisations.terms_of_trade_plot import (
        plot_terms_of_trade, plot_energy_pressure,
    )
    from visualisations.cot_plot import (
        plot_cot_positioning, plot_dealer_vs_spec, plot_positioning_heatmap,
    )
    from visualisations.risk_reversal_plot import (
        plot_risk_reversals, plot_rr_vs_positioning,
    )
    from visualisations.model_plots import (
        plot_var_forecast, plot_irf, plot_fevd,
        plot_granger_heatmap, plot_glm_importance,
        plot_rolling_correlation_heatmap,
        plot_scenarios, plot_residual_diagnostics,
    )
    from visualisations.dashboard import create_dashboard

    raw = fe_result["raw_data"]
    fx = raw.get("fx")
    yields = raw.get("yields")
    oil = raw.get("oil")
    cot = raw.get("cot")
    rr = raw.get("risk_reversals")

    print("\n" + "=" * 60)
    print("   GENERATING VISUALISATIONS")
    print("=" * 60 + "\n")

    plots_generated = 0

    # ── Yield spread plots ──
    try:
        if yields is not None and fx is not None:
            var_fc = None
            if var_results:
                fcs = var_results.get("forecasts", {})
                var_fc = fcs.get(f"{FORECAST_HORIZONS[0]}m")

            plot_yield_spread_vs_eurusd(yields, fx, var_forecast=var_fc)
            plot_yield_spread_correlation(yields, fx)
            plots_generated += 2
    except Exception as e:
        print(f"  [!] Yield spread plots failed: {e}")

    # ── Terms of trade plots ──
    try:
        if oil is not None and fx is not None:
            plot_terms_of_trade(oil, fx)
            plot_energy_pressure(oil, fx)
            plots_generated += 2
    except Exception as e:
        print(f"  [!] ToT plots failed: {e}")

    # ── COT plots ──
    try:
        if cot is not None and not cot.empty and fx is not None:
            plot_cot_positioning(cot, fx)
            plot_dealer_vs_spec(cot, fx)
            plot_positioning_heatmap(cot)
            plots_generated += 3
    except Exception as e:
        print(f"  [!] COT plots failed: {e}")

    # ── Risk reversal plots ──
    try:
        if rr is not None and not rr.empty and fx is not None:
            plot_risk_reversals(rr, fx)
            if cot is not None and not cot.empty:
                plot_rr_vs_positioning(rr, cot, fx)
                plots_generated += 1
            plots_generated += 1
    except Exception as e:
        print(f"  [!] Risk reversal plots failed: {e}")

    # ── Model plots ──
    try:
        if var_results:
            plot_var_forecast(var_results, fx)
            plot_irf(var_results)
            plot_fevd(var_results)
            plot_residual_diagnostics(var_results)
            plots_generated += 4
    except Exception as e:
        print(f"  [!] VAR model plots failed: {e}")

    try:
        if granger_results:
            gc_matrix = granger_results.get("granger_matrix")
            if gc_matrix is not None:
                # Import from var_model if needed
                from models.var_model import run_full_granger_matrix
                plot_granger_heatmap(gc_matrix)
                plots_generated += 1
    except Exception as e:
        print(f"  [!] Granger heatmap failed: {e}")

    try:
        if glm_results:
            for horizon in glm_results.keys():
                plot_glm_importance(glm_results, horizon=horizon)
                plots_generated += 1
    except Exception as e:
        print(f"  [!] GLM importance plots failed: {e}")

    # ── Rolling correlation heatmap ──
    try:
        plot_rolling_correlation_heatmap(fe_result["full_df"])
        plots_generated += 1
    except Exception as e:
        print(f"  [!] Rolling correlation heatmap failed: {e}")

    # ── Scenario analysis ──
    try:
        if forecast_results:
            scenarios = forecast_results.get("scenarios")
            if scenarios is not None and not scenarios.empty:
                plot_scenarios(scenarios)
                plots_generated += 1
    except Exception as e:
        print(f"  [!] Scenario plot failed: {e}")

    # ── Dashboard ──
    try:
        create_dashboard(
            raw_data=raw,
            var_results=var_results,
            glm_results=glm_results,
            forecast_results=forecast_results,
        )
        plots_generated += 1
    except Exception as e:
        print(f"  [!] Dashboard creation failed: {e}")

    print(f"\n[✓] Visualisation complete: {plots_generated} plots generated")

    return plots_generated


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def main(skip_plots=False, data_only=False):
    """
    Run the complete EUR/USD analysis pipeline.

    Parameters
    ----------
    skip_plots : bool
        If True, skip the visualisation step.
    data_only : bool
        If True, only run data collection and feature engineering.
    """
    pipeline_start = time.time()

    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                        ║")
    print("║   EUR/USD MACRO ANALYSIS PIPELINE                      ║")
    print("║   Will the Euro Appreciate vs the Dollar?              ║")
    print("║                                                        ║")
    print("║   Period: {} → {}                      ║".format(START_DATE, END_DATE))
    print("║   Forecast horizons: {}                         ║".format(
        ", ".join(f"{h}m" for h in FORECAST_HORIZONS)))
    print("║                                                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    ensure_directories()

    results = {
        "fe": None,
        "var": None,
        "glm": None,
        "granger": None,
        "forecast": None,
        "n_plots": 0,
    }

    # ──────────────────────────────────────────────────────
    # STEP 1: FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "▓" * 60)
    print("  STEP 1/6: FEATURE ENGINEERING")
    print("▓" * 60)

    try:
        results["fe"] = step_feature_engineering()
        elapsed = time.time() - step_start
        print(f"\n[✓] Step 1 complete ({elapsed:.1f}s)")
    except Exception as e:
        print(f"\n[✗] Step 1 FAILED: {e}")
        traceback.print_exc()
        print("\nCannot continue without feature engineering. Exiting.")
        return results

    if data_only:
        elapsed_total = time.time() - pipeline_start
        print(f"\n[✓] Data-only mode complete ({elapsed_total:.1f}s)")
        return results

    # ──────────────────────────────────────────────────────
    # STEP 2: VAR MODEL
    # ──────────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "▓" * 60)
    print("  STEP 2/6: VAR MODEL")
    print("▓" * 60)

    try:
        results["var"] = step_var_model(results["fe"])
        elapsed = time.time() - step_start
        print(f"\n[✓] Step 2 complete ({elapsed:.1f}s)")
    except Exception as e:
        print(f"\n[✗] Step 2 FAILED: {e}")
        traceback.print_exc()
        print("  Continuing with remaining steps...")

    # ──────────────────────────────────────────────────────
    # STEP 3: GLM MODEL
    # ──────────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "▓" * 60)
    print("  STEP 3/6: GLM MODEL")
    print("▓" * 60)

    try:
        results["glm"] = step_glm_model(results["fe"])
        elapsed = time.time() - step_start
        print(f"\n[✓] Step 3 complete ({elapsed:.1f}s)")
    except Exception as e:
        print(f"\n[✗] Step 3 FAILED: {e}")
        traceback.print_exc()
        print("  Continuing with remaining steps...")

    # ──────────────────────────────────────────────────────
    # STEP 4: GRANGER CAUSALITY
    # ──────────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "▓" * 60)
    print("  STEP 4/6: GRANGER CAUSALITY ANALYSIS")
    print("▓" * 60)

    try:
        results["granger"] = step_granger_analysis(results["fe"])
        elapsed = time.time() - step_start
        print(f"\n[✓] Step 4 complete ({elapsed:.1f}s)")
    except Exception as e:
        print(f"\n[✗] Step 4 FAILED: {e}")
        traceback.print_exc()
        print("  Continuing with remaining steps...")

    # ──────────────────────────────────────────────────────
    # STEP 5: FORECAST COMPARISON & THESIS
    # ──────────────────────────────────────────────────────
    step_start = time.time()
    print("\n" + "▓" * 60)
    print("  STEP 5/6: FORECAST COMPARISON & THESIS ASSESSMENT")
    print("▓" * 60)

    if results["var"] or results["glm"]:
        try:
            results["forecast"] = step_forecast_comparison(
                var_results=results["var"] or {},
                glm_results=results["glm"] or {},
                granger_results=results["granger"] or {},
                fe_result=results["fe"],
            )
            elapsed = time.time() - step_start
            print(f"\n[✓] Step 5 complete ({elapsed:.1f}s)")
        except Exception as e:
            print(f"\n[✗] Step 5 FAILED: {e}")
            traceback.print_exc()
    else:
        print("\n[·] Skipping — no model results available")

    # ──────────────────────────────────────────────────────
    # STEP 6: VISUALISATIONS
    # ──────────────────────────────────────────────────────
    if not skip_plots:
        step_start = time.time()
        print("\n" + "▓" * 60)
        print("  STEP 6/6: VISUALISATIONS")
        print("▓" * 60)

        try:
            results["n_plots"] = step_visualisations(
                fe_result=results["fe"],
                var_results=results["var"],
                glm_results=results["glm"],
                granger_results=results["granger"],
                forecast_results=results["forecast"],
            )
            elapsed = time.time() - step_start
            print(f"\n[✓] Step 6 complete ({elapsed:.1f}s)")
        except Exception as e:
            print(f"\n[✗] Step 6 FAILED: {e}")
            traceback.print_exc()
    else:
        print("\n[·] Skipping visualisations (--skip-plots)")

    # ──────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_start

    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                PIPELINE COMPLETE                        ║")
    print("╠══════════════════════════════════════════════════════════╣")

    step_status = {
        "Feature Engineering": results["fe"] is not None,
        "VAR Model":           results["var"] is not None,
        "GLM Model":           results["glm"] is not None,
        "Granger Causality":   results["granger"] is not None,
        "Forecast & Thesis":   results["forecast"] is not None,
        "Visualisations":      results["n_plots"] > 0,
    }

    for step_name, success in step_status.items():
        icon = "✓" if success else "✗"
        print(f"║  [{icon}] {step_name:<30s}                     ║")

    print(f"║                                                        ║")
    print(f"║  Total runtime: {elapsed_total:.1f}s ({elapsed_total/60:.1f} minutes){'':>16s}║")
    print(f"║  Plots generated: {results['n_plots']:<5d}{'':>31s}║")

    # Thesis verdict
    if results["forecast"]:
        thesis = results["forecast"].get("thesis", {})
        direction = thesis.get("overall_direction", "N/A")
        confidence = thesis.get("confidence", 0)
        print(f"║                                                        ║")
        print(f"║  VERDICT: {direction:<30s}          ║")
        print(f"║  Confidence: {confidence:.0%}{'':>40s}║")

    print(f"║                                                        ║")
    print(f"║  Outputs saved to: output/                             ║")
    print(f"║    - output/data/    (CSV data files)                  ║")
    print(f"║    - output/models/  (model results)                   ║")
    print(f"║    - output/plots/   (PNG visualisations)              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    return results


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EUR/USD Macro Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run full pipeline
  python main.py --skip-plots       Run without generating plots
  python main.py --data-only        Only fetch data and engineer features
        """,
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip visualisation generation",
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only run data collection and feature engineering",
    )

    args = parser.parse_args()

    main(skip_plots=args.skip_plots, data_only=args.data_only)
