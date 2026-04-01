# eur_usd_pipeline/visualisations/dashboard.py
"""
Combined summary dashboard — multi-panel overview of the full analysis.

Brings together the key visuals from every module into a single
publication-quality figure that tells the complete EUR/USD story.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOT_DIR


COLORS = {
    "eurusd": "#1f77b4",
    "spread": "#d62728",
    "brent": "#ff7f0e",
    "cot": "#2ca02c",
    "rr": "#9467bd",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#999999",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#333333",
    "panel_bg": "#ffffff",
}


def create_dashboard(raw_data, var_results=None, glm_results=None,
                     forecast_results=None, save=True, show=False):
    """
    Create a comprehensive multi-panel dashboard.

    Layout (6 panels):
      ┌──────────────────────────────┐
      │  EUR/USD + VAR Forecast      │
      ├──────────────┬───────────────┤
      │ Yield Spread │   Oil / ToT   │
      ├──────────────┼───────────────┤
      │ COT Position │  Risk Reversal│
      ├──────────────┴───────────────┤
      │     Thesis Verdict Panel     │
      └──────────────────────────────┘

    Parameters
    ----------
    raw_data : dict
        From feature engineering pipeline.
    var_results : dict, optional
        From VAR pipeline.
    glm_results : dict, optional
        From GLM pipeline.
    forecast_results : dict, optional
        From forecast pipeline (contains thesis assessment).
    """
    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor(COLORS["bg"])

    gs = GridSpec(4, 2, height_ratios=[2.5, 2, 2, 1.2],
                  hspace=0.30, wspace=0.25)

    # ──────────────────────────────────────────────────────
    # PANEL 1: EUR/USD + VAR Forecast (full width)
    # ──────────────────────────────────────────────────────
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor(COLORS["panel_bg"])

    fx = raw_data.get("fx")
    if fx is not None and "eurusd_close" in fx.columns:
        ax_main.plot(fx.index, fx["eurusd_close"],
                     color=COLORS["eurusd"], linewidth=2.0,
                     label="EUR/USD Spot")

        # Add VAR forecast
        if var_results:
            forecasts = var_results.get("forecasts", {})
            forecast_cis = var_results.get("forecast_cis", {})

            for label, fc in forecasts.items():
                if "eurusd_close" in fc.columns:
                    # Connect line
                    ax_main.plot(
                        [fx.index[-1], fc.index[0]],
                        [fx["eurusd_close"].iloc[-1], fc["eurusd_close"].iloc[0]],
                        color=COLORS["eurusd"], linewidth=1.5,
                        linestyle="--", alpha=0.5,
                    )
                    ax_main.plot(fc.index, fc["eurusd_close"],
                                 color=COLORS["eurusd"], linewidth=2.0,
                                 linestyle="--", alpha=0.7,
                                 label=f"VAR {label} Forecast")

                    # Confidence interval
                    ci = forecast_cis.get(label)
                    if ci and "eurusd_close" in ci["lower"].columns:
                        ax_main.fill_between(
                            fc.index,
                            ci["lower"]["eurusd_close"],
                            ci["upper"]["eurusd_close"],
                            color=COLORS["eurusd"], alpha=0.08,
                        )

                    # End annotation
                    end_val = fc["eurusd_close"].iloc[-1]
                    var_model = var_results.get("model")
                    if var_model:
                        current = var_model.last_levels.get("eurusd_close", end_val)
                        chg = (end_val - current) / current * 100
                        ax_main.annotate(
                            f"{end_val:.4f} ({chg:+.1f}%)",
                            xy=(fc.index[-1], end_val),
                            xytext=(10, 0), textcoords="offset points",
                            fontsize=9, fontweight="bold",
                            color=COLORS["positive"] if chg > 0 else COLORS["negative"],
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", alpha=0.9),
                        )

    ax_main.set_title("EUR/USD Historical & Forecast",
                      fontsize=14, fontweight="bold", pad=10)
    ax_main.set_ylabel("EUR/USD", fontsize=11)
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_main.grid(True, alpha=0.2)
    ax_main.xaxis.set_major_locator(mdates.YearLocator())
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ──────────────────────────────────────────────────────
    # PANEL 2: Yield Spread (left)
    # ──────────────────────────────────────────────────────
    ax_yield = fig.add_subplot(gs[1, 0])
    ax_yield.set_facecolor(COLORS["panel_bg"])

    yields = raw_data.get("yields")
    if yields is not None and "yield_spread_2y" in yields.columns:
        ax_yield.plot(yields.index, yields["yield_spread_2y"],
                      color=COLORS["spread"], linewidth=1.5)
        ax_yield.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)

        # Fill above/below zero
        ax_yield.fill_between(yields.index, yields["yield_spread_2y"], 0,
                               where=(yields["yield_spread_2y"] > yields["yield_spread_2y"].rolling(26).mean()),
                               color=COLORS["positive"], alpha=0.15)
        ax_yield.fill_between(yields.index, yields["yield_spread_2y"], 0,
                               where=(yields["yield_spread_2y"] <= yields["yield_spread_2y"].rolling(26).mean()),
                               color=COLORS["negative"], alpha=0.15)

        latest = yields["yield_spread_2y"].iloc[-1]
        ax_yield.set_title(f"2Y Yield Spread (DE−US): {latest:.2f}%",
                           fontsize=11, fontweight="bold")
    else:
        ax_yield.set_title("2Y Yield Spread (data unavailable)",
                           fontsize=11, fontweight="bold")
        ax_yield.text(0.5, 0.5, "No Data", ha="center", va="center",
                      transform=ax_yield.transAxes, fontsize=14, color="gray")

    ax_yield.set_ylabel("Spread (%)", fontsize=10)
    ax_yield.grid(True, alpha=0.2)
    ax_yield.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ──────────────────────────────────────────────────────
    # PANEL 3: Oil / Terms of Trade (right)
    # ──────────────────────────────────────────────────────
    ax_oil = fig.add_subplot(gs[1, 1])
    ax_oil.set_facecolor(COLORS["panel_bg"])

    oil = raw_data.get("oil")
    if oil is not None and "brent_close" in oil.columns:
        ax_oil.plot(oil.index, oil["brent_close"],
                    color=COLORS["brent"], linewidth=1.5)

        latest_brent = oil["brent_close"].iloc[-1]
        ax_oil.set_title(f"Brent Crude: ${latest_brent:.1f}/bbl",
                         fontsize=11, fontweight="bold")
    else:
        ax_oil.set_title("Brent Crude (data unavailable)",
                         fontsize=11, fontweight="bold")
        ax_oil.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax_oil.transAxes, fontsize=14, color="gray")

    ax_oil.set_ylabel("$/barrel", fontsize=10)
    ax_oil.grid(True, alpha=0.2)
    ax_oil.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ──────────────────────────────────────────────────────
    # PANEL 4: COT Positioning (left)
    # ──────────────────────────────────────────────────────
    ax_cot = fig.add_subplot(gs[2, 0])
    ax_cot.set_facecolor(COLORS["panel_bg"])

    cot = raw_data.get("cot")
    if cot is not None and not cot.empty and "net_spec_position" in cot.columns:
        net = cot["net_spec_position"]
        bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                      for v in net]
        ax_cot.bar(cot.index, net, width=5, color=bar_colors, alpha=0.5)
        ax_cot.axhline(y=0, color="gray", linewidth=1)

        latest_pos = net.iloc[-1]
        direction = "LONG" if latest_pos > 0 else "SHORT"
        ax_cot.set_title(f"COT Net Spec: {latest_pos:,.0f} ({direction})",
                         fontsize=11, fontweight="bold")
    else:
        ax_cot.set_title("COT Positioning (data unavailable)",
                         fontsize=11, fontweight="bold")
        ax_cot.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax_cot.transAxes, fontsize=14, color="gray")

    ax_cot.set_ylabel("Net Contracts", fontsize=10)
    ax_cot.grid(True, alpha=0.2)
    ax_cot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ──────────────────────────────────────────────────────
    # PANEL 5: Risk Reversal (right)
    # ──────────────────────────────────────────────────────
    ax_rr = fig.add_subplot(gs[2, 1])
    ax_rr.set_facecolor(COLORS["panel_bg"])

    rr = raw_data.get("risk_reversals")
    if rr is not None and not rr.empty and "rr_25d_proxy" in rr.columns:
        rr_vals = rr["rr_25d_proxy"]

        ax_rr.fill_between(rr.index, rr_vals, 0,
                           where=(rr_vals > 0), color=COLORS["positive"], alpha=0.3)
        ax_rr.fill_between(rr.index, rr_vals, 0,
                           where=(rr_vals <= 0), color=COLORS["negative"], alpha=0.3)
        ax_rr.plot(rr.index, rr_vals, color=COLORS["rr"], linewidth=1.5)
        ax_rr.axhline(y=0, color="gray", linewidth=1)

        latest_rr = rr_vals.iloc[-1]
        signal = "Calls Bid" if latest_rr > 0 else "Puts Bid"
        ax_rr.set_title(f"25Δ Risk Reversal: {latest_rr:.3f} ({signal})",
                         fontsize=11, fontweight="bold")
    else:
        ax_rr.set_title("Risk Reversal (data unavailable)",
                         fontsize=11, fontweight="bold")
        ax_rr.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax_rr.transAxes, fontsize=14, color="gray")

    ax_rr.set_ylabel("RR (vol pts)", fontsize=10)
    ax_rr.grid(True, alpha=0.2)
    ax_rr.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ──────────────────────────────────────────────────────
    # PANEL 6: Thesis Verdict (full width)
    # ──────────────────────────────────────────────────────
    ax_verdict = fig.add_subplot(gs[3, :])
    ax_verdict.set_facecolor("#f0f0f0")
    ax_verdict.set_xlim(0, 10)
    ax_verdict.set_ylim(0, 3)
    ax_verdict.axis("off")

    # Build verdict from thesis assessment
    thesis = None
    if forecast_results:
        thesis = forecast_results.get("thesis")

    if thesis:
        direction = thesis.get("overall_direction", "INCONCLUSIVE")
        confidence = thesis.get("confidence", 0)
        signals = thesis.get("signals", {})

        # Background colour based on direction
        if "APPRECIATION" in direction:
            verdict_color = "#d4edda"
            text_color = "#155724"
            icon = "📈"
        elif "DEPRECIATION" in direction:
            verdict_color = "#f8d7da"
            text_color = "#721c24"
            icon = "📉"
        else:
            verdict_color = "#fff3cd"
            text_color = "#856404"
            icon = "⚖️"

        ax_verdict.set_facecolor(verdict_color)

        # Main verdict
        ax_verdict.text(5, 2.3, f"{icon}  {direction}  {icon}",
                        ha="center", va="center", fontsize=20,
                        fontweight="bold", color=text_color)
        ax_verdict.text(5, 1.7, f"Confidence: {confidence:.0%}",
                        ha="center", va="center", fontsize=14, color=text_color)

        # Signal summary bar
        n_bull = sum(1 for s in signals.values() if s.get("direction") == "BULLISH")
        n_bear = sum(1 for s in signals.values() if s.get("direction") == "BEARISH")
        n_neut = len(signals) - n_bull - n_bear

        summary = f"🟢 {n_bull} Bullish  |  🔴 {n_bear} Bearish  |  🟡 {n_neut} Neutral"
        ax_verdict.text(5, 1.0, summary,
                        ha="center", va="center", fontsize=12, color=text_color)

        # Key points
        drivers = thesis.get("key_drivers", [])
        risks = thesis.get("key_risks", [])

        if drivers:
            driver_text = "Key drivers: " + drivers[0][:60]
            ax_verdict.text(5, 0.4, driver_text,
                            ha="center", va="center", fontsize=9, color=text_color,
                            style="italic")
    else:
        ax_verdict.text(5, 1.5, "THESIS ASSESSMENT PENDING",
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color="gray")
        ax_verdict.text(5, 0.8, "Run full pipeline for verdict",
                        ha="center", va="center", fontsize=11, color="gray")

    # ── Global title ──
    fig.suptitle(
        "EUR/USD Macro Analysis Dashboard",
        fontsize=18, fontweight="bold", color=COLORS["text"],
        y=0.98,
    )

    # ── Timestamp ──
    fig.text(0.99, 0.01, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
             ha="right", fontsize=8, color="gray")
    fig.text(0.01, 0.01, "Data: yfinance, FRED, ECB SDW, CFTC | Models: VAR, GLM",
             ha="left", fontsize=8, color="gray")

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "dashboard.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Dashboard module loaded. Run from main.py for full pipeline.")