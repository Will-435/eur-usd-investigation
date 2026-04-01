# eur_usd_pipeline/visualisations/yield_spread_plot.py
"""
Yield Spread vs EUR/USD Spot Rate visualisation.

Plots the EUR/USD spot price overlaying the difference between the
2-year German Bund yield and the 2-year US Treasury yield.

Core thesis visual:
  - Spread narrowing (DE-US becoming less negative) → EUR appreciates
  - Spread widening (DE-US becoming more negative) → EUR depreciates

If the thesis holds, the two lines should move broadly together.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOT_DIR


# ──────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────

COLORS = {
    "eurusd": "#1f77b4",         # Blue
    "spread": "#d62728",         # Red
    "spread_fill_pos": "#c6efce",  # Light green (narrowing)
    "spread_fill_neg": "#ffc7ce",  # Light red (widening)
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#333333",
    "annotation": "#555555",
}


def plot_yield_spread_vs_eurusd(yield_df, eurusd_df, var_forecast=None,
                                 save=True, show=False):
    """
    Create a dual-axis chart: EUR/USD spot vs 2Y yield spread.

    Parameters
    ----------
    yield_df : pd.DataFrame
        Must contain 'yield_spread_2y'.
    eurusd_df : pd.DataFrame
        Must contain 'eurusd_close'.
    var_forecast : pd.DataFrame, optional
        VAR forecast with 'eurusd_close' for forward projection.
    save : bool
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Merge data ──
    merged = eurusd_df[["eurusd_close"]].join(
        yield_df[["yield_spread_2y"]], how="inner"
    ).dropna()

    if merged.empty:
        print("[!] No overlapping data for yield spread plot")
        return None

    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax1.set_facecolor(COLORS["bg"])

    # ── Primary axis: EUR/USD ──
    ax1.plot(
        merged.index, merged["eurusd_close"],
        color=COLORS["eurusd"], linewidth=2.0, label="EUR/USD Spot",
        zorder=3,
    )

    # Add VAR forecast if available
    if var_forecast is not None and "eurusd_close" in var_forecast.columns:
        # Connection line from last actual to first forecast
        ax1.plot(
            [merged.index[-1], var_forecast.index[0]],
            [merged["eurusd_close"].iloc[-1], var_forecast["eurusd_close"].iloc[0]],
            color=COLORS["eurusd"], linewidth=1.5, linestyle="--", alpha=0.6,
        )
        ax1.plot(
            var_forecast.index, var_forecast["eurusd_close"],
            color=COLORS["eurusd"], linewidth=2.0, linestyle="--",
            label="VAR Forecast", alpha=0.7, zorder=3,
        )

    ax1.set_ylabel("EUR/USD Spot Rate", color=COLORS["eurusd"],
                    fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=COLORS["eurusd"])

    # ── Secondary axis: Yield Spread ──
    ax2 = ax1.twinx()

    spread = merged["yield_spread_2y"]
    ax2.plot(
        merged.index, spread,
        color=COLORS["spread"], linewidth=1.5, label="2Y Yield Spread (DE-US)",
        alpha=0.85, zorder=2,
    )

    # ── Fill regions: narrowing vs widening ──
    # Narrowing (spread increasing / becoming less negative) = EUR positive
    spread_change = spread.diff()

    # Create filled regions showing narrowing (green) vs widening (red)
    ax2.fill_between(
        merged.index, spread, spread.rolling(26).mean(),
        where=(spread > spread.rolling(26).mean()),
        color=COLORS["spread_fill_pos"], alpha=0.3,
        label="Spread narrowing (EUR +)",
        zorder=1,
    )
    ax2.fill_between(
        merged.index, spread, spread.rolling(26).mean(),
        where=(spread <= spread.rolling(26).mean()),
        color=COLORS["spread_fill_neg"], alpha=0.3,
        label="Spread widening (EUR −)",
        zorder=1,
    )

    # Zero line on spread axis
    ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)

    ax2.set_ylabel("2Y Yield Spread (DE − US, %)",
                    color=COLORS["spread"], fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLORS["spread"])

    # ── Title and annotations ──
    ax1.set_title(
        "EUR/USD Spot vs 2-Year Yield Spread (German Bund − US Treasury)",
        fontsize=14, fontweight="bold", color=COLORS["text"], pad=20,
    )

    # ── Rolling correlation annotation ──
    corr_52w = merged["eurusd_close"].rolling(52, min_periods=26).corr(
        merged["yield_spread_2y"]
    )
    latest_corr = corr_52w.iloc[-1] if not corr_52w.empty else np.nan
    full_corr = merged["eurusd_close"].corr(merged["yield_spread_2y"])

    annotation_text = (
        f"Full-sample correlation: {full_corr:.3f}\n"
        f"Rolling 52w correlation: {latest_corr:.3f}\n"
        f"Latest spread: {spread.iloc[-1]:.2f}%"
    )

    ax1.annotate(
        annotation_text,
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=9, color=COLORS["annotation"],
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    # ── Thesis indicator ──
    recent_narrowing = spread.iloc[-1] > spread.iloc[-13] if len(spread) > 13 else False
    thesis_text = "THESIS: SUPPORTED ✓" if recent_narrowing else "THESIS: NOT CONFIRMED ✗"
    thesis_color = "#2d862d" if recent_narrowing else "#cc3333"

    ax1.annotate(
        thesis_text,
        xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=10, fontweight="bold", color=thesis_color,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=thesis_color, alpha=0.9),
    )

    # ── Key event annotations ──
    _annotate_events(ax1, merged)

    # ── Formatting ──
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.grid(True, alpha=0.3, color=COLORS["grid"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="lower left", fontsize=9, framealpha=0.9,
    )

    plt.tight_layout()

    # ── Save ──
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "yield_spread_vs_eurusd.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def _annotate_events(ax, data):
    """Add key macro event annotations to the chart."""
    events = [
        ("2020-03-15", "COVID-19\nPandemic"),
        ("2022-02-24", "Russia\nInvades\nUkraine"),
        ("2022-07-21", "ECB First\nRate Hike"),
        ("2023-10-01", "US 2Y\nYield Peak"),
    ]

    for date_str, label in events:
        try:
            event_date = pd.Timestamp(date_str)
            if event_date >= data.index.min() and event_date <= data.index.max():
                # Find nearest data point
                idx = data.index.get_indexer([event_date], method="nearest")[0]
                if idx >= 0 and idx < len(data):
                    y_val = data["eurusd_close"].iloc[idx]
                    ax.annotate(
                        label,
                        xy=(data.index[idx], y_val),
                        xytext=(0, 30),
                        textcoords="offset points",
                        fontsize=7, color=COLORS["annotation"],
                        ha="center",
                        arrowprops=dict(
                            arrowstyle="->", color=COLORS["annotation"],
                            alpha=0.6, lw=0.8,
                        ),
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="#ddd", alpha=0.8),
                    )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# ROLLING CORRELATION SUB-PLOT
# ──────────────────────────────────────────────────────────────

def plot_yield_spread_correlation(yield_df, eurusd_df, save=True, show=False):
    """
    Plot the rolling correlation between yield spread and EUR/USD.

    Shows how the relationship strength varies over time — useful for
    identifying regime changes.
    """
    merged = eurusd_df[["eurusd_close"]].join(
        yield_df[["yield_spread_2y"]], how="inner"
    ).dropna()

    if merged.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                     height_ratios=[2, 1],
                                     sharex=True)
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Top: Price + spread (compact version) ──
    ax1.set_facecolor(COLORS["bg"])
    ax1_twin = ax1.twinx()

    ax1.plot(merged.index, merged["eurusd_close"],
             color=COLORS["eurusd"], linewidth=1.8, label="EUR/USD")
    ax1_twin.plot(merged.index, merged["yield_spread_2y"],
                  color=COLORS["spread"], linewidth=1.2, alpha=0.7, label="2Y Spread")

    ax1.set_ylabel("EUR/USD", color=COLORS["eurusd"], fontsize=11)
    ax1_twin.set_ylabel("Spread (%)", color=COLORS["spread"], fontsize=11)
    ax1.set_title("Yield Spread Correlation Regime Analysis",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.2)

    # ── Bottom: Rolling correlations ──
    ax2.set_facecolor(COLORS["bg"])

    for window, style, label in [(26, "-", "26w rolling corr"),
                                   (52, "--", "52w rolling corr"),
                                   (104, ":", "104w rolling corr")]:
        corr = merged["eurusd_close"].rolling(window, min_periods=window // 2).corr(
            merged["yield_spread_2y"]
        )
        ax2.plot(merged.index, corr, linestyle=style, linewidth=1.5, label=label, alpha=0.8)

    ax2.axhline(y=0, color="gray", linewidth=1, linestyle="-")
    ax2.axhline(y=0.5, color="green", linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.axhline(y=-0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.4)

    # Fill positive / negative correlation regimes
    corr_52 = merged["eurusd_close"].rolling(52, min_periods=26).corr(
        merged["yield_spread_2y"]
    )
    ax2.fill_between(merged.index, corr_52, 0,
                     where=(corr_52 > 0), color="green", alpha=0.1)
    ax2.fill_between(merged.index, corr_52, 0,
                     where=(corr_52 <= 0), color="red", alpha=0.1)

    ax2.set_ylabel("Correlation", fontsize=11)
    ax2.set_ylim(-1, 1)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.2)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "yield_spread_correlation.png")
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
    from data.fetch_fx import fetch_all_fx
    from data.fetch_yields import fetch_all_yields

    fx = fetch_all_fx(save=False)
    yields = fetch_all_yields(save=False)

    plot_yield_spread_vs_eurusd(yields, fx, save=True, show=True)
    plot_yield_spread_correlation(yields, fx, save=True, show=True)