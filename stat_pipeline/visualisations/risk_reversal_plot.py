# eur_usd_pipeline/visualisations/risk_reversal_plot.py
"""
25-Delta Risk Reversal visualisation for EUR/USD.

Shows whether the options market is paying a premium for
EUR calls (bullish) or puts (bearish), overlaid with spot.

Key interpretation:
  RR > 0 → Call premium → Market bullish EUR
  RR < 0 → Put premium → Market bearish EUR

Note: This uses our SYNTHETIC risk reversal proxy, not actual
options market data. Limitations are flagged on the chart.
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
    "rr_positive": "#2ca02c",
    "rr_negative": "#d62728",
    "rr_line": "#9467bd",
    "call_zone": "#c6efce",
    "put_zone": "#ffc7ce",
    "neutral_zone": "#fff2cc",
    "momentum": "#ff7f0e",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#333333",
    "annotation": "#555555",
}


def plot_risk_reversals(rr_df, eurusd_df, save=True, show=False):
    """
    Multi-panel risk reversal chart.

    Panel 1: EUR/USD spot vs RR proxy with regime bands
    Panel 2: RR z-score (how extreme is current sentiment)
    Panel 3: RR momentum (4-week change)

    Parameters
    ----------
    rr_df : pd.DataFrame
        From build_synthetic_risk_reversal(). Must have 'rr_25d_proxy'.
    eurusd_df : pd.DataFrame
        Must have 'eurusd_close'.
    """
    if rr_df.empty or "rr_25d_proxy" not in rr_df.columns:
        print("[!] Risk reversal data not available for plotting")
        return None

    merged = eurusd_df[["eurusd_close"]].join(rr_df, how="inner").dropna(
        subset=["eurusd_close", "rr_25d_proxy"]
    )

    if merged.empty:
        print("[!] No overlapping RR + EUR/USD data")
        return None

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.25)

    # ──────────────────────────────────────────────────────
    # PANEL 1: EUR/USD vs Risk Reversal with Regime Bands
    # ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLORS["bg"])

    # EUR/USD
    ax1.plot(merged.index, merged["eurusd_close"],
             color=COLORS["eurusd"], linewidth=2.0, label="EUR/USD Spot",
             zorder=4)
    ax1.set_ylabel("EUR/USD", color=COLORS["eurusd"],
                    fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=COLORS["eurusd"])

    # RR on secondary axis
    ax1b = ax1.twinx()
    rr = merged["rr_25d_proxy"]

    # Fill above/below zero
    ax1b.fill_between(merged.index, rr, 0,
                      where=(rr > 0), color=COLORS["call_zone"],
                      alpha=0.4, label="Call premium (EUR bullish)")
    ax1b.fill_between(merged.index, rr, 0,
                      where=(rr <= 0), color=COLORS["put_zone"],
                      alpha=0.4, label="Put premium (EUR bearish)")

    ax1b.plot(merged.index, rr, color=COLORS["rr_line"],
              linewidth=1.5, alpha=0.9, label="25Δ RR Proxy", zorder=3)

    ax1b.axhline(y=0, color="gray", linewidth=1, linestyle="-", alpha=0.6)

    # Regime bands
    for y_val, label in [(1.5, "Strong call premium"),
                          (-1.5, "Strong put premium")]:
        ax1b.axhline(y=y_val, color="gray", linewidth=0.6,
                      linestyle=":", alpha=0.4)
        ax1b.annotate(label, xy=(merged.index[5], y_val),
                      fontsize=7, color="gray", alpha=0.6,
                      verticalalignment="bottom" if y_val > 0 else "top")

    ax1b.set_ylabel("25Δ Risk Reversal (vol pts, synthetic)",
                     color=COLORS["rr_line"], fontsize=11, fontweight="bold")
    ax1b.tick_params(axis="y", labelcolor=COLORS["rr_line"])

    ax1.set_title(
        "EUR/USD vs 25-Delta Risk Reversal (Synthetic Proxy)\n"
        "Green = Call Premium (Bullish EUR) | Red = Put Premium (Bearish EUR)",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b,
               loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2, color=COLORS["grid"])

    # Current state annotation
    latest_rr = rr.iloc[-1]
    latest_signal = merged["rr_signal"].iloc[-1] if "rr_signal" in merged.columns else "N/A"
    latest_regime = merged["rr_regime"].iloc[-1] if "rr_regime" in merged.columns else "N/A"

    signal_color = COLORS["rr_positive"] if latest_rr > 0 else COLORS["rr_negative"]

    ax1.annotate(
        f"Current 25Δ RR: {latest_rr:.3f} vol pts\n"
        f"Signal: {latest_signal}\n"
        f"Regime: {latest_regime}\n"
        f"⚠ Synthetic proxy — not actual market data",
        xy=(0.98, 0.03), xycoords="axes fraction",
        fontsize=9, color=COLORS["annotation"],
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=signal_color, alpha=0.9, linewidth=2),
    )

    # ──────────────────────────────────────────────────────
    # PANEL 2: RR Z-Score
    # ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLORS["bg"])

    if "rr_zscore" in merged.columns:
        z = merged["rr_zscore"]

        ax2.plot(merged.index, z, color=COLORS["rr_line"],
                 linewidth=1.5, label="RR Z-Score (52w rolling)")

        # Fill extreme zones
        ax2.fill_between(merged.index, z, 0,
                         where=(z > 1.5), color=COLORS["call_zone"],
                         alpha=0.4, label="Extreme call premium")
        ax2.fill_between(merged.index, z, 0,
                         where=(z < -1.5), color=COLORS["put_zone"],
                         alpha=0.4, label="Extreme put premium")

        ax2.axhline(y=0, color="gray", linewidth=1, alpha=0.5)
        ax2.axhline(y=2, color=COLORS["rr_positive"], linewidth=0.6,
                     linestyle="--", alpha=0.4)
        ax2.axhline(y=-2, color=COLORS["rr_negative"], linewidth=0.6,
                     linestyle="--", alpha=0.4)

        # Current z marker
        current_z = z.iloc[-1] if not z.empty else np.nan
        if pd.notna(current_z):
            ax2.scatter([merged.index[-1]], [current_z],
                        color=COLORS["rr_line"], s=80, zorder=5,
                        edgecolors="black", linewidth=1)

        ax2.set_title("Risk Reversal Z-Score (extreme readings signal turning points)",
                      fontsize=11, fontweight="bold", pad=8)
        ax2.set_ylabel("Z-Score", fontsize=11)
        ax2.set_ylim(-3.5, 3.5)
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.2, color=COLORS["grid"])

    # ──────────────────────────────────────────────────────
    # PANEL 3: RR Momentum
    # ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(COLORS["bg"])

    if "rr_momentum" in merged.columns:
        mom = merged["rr_momentum"]

        mom_colors = [COLORS["rr_positive"] if v > 0 else COLORS["rr_negative"]
                      for v in mom]

        ax3.bar(merged.index, mom, width=5, color=mom_colors, alpha=0.5)
        ax3.axhline(y=0, color="gray", linewidth=1)

        # 8-week smoothed momentum
        mom_smooth = mom.rolling(8).mean()
        ax3.plot(merged.index, mom_smooth, color=COLORS["momentum"],
                 linewidth=1.5, label="8w smoothed momentum")

        ax3.set_title("Risk Reversal Momentum (4-Week Change)",
                      fontsize=11, fontweight="bold", pad=8)
        ax3.set_ylabel("Δ RR (4w)", fontsize=11)
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.2, color=COLORS["grid"])

        # Annotation
        latest_mom = mom.iloc[-1] if not mom.empty else np.nan
        if pd.notna(latest_mom):
            direction = "improving (more bullish)" if latest_mom > 0 else "deteriorating (more bearish)"
            ax3.annotate(
                f"Current momentum: {latest_mom:+.3f} ({direction})",
                xy=(0.98, 0.90), xycoords="axes fraction",
                fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.9),
            )

    # X-axis formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "risk_reversals.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# RR vs POSITIONING CROSS-CHART
# ──────────────────────────────────────────────────────────────

def plot_rr_vs_positioning(rr_df, cot_df, eurusd_df, save=True, show=False):
    """
    Scatter plot: risk reversal vs COT positioning.

    When both RR and positioning are at extremes in the same direction,
    the signal is strongest. When they diverge, watch out.
    """
    required_rr = "rr_25d_proxy" in rr_df.columns if not rr_df.empty else False
    required_cot = "net_spec_position" in cot_df.columns if not cot_df.empty else False

    if not required_rr or not required_cot:
        print("[!] Need both RR and COT data for cross-chart")
        return None

    merged = eurusd_df[["eurusd_close"]].join(
        rr_df[["rr_25d_proxy"]], how="inner"
    ).join(
        cot_df[["net_spec_position", "crowding_zscore"]], how="inner"
    ).dropna()

    if len(merged) < 20:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Colour by subsequent 13-week EUR/USD return
    future_return = merged["eurusd_close"].pct_change(13).shift(-13)
    merged["future_return"] = future_return

    has_returns = merged["future_return"].notna()
    plot_data = merged[has_returns]

    if len(plot_data) > 10:
        scatter = ax.scatter(
            plot_data["crowding_zscore"],
            plot_data["rr_25d_proxy"],
            c=plot_data["future_return"] * 100,
            cmap="RdYlGn",
            s=30, alpha=0.6,
            edgecolors="gray", linewidth=0.3,
            vmin=-10, vmax=10,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Subsequent 13w EUR/USD Return (%)", fontsize=10)
    else:
        ax.scatter(
            merged["crowding_zscore"], merged["rr_25d_proxy"],
            color=COLORS["rr_line"], s=30, alpha=0.6,
        )

    # Quadrant labels
    ax.axhline(y=0, color="gray", linewidth=1, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=1, alpha=0.5)

    # Quadrant annotations
    props = dict(fontsize=9, fontweight="bold", alpha=0.4, ha="center")
    ax.text(1.5, 1.0, "BULLISH\n(long + calls bid)", color=COLORS["rr_positive"], **props)
    ax.text(-1.5, -1.0, "BEARISH\n(short + puts bid)", color=COLORS["rr_negative"], **props)
    ax.text(-1.5, 1.0, "SQUEEZE SETUP\n(short + calls bid)", color=COLORS["momentum"], **props)
    ax.text(1.5, -1.0, "CROWDED + HEDGED\n(long + puts bid)", color="gray", **props)

    # Mark current position
    if len(merged) > 0:
        latest = merged.iloc[-1]
        ax.scatter(
            [latest["crowding_zscore"]], [latest["rr_25d_proxy"]],
            color="black", s=150, marker="*", zorder=5,
            label=f"Current ({merged.index[-1].strftime('%Y-%m-%d')})"
        )

    ax.set_xlabel("Positioning Crowding Z-Score (negative = short)", fontsize=11)
    ax.set_ylabel("25Δ Risk Reversal Proxy (positive = calls bid)", fontsize=11)
    ax.set_title(
        "Options Sentiment vs Futures Positioning\n"
        "Colour = subsequent 13-week EUR/USD return",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "rr_vs_positioning.png")
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
    from data.fetch_risk_reversals import fetch_all_risk_reversals
    from data.fetch_cot import fetch_cot_data

    fx = fetch_all_fx(save=False)
    rr = fetch_all_risk_reversals(save=False)
    cot = fetch_cot_data(save=False)

    plot_risk_reversals(rr, fx, save=True, show=True)
    plot_rr_vs_positioning(rr, cot, fx, save=True, show=True)