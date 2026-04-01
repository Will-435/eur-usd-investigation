# eur_usd_pipeline/visualisations/cot_plot.py
"""
Commitments of Traders (COT) positioning visualisation.

Shows how hedge funds and speculators are positioned in Euro FX futures,
overlaid with EUR/USD spot to identify:

  - Short squeeze setups (heavy shorts + start of covering)
  - Crowded trade risk (extreme longs)
  - Positioning-driven turning points

If specs are heavily short and any catalyst triggers covering,
the resulting short squeeze can propel EUR/USD significantly higher.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOT_DIR


COLORS = {
    "eurusd": "#1f77b4",
    "net_long": "#2ca02c",
    "net_short": "#d62728",
    "squeeze_zone": "#ff9999",
    "crowded_zone": "#99ccff",
    "neutral": "#cccccc",
    "zscore": "#9467bd",
    "dealer": "#ff7f0e",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#333333",
    "annotation": "#555555",
}


def plot_cot_positioning(cot_df, eurusd_df, save=True, show=False):
    """
    Multi-panel COT positioning chart.

    Panel 1: EUR/USD spot with positioning overlay (bar chart)
    Panel 2: Crowding z-score with squeeze/crowded zones
    Panel 3: Weekly position changes (flow)

    Parameters
    ----------
    cot_df : pd.DataFrame
        From fetch_cot_data(). Must have 'net_spec_position'.
    eurusd_df : pd.DataFrame
        Must have 'eurusd_close'.
    """
    if cot_df.empty or "net_spec_position" not in cot_df.columns:
        print("[!] COT data not available for plotting")
        return None

    # Align data
    merged = eurusd_df[["eurusd_close"]].join(
        cot_df, how="inner"
    ).dropna(subset=["eurusd_close", "net_spec_position"])

    if merged.empty:
        print("[!] No overlapping COT + EUR/USD data")
        return None

    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = GridSpec(3, 1, height_ratios=[3, 2, 1.5], hspace=0.25)

    # ──────────────────────────────────────────────────────
    # PANEL 1: EUR/USD vs Net Speculative Position
    # ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLORS["bg"])

    # EUR/USD line
    ax1.plot(merged.index, merged["eurusd_close"],
             color=COLORS["eurusd"], linewidth=2.0, label="EUR/USD Spot",
             zorder=4)

    ax1.set_ylabel("EUR/USD Spot Rate", color=COLORS["eurusd"],
                    fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=COLORS["eurusd"])

    # Net position as bars on secondary axis
    ax1b = ax1.twinx()

    net_pos = merged["net_spec_position"]
    bar_colors = [COLORS["net_long"] if v >= 0 else COLORS["net_short"]
                  for v in net_pos]

    ax1b.bar(merged.index, net_pos, width=5, color=bar_colors,
             alpha=0.4, label="Net Spec Position", zorder=1)

    ax1b.axhline(y=0, color="gray", linewidth=1, linestyle="-", alpha=0.5)
    ax1b.set_ylabel("Net Speculative Position (contracts)",
                     fontsize=11, fontweight="bold")

    # ── Highlight squeeze risk zones ──
    if "squeeze_risk" in merged.columns:
        squeeze_mask = merged["squeeze_risk"].isin(["HIGH", "ELEVATED"])
        if squeeze_mask.any():
            squeeze_dates = merged.index[squeeze_mask]
            for date in squeeze_dates:
                ax1.axvspan(date - pd.Timedelta(days=3),
                           date + pd.Timedelta(days=3),
                           color=COLORS["squeeze_zone"], alpha=0.15, zorder=0)

    ax1.set_title(
        "EUR/USD vs Speculative Positioning (CFTC COT Report)\n"
        "Green bars = Net LONG | Red bars = Net SHORT | Pink zones = Squeeze Risk",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )

    # Legend
    long_patch = mpatches.Patch(color=COLORS["net_long"], alpha=0.5, label="Net Long")
    short_patch = mpatches.Patch(color=COLORS["net_short"], alpha=0.5, label="Net Short")
    squeeze_patch = mpatches.Patch(color=COLORS["squeeze_zone"], alpha=0.3, label="Squeeze Risk Zone")
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=lines1 + [long_patch, short_patch, squeeze_patch],
               loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2, color=COLORS["grid"])

    # Positioning summary annotation
    latest_pos = net_pos.iloc[-1]
    pos_direction = "LONG" if latest_pos > 0 else "SHORT"
    pct_rank = merged.get("positioning_percentile")
    rank_str = f"{pct_rank.iloc[-1]:.0f}th pctl" if pct_rank is not None and not pct_rank.empty else "N/A"

    ax1.annotate(
        f"Latest: {latest_pos:,.0f} contracts ({pos_direction})\n"
        f"Historical rank: {rank_str}\n"
        f"Squeeze risk: {merged.get('squeeze_risk', pd.Series(['N/A'])).iloc[-1]}",
        xy=(0.98, 0.03), xycoords="axes fraction",
        fontsize=9, color=COLORS["annotation"],
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#ccc", alpha=0.9),
    )

    # ──────────────────────────────────────────────────────
    # PANEL 2: Crowding Z-Score
    # ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLORS["bg"])

    if "crowding_zscore" in merged.columns:
        z = merged["crowding_zscore"]

        ax2.plot(merged.index, z, color=COLORS["zscore"],
                 linewidth=1.5, label="Crowding Z-Score", zorder=3)

        # Fill extreme zones
        ax2.fill_between(merged.index, z, -1.5,
                         where=(z < -1.5), color=COLORS["net_short"],
                         alpha=0.2, label="Extreme SHORT (squeeze risk)")
        ax2.fill_between(merged.index, z, 1.5,
                         where=(z > 1.5), color=COLORS["net_long"],
                         alpha=0.2, label="Extreme LONG (crowded trade)")

        # Reference lines
        ax2.axhline(y=0, color="gray", linewidth=1, linestyle="-", alpha=0.5)
        ax2.axhline(y=1.5, color=COLORS["net_long"], linewidth=0.8,
                     linestyle="--", alpha=0.5)
        ax2.axhline(y=-1.5, color=COLORS["net_short"], linewidth=0.8,
                     linestyle="--", alpha=0.5)
        ax2.axhline(y=2.0, color=COLORS["net_long"], linewidth=0.8,
                     linestyle=":", alpha=0.4)
        ax2.axhline(y=-2.0, color=COLORS["net_short"], linewidth=0.8,
                     linestyle=":", alpha=0.4)

        # Current z-score marker
        current_z = z.iloc[-1]
        ax2.scatter([merged.index[-1]], [current_z], color=COLORS["zscore"],
                    s=80, zorder=5, edgecolors="black", linewidth=1)
        ax2.annotate(
            f"Current: z = {current_z:.2f}",
            xy=(merged.index[-1], current_z),
            xytext=(-80, 15), textcoords="offset points",
            fontsize=9, fontweight="bold", color=COLORS["zscore"],
            arrowprops=dict(arrowstyle="->", color=COLORS["zscore"]),
        )

        ax2.set_title("Positioning Crowding Z-Score (2-Year Rolling)",
                      fontsize=11, fontweight="bold", pad=10)
        ax2.set_ylabel("Z-Score", fontsize=11)
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.2, color=COLORS["grid"])
        ax2.set_ylim(-3.5, 3.5)

    # ──────────────────────────────────────────────────────
    # PANEL 3: Weekly Position Changes (Flow)
    # ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(COLORS["bg"])

    if "spec_position_change" in merged.columns:
        changes = merged["spec_position_change"]
        change_colors = [COLORS["net_long"] if v >= 0 else COLORS["net_short"]
                         for v in changes]

        ax3.bar(merged.index, changes, width=5, color=change_colors, alpha=0.6)
        ax3.axhline(y=0, color="gray", linewidth=1)

        # 4-week moving average of changes
        ma4 = changes.rolling(4).mean()
        ax3.plot(merged.index, ma4, color=COLORS["zscore"],
                 linewidth=1.5, label="4-week avg change")

        ax3.set_title("Weekly Change in Net Speculative Position",
                      fontsize=11, fontweight="bold", pad=8)
        ax3.set_ylabel("Contracts (weekly Δ)", fontsize=11)
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.2, color=COLORS["grid"])

        # Annotation: recent flow direction
        recent_flow = changes.iloc[-4:].mean()
        flow_dir = "adding longs" if recent_flow > 0 else "adding shorts"
        ax3.annotate(
            f"Recent 4w avg flow: {recent_flow:+,.0f} ({flow_dir})",
            xy=(0.98, 0.95), xycoords="axes fraction",
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
        filepath = os.path.join(PLOT_DIR, "cot_positioning.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# DEALER vs SPEC DIVERGENCE PLOT
# ──────────────────────────────────────────────────────────────

def plot_dealer_vs_spec(cot_df, eurusd_df, save=True, show=False):
    """
    Plot dealer positioning vs speculative positioning.

    Dealers (market makers) often take the opposite side of specs.
    Large divergence where dealers are long and specs are short
    can be a strong bullish signal — "smart money" vs "crowd".
    """
    required = ["net_spec_position", "net_dealer_position"]
    if not all(c in cot_df.columns for c in required):
        print("[!] Dealer + spec data not available")
        return None

    merged = eurusd_df[["eurusd_close"]].join(cot_df[required], how="inner").dropna()

    if merged.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                     height_ratios=[2, 1], sharex=True)
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Top: EUR/USD with dealer and spec bars ──
    ax1.set_facecolor(COLORS["bg"])
    ax1.plot(merged.index, merged["eurusd_close"],
             color=COLORS["eurusd"], linewidth=2, label="EUR/USD", zorder=4)
    ax1.set_ylabel("EUR/USD", color=COLORS["eurusd"], fontsize=11)

    ax1b = ax1.twinx()
    width = 3

    # Offset bars slightly so they don't overlap
    spec_dates = merged.index - pd.Timedelta(days=2)
    dealer_dates = merged.index + pd.Timedelta(days=2)

    spec_colors = [COLORS["net_long"] if v >= 0 else COLORS["net_short"]
                   for v in merged["net_spec_position"]]
    dealer_colors = [COLORS["dealer"] for _ in merged["net_dealer_position"]]

    ax1b.bar(spec_dates, merged["net_spec_position"], width=width,
             color=spec_colors, alpha=0.35, label="Spec Position")
    ax1b.bar(dealer_dates, merged["net_dealer_position"], width=width,
             color=dealer_colors, alpha=0.35, label="Dealer Position")

    ax1b.axhline(y=0, color="gray", linewidth=1, alpha=0.5)
    ax1b.set_ylabel("Net Position (contracts)", fontsize=11)

    ax1.set_title("Dealer vs Speculative Positioning — 'Smart Money' Divergence",
                  fontsize=13, fontweight="bold", pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # ── Bottom: Divergence ──
    ax2.set_facecolor(COLORS["bg"])
    divergence = merged["net_dealer_position"] - merged["net_spec_position"]

    div_colors = [COLORS["net_long"] if v > 0 else COLORS["net_short"]
                  for v in divergence]
    ax2.bar(merged.index, divergence, width=5, color=div_colors, alpha=0.5)
    ax2.axhline(y=0, color="gray", linewidth=1)

    # Rolling mean
    ax2.plot(merged.index, divergence.rolling(13).mean(),
             color=COLORS["zscore"], linewidth=1.5, label="13w avg divergence")

    ax2.set_title("Dealer − Spec Divergence (positive = dealers more bullish than specs)",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylabel("Divergence (contracts)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "dealer_vs_spec.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# POSITIONING PERCENTILE HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_positioning_heatmap(cot_df, save=True, show=False):
    """
    Heatmap showing historical positioning percentile over time.
    Useful for quickly seeing where current positioning sits
    relative to history.
    """
    if "positioning_percentile" not in cot_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    pctl = cot_df["positioning_percentile"].dropna()

    # Create a colour-mapped scatter
    scatter = ax.scatter(
        pctl.index, [0.5] * len(pctl),
        c=pctl.values, cmap="RdYlGn",
        s=8, marker="s", vmin=0, vmax=100,
    )

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Speculative Positioning Percentile (0th = max short, 100th = max long)",
                 fontsize=11, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal",
                        pad=0.3, aspect=40)
    cbar.set_label("Percentile Rank", fontsize=10)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Mark current position
    current_pctl = pctl.iloc[-1]
    ax.annotate(
        f"NOW: {current_pctl:.0f}th pctl",
        xy=(pctl.index[-1], 0.5),
        xytext=(0, 20), textcoords="offset points",
        fontsize=9, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="black", alpha=0.9),
    )

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "positioning_heatmap.png")
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
    from data.fetch_cot import fetch_cot_data

    fx = fetch_all_fx(save=False)
    cot = fetch_cot_data(save=False)

    plot_cot_positioning(cot, fx, save=True, show=True)
    plot_dealer_vs_spec(cot, fx, save=True, show=True)
    plot_positioning_heatmap(cot, save=True, show=True)