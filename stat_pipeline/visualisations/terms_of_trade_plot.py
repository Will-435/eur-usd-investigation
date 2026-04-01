# eur_usd_pipeline/visualisations/terms_of_trade_plot.py
"""
Terms of Trade visualisation: Brent Crude (inverted) vs EUR/USD.

Core thesis visual:
  If oil prices drop, the Eurozone benefits from improved terms
  of trade (lower import costs), historically supporting EUR.

  We invert Brent so that rising line = falling oil = EUR positive.
  If the two lines track together, the oil-EUR relationship holds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOT_DIR


COLORS = {
    "eurusd": "#1f77b4",
    "brent": "#ff7f0e",
    "brent_inv": "#2ca02c",
    "corr_pos": "#2ca02c",
    "corr_neg": "#d62728",
    "energy_idx": "#9467bd",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#333333",
    "annotation": "#555555",
}


def plot_terms_of_trade(oil_df, eurusd_df, save=True, show=False):
    """
    Multi-panel Terms of Trade chart:
      Panel 1: EUR/USD vs Inverted Brent (normalised for overlay)
      Panel 2: EUR/USD vs Brent (original, showing inverse relationship)
      Panel 3: Rolling correlation

    Parameters
    ----------
    oil_df : pd.DataFrame
        Must contain 'brent_close', 'brent_inverted'.
    eurusd_df : pd.DataFrame
        Must contain 'eurusd_close'.
    """
    merged = eurusd_df[["eurusd_close"]].join(
        oil_df[["brent_close", "brent_inverted"]], how="inner"
    ).dropna()

    if merged.empty:
        print("[!] No overlapping data for ToT plot")
        return None

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = GridSpec(3, 1, height_ratios=[3, 2, 1.5], hspace=0.25)

    # ──────────────────────────────────────────────────────
    # PANEL 1: EUR/USD vs Inverted Brent (normalised)
    # ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLORS["bg"])

    # Normalise both to z-scores for visual comparison
    eur_z = (merged["eurusd_close"] - merged["eurusd_close"].mean()) / merged["eurusd_close"].std()
    inv_z = (merged["brent_inverted"] - merged["brent_inverted"].mean()) / merged["brent_inverted"].std()

    ax1.plot(merged.index, eur_z, color=COLORS["eurusd"],
             linewidth=2.0, label="EUR/USD (z-score)", zorder=3)
    ax1.plot(merged.index, inv_z, color=COLORS["brent_inv"],
             linewidth=1.8, label="Inverted Brent (z-score)",
             alpha=0.85, zorder=2)

    # Highlight periods where they converge/diverge
    diff = eur_z - inv_z
    ax1.fill_between(
        merged.index, eur_z, inv_z,
        where=(eur_z >= inv_z),
        color=COLORS["eurusd"], alpha=0.08,
        label="EUR above inverted oil",
    )
    ax1.fill_between(
        merged.index, eur_z, inv_z,
        where=(eur_z < inv_z),
        color=COLORS["brent_inv"], alpha=0.08,
        label="Inverted oil above EUR",
    )

    ax1.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Z-Score (normalised)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Terms of Trade: EUR/USD vs Inverted Brent Crude\n"
        "(↑ Inverted Brent = ↓ Oil Price = EUR Positive)",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2, color=COLORS["grid"])

    # Correlation annotation
    full_corr = merged["eurusd_close"].corr(merged["brent_inverted"])
    inv_corr = merged["eurusd_close"].corr(merged["brent_close"])

    ax1.annotate(
        f"EUR vs Inverted Oil: r = {full_corr:.3f}\n"
        f"EUR vs Oil (direct): r = {inv_corr:.3f}\n"
        f"→ {'Positive relationship confirmed' if full_corr > 0.2 else 'Weak/No relationship'}",
        xy=(0.98, 0.03), xycoords="axes fraction",
        fontsize=9, color=COLORS["annotation"],
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#ccc", alpha=0.9),
    )

    # ──────────────────────────────────────────────────────
    # PANEL 2: EUR/USD vs Brent (original scales, dual axis)
    # ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLORS["bg"])
    ax2_twin = ax2.twinx()

    ax2.plot(merged.index, merged["eurusd_close"],
             color=COLORS["eurusd"], linewidth=1.8, label="EUR/USD (LHS)")
    ax2_twin.plot(merged.index, merged["brent_close"],
                  color=COLORS["brent"], linewidth=1.5, alpha=0.8, label="Brent Crude (RHS)")

    ax2.set_ylabel("EUR/USD", color=COLORS["eurusd"], fontsize=11)
    ax2_twin.set_ylabel("Brent ($/bbl)", color=COLORS["brent"], fontsize=11)
    ax2.tick_params(axis="y", labelcolor=COLORS["eurusd"])
    ax2_twin.tick_params(axis="y", labelcolor=COLORS["brent"])

    # Invert Brent axis to visually show the inverse relationship
    ax2_twin.invert_yaxis()

    ax2.set_title("EUR/USD vs Brent Crude (Brent axis INVERTED)",
                  fontsize=11, fontweight="bold", pad=10)

    lines2a, labels2a = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines2a + lines2b, labels2a + labels2b,
               loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ── Annotate oil regime periods ──
    _annotate_oil_events(ax2, merged)

    # ──────────────────────────────────────────────────────
    # PANEL 3: Rolling Correlation
    # ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(COLORS["bg"])

    # Correlation between EUR/USD and inverted Brent (should be positive)
    corr_26 = merged["eurusd_close"].rolling(26, min_periods=13).corr(
        merged["brent_inverted"]
    )
    corr_52 = merged["eurusd_close"].rolling(52, min_periods=26).corr(
        merged["brent_inverted"]
    )

    ax3.plot(merged.index, corr_26, color=COLORS["corr_pos"],
             linewidth=1.2, alpha=0.6, label="26w rolling corr")
    ax3.plot(merged.index, corr_52, color=COLORS["corr_pos"],
             linewidth=1.8, label="52w rolling corr")

    ax3.fill_between(merged.index, corr_52, 0,
                     where=(corr_52 > 0), color=COLORS["corr_pos"], alpha=0.15)
    ax3.fill_between(merged.index, corr_52, 0,
                     where=(corr_52 <= 0), color=COLORS["corr_neg"], alpha=0.15)

    ax3.axhline(y=0, color="gray", linewidth=1, linestyle="-")
    ax3.set_ylabel("Correlation", fontsize=11)
    ax3.set_ylim(-1, 1)
    ax3.set_title("Rolling Correlation: EUR/USD vs Inverted Brent",
                  fontsize=11, fontweight="bold", pad=8)
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, alpha=0.2)

    # Thesis verdict
    recent_corr = corr_52.iloc[-1] if not corr_52.empty else np.nan
    thesis_holds = recent_corr > 0.15 if pd.notna(recent_corr) else False
    verdict = "OIL THESIS: SUPPORTED ✓" if thesis_holds else "OIL THESIS: WEAK/NOT CONFIRMED ✗"
    verdict_color = "#2d862d" if thesis_holds else "#cc3333"

    ax3.annotate(
        verdict,
        xy=(0.98, 0.95), xycoords="axes fraction",
        fontsize=10, fontweight="bold", color=verdict_color,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=verdict_color, alpha=0.9),
    )

    # X-axis formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "terms_of_trade.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def _annotate_oil_events(ax, data):
    """Annotate key oil market events."""
    events = [
        ("2020-04-20", "Oil Futures\nGo Negative"),
        ("2022-03-08", "Brent Hits\n$130/bbl"),
        ("2023-06-04", "OPEC+\nCuts"),
    ]

    for date_str, label in events:
        try:
            event_date = pd.Timestamp(date_str)
            if event_date >= data.index.min() and event_date <= data.index.max():
                idx = data.index.get_indexer([event_date], method="nearest")[0]
                if 0 <= idx < len(data):
                    y_val = data["eurusd_close"].iloc[idx]
                    ax.annotate(
                        label,
                        xy=(data.index[idx], y_val),
                        xytext=(0, 25),
                        textcoords="offset points",
                        fontsize=7, color=COLORS["annotation"],
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color=COLORS["annotation"],
                                        alpha=0.6, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="#ddd", alpha=0.8),
                    )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# ENERGY PRESSURE INDEX PLOT
# ──────────────────────────────────────────────────────────────

def plot_energy_pressure(oil_df, eurusd_df, save=True, show=False):
    """
    Plot the energy pressure index vs EUR/USD.

    Higher pressure = worse for EUR (rising oil costs).
    """
    if "energy_pressure_index" not in oil_df.columns:
        print("[!] Energy pressure index not available")
        return None

    merged = eurusd_df[["eurusd_close"]].join(
        oil_df[["energy_pressure_index"]], how="inner"
    ).dropna()

    if merged.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax1.set_facecolor(COLORS["bg"])

    ax1.plot(merged.index, merged["eurusd_close"],
             color=COLORS["eurusd"], linewidth=1.8, label="EUR/USD")
    ax1.set_ylabel("EUR/USD", color=COLORS["eurusd"], fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(merged.index, merged["energy_pressure_index"],
             color=COLORS["energy_idx"], linewidth=1.5, alpha=0.8,
             label="Energy Pressure Index")

    # Invert: high pressure should visually align with low EUR
    ax2.invert_yaxis()
    ax2.set_ylabel("Energy Pressure (inverted)", color=COLORS["energy_idx"], fontsize=11)

    ax1.set_title("EUR/USD vs Energy Pressure Index (inverted)",
                  fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.2)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "energy_pressure.png")
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
    from data.fetch_oil import fetch_all_oil

    fx = fetch_all_fx(save=False)
    oil = fetch_all_oil(save=False)

    plot_terms_of_trade(oil, fx, save=True, show=True)
    plot_energy_pressure(oil, fx, save=True, show=True)