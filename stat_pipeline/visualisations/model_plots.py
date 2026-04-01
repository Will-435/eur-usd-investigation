# eur_usd_pipeline/visualisations/model_plots.py
"""
Model output visualisations:

  - VAR forecast with confidence intervals
  - Impulse Response Functions (IRFs)
  - Forecast Error Variance Decomposition (FEVD)
  - GLM coefficient importance
  - VAR vs GLM forecast comparison
  - Granger causality heatmap
  - Rolling correlation heatmap
  - Residual diagnostic plots
  - Scenario analysis chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOT_DIR


COLORS = {
    "var": "#1f77b4",
    "glm": "#ff7f0e",
    "ensemble": "#2ca02c",
    "actual": "#333333",
    "ci_fill": "#a6cee3",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#cccccc",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
    "text": "#333333",
}


# ──────────────────────────────────────────────────────────────
# VAR FORECAST PLOT
# ──────────────────────────────────────────────────────────────

def plot_var_forecast(var_results, eurusd_df, save=True, show=False):
    """
    Plot VAR forecast for EUR/USD with confidence intervals.

    Shows historical data + forward projections for 12m and 24m.
    """
    var_model = var_results.get("model")
    forecasts = var_results.get("forecasts", {})
    forecast_cis = var_results.get("forecast_cis", {})

    if var_model is None or not forecasts:
        print("[!] VAR results not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # ── Historical EUR/USD ──
    if "eurusd_close" in eurusd_df.columns:
        hist = eurusd_df["eurusd_close"].dropna()
        # Show last 3 years of history for context
        cutoff = hist.index.max() - pd.DateOffset(years=3)
        hist_plot = hist[hist.index >= cutoff]

        ax.plot(hist_plot.index, hist_plot.values,
                color=COLORS["actual"], linewidth=2.0,
                label="Historical EUR/USD", zorder=4)

    # ── Forecast lines ──
    forecast_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    forecast_styles = ["-", "--", ":"]

    for i, (label, fc) in enumerate(forecasts.items()):
        if "eurusd_close" not in fc.columns:
            continue

        color = forecast_colors[i % len(forecast_colors)]
        style = forecast_styles[i % len(forecast_styles)]

        # Connect historical to forecast
        if "eurusd_close" in eurusd_df.columns:
            connect_date = eurusd_df["eurusd_close"].dropna().index[-1]
            connect_val = eurusd_df["eurusd_close"].dropna().iloc[-1]
            ax.plot(
                [connect_date, fc.index[0]],
                [connect_val, fc["eurusd_close"].iloc[0]],
                color=color, linewidth=1.5, linestyle=style, alpha=0.7,
            )

        ax.plot(fc.index, fc["eurusd_close"],
                color=color, linewidth=2.0, linestyle=style,
                label=f"VAR Forecast ({label})", zorder=3)

        # Confidence intervals
        ci = forecast_cis.get(label)
        if ci is not None:
            lower = ci["lower"]["eurusd_close"] if "eurusd_close" in ci["lower"].columns else None
            upper = ci["upper"]["eurusd_close"] if "eurusd_close" in ci["upper"].columns else None

            if lower is not None and upper is not None:
                ax.fill_between(
                    fc.index, lower, upper,
                    color=color, alpha=0.1,
                    label=f"95% CI ({label})",
                )

        # End-point annotation
        end_val = fc["eurusd_close"].iloc[-1]
        current = var_model.last_levels.get("eurusd_close", end_val)
        chg_pct = (end_val - current) / current * 100
        direction = "▲" if chg_pct > 0 else "▼"

        ax.annotate(
            f"{end_val:.4f}\n({chg_pct:+.1f}%) {direction}",
            xy=(fc.index[-1], end_val),
            xytext=(15, 0), textcoords="offset points",
            fontsize=9, fontweight="bold", color=color,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.9),
        )

    # ── Current level marker ──
    if var_model and hasattr(var_model, "last_levels"):
        current = var_model.last_levels.get("eurusd_close")
        if current:
            ax.axhline(y=current, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.4)
            ax.annotate(
                f"Current: {current:.4f}",
                xy=(ax.get_xlim()[0], current),
                fontsize=8, color="gray", va="bottom",
            )

    ax.set_title("EUR/USD VAR Model Forecast with 95% Confidence Intervals",
                 fontsize=14, fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_ylabel("EUR/USD", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "var_forecast.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# IMPULSE RESPONSE FUNCTIONS
# ──────────────────────────────────────────────────────────────

def plot_irf(var_results, target="eurusd_close", periods=26,
             save=True, show=False):
    """
    Plot impulse response functions: how EUR/USD responds to
    shocks in each variable.
    """
    irf_result = var_results.get("irf_result")
    var_model = var_results.get("model")

    if irf_result is None or var_model is None:
        print("[!] IRF results not available")
        return None

    variables = var_model.variables
    if target not in variables:
        print(f"[!] '{target}' not in VAR variables")
        return None

    target_idx = variables.index(target)
    n_vars = len(variables)
    other_vars = [v for v in variables if v != target]

    # Grid layout
    n_plots = len(other_vars)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor(COLORS["bg"])

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, impulse_var in enumerate(other_vars):
        ax = axes[i]
        ax.set_facecolor(COLORS["bg"])

        impulse_idx = variables.index(impulse_var)

        # IRF values: response of target to shock in impulse_var
        irf_vals = irf_result.irfs[:periods + 1, target_idx, impulse_idx]

        # Confidence intervals
        try:
            lower = irf_result.ci[:periods + 1, target_idx, impulse_idx, 0]
            upper = irf_result.ci[:periods + 1, target_idx, impulse_idx, 1]
            has_ci = True
        except (AttributeError, IndexError):
            has_ci = False

        x = range(len(irf_vals))

        # Plot IRF
        ax.plot(x, irf_vals, color=COLORS["var"], linewidth=2.0)

        if has_ci:
            ax.fill_between(x, lower, upper, color=COLORS["ci_fill"], alpha=0.3)

        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")

        # Color the area based on sign
        ax.fill_between(x, irf_vals, 0,
                         where=(np.array(irf_vals) > 0),
                         color=COLORS["positive"], alpha=0.1)
        ax.fill_between(x, irf_vals, 0,
                         where=(np.array(irf_vals) <= 0),
                         color=COLORS["negative"], alpha=0.1)

        # Clean variable name for title
        clean_name = impulse_var.replace("_", " ").title()
        ax.set_title(f"Shock: {clean_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Weeks", fontsize=9)
        ax.set_ylabel("EUR/USD Response", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Peak response annotation
        peak_idx = np.argmax(np.abs(irf_vals))
        peak_val = irf_vals[peak_idx]
        ax.annotate(
            f"Peak: {peak_val:.5f}\n(week {peak_idx})",
            xy=(peak_idx, peak_val),
            xytext=(5, 10 if peak_val > 0 else -20),
            textcoords="offset points",
            fontsize=7, color=COLORS["text"],
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Impulse Response Functions: EUR/USD Response to 1-σ Shocks\n"
        f"(Orthogonalised, {periods}-week horizon)",
        fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02,
    )

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "impulse_response.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# FORECAST ERROR VARIANCE DECOMPOSITION
# ──────────────────────────────────────────────────────────────

def plot_fevd(var_results, save=True, show=False):
    """
    Stacked area chart of forecast error variance decomposition.

    Shows what percentage of EUR/USD forecast variance is explained
    by each variable at different horizons.
    """
    fevd_table = var_results.get("fevd_table")

    if fevd_table is None or fevd_table.empty:
        print("[!] FEVD data not available")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Stacked area chart
    cmap = plt.cm.Set3
    colors = [cmap(i / len(fevd_table.columns))
              for i in range(len(fevd_table.columns))]

    ax.stackplot(
        fevd_table.index,
        *[fevd_table[col] * 100 for col in fevd_table.columns],
        labels=[c.replace("_", " ").title() for c in fevd_table.columns],
        colors=colors,
        alpha=0.8,
    )

    # Key horizon markers
    for h in [4, 13, 26, 52]:
        if h in fevd_table.index:
            ax.axvline(x=h, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
            ax.annotate(f"{h}w", xy=(h, 102), fontsize=8, color="gray",
                        ha="center", va="bottom")

    ax.set_title(
        "Forecast Error Variance Decomposition for EUR/USD\n"
        "(What explains EUR/USD forecast uncertainty at each horizon?)",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.set_xlabel("Forecast Horizon (weeks)", fontsize=11)
    ax.set_ylabel("Variance Explained (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(1, fevd_table.index.max())

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y", color=COLORS["grid"])

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "fevd.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# GRANGER CAUSALITY HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_granger_heatmap(granger_matrix, save=True, show=False):
    """
    Heatmap of pairwise Granger causality p-values.

    Cell (i, j) shows p-value for 'column j Granger-causes row i'.
    Lower p-value (darker) = stronger causal relationship.
    """
    if granger_matrix is None or granger_matrix.empty:
        print("[!] Granger matrix not available")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(COLORS["bg"])

    # Transform for better visual contrast: -log10(p)
    # Higher value = more significant
    display_matrix = -np.log10(granger_matrix.clip(lower=1e-10))

    # Clean labels
    clean_labels = [c.replace("_", " ").title()[:20] for c in granger_matrix.columns]

    # Mask diagonal
    mask = np.eye(len(granger_matrix), dtype=bool)

    sns.heatmap(
        display_matrix,
        ax=ax,
        mask=mask,
        annot=granger_matrix.round(3).values,
        fmt="",
        cmap="YlOrRd",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "−log₁₀(p-value)  [higher = more significant]"},
        vmin=0,
        vmax=5,
    )

    # Mark significant cells
    for i in range(len(granger_matrix)):
        for j in range(len(granger_matrix.columns)):
            if i != j:
                p_val = granger_matrix.iloc[i, j]
                if p_val < 0.01:
                    ax.text(j + 0.5, i + 0.85, "***", ha="center",
                            fontsize=8, color="white", fontweight="bold")
                elif p_val < 0.05:
                    ax.text(j + 0.5, i + 0.85, "**", ha="center",
                            fontsize=8, color="white", fontweight="bold")
                elif p_val < 0.10:
                    ax.text(j + 0.5, i + 0.85, "*", ha="center",
                            fontsize=8, color="black")

    ax.set_title(
        "Granger Causality Matrix\n"
        "Cell (row, col): p-value for 'column Granger-causes row'\n"
        "Annotation shows raw p-value; *** p<0.01  ** p<0.05  * p<0.10",
        fontsize=12, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.set_xlabel("CAUSE →", fontsize=11, fontweight="bold")
    ax.set_ylabel("← EFFECT", fontsize=11, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "granger_heatmap.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# GLM FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────

def plot_glm_importance(glm_results, horizon="13w", save=True, show=False):
    """
    Horizontal bar chart of GLM feature importance.

    Colour-coded by direction (green = EUR positive, red = EUR negative).
    """
    horizon_data = glm_results.get(horizon)
    if horizon_data is None:
        print(f"[!] GLM results not available for horizon '{horizon}'")
        return None

    best_model = horizon_data.get("best_model")
    if best_model is None:
        return None

    try:
        importance = best_model.feature_importance()
    except Exception as e:
        print(f"[!] Could not compute feature importance: {e}")
        return None

    if importance.empty:
        return None

    # Top 20 features
    top = importance.head(20).copy()
    top = top.sort_values("abs_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.4)))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    colors = [COLORS["positive"] if d == "EUR_POSITIVE" else COLORS["negative"]
              for d in top["direction"]]

    bars = ax.barh(range(len(top)), top["coefficient"], color=colors, alpha=0.7,
                    edgecolor="white", linewidth=0.5)

    # Labels
    clean_labels = [f.replace("_", " ").title()[:35] for f in top["feature"]]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(clean_labels, fontsize=9)

    # Group annotations
    for i, (_, row) in enumerate(top.iterrows()):
        group = row.get("group", "")
        if group:
            ax.annotate(f"[{group}]", xy=(0, i),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=7, color="gray", va="center")

    ax.axvline(x=0, color="gray", linewidth=1)
    ax.set_xlabel("Standardised Coefficient", fontsize=11)
    ax.set_title(
        f"GLM Feature Importance ({horizon} horizon)\n"
        f"Green = EUR Positive | Red = EUR Negative",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.grid(True, alpha=0.2, axis="x")

    # Percentage annotation
    for i, (_, row) in enumerate(top.iterrows()):
        pct = row.get("pct_of_total", 0)
        coef = row["coefficient"]
        x_pos = coef + (0.002 if coef >= 0 else -0.002)
        ha = "left" if coef >= 0 else "right"
        ax.annotate(f"{pct:.1f}%", xy=(x_pos, i),
                    fontsize=8, color="gray", ha=ha, va="center")

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, f"glm_importance_{horizon}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# ROLLING CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_rolling_correlation_heatmap(df, target="eurusd_close",
                                      window=52, save=True, show=False):
    """
    Heatmap showing how correlations between EUR/USD and drivers
    change over time. Useful for identifying regime shifts.
    """
    from models.var_model import VAR_CORE_VARIABLES

    variables = [v for v in VAR_CORE_VARIABLES if v in df.columns and v != target]

    if target not in df.columns or len(variables) < 2:
        print("[!] Insufficient data for rolling correlation heatmap")
        return None

    # Compute rolling correlations
    corr_data = {}
    for var in variables:
        pair = df[[target, var]].dropna()
        if len(pair) > window:
            corr = pair[target].rolling(window, min_periods=window // 2).corr(pair[var])
            # Resample to quarterly for readability
            corr_q = corr.resample("QS").mean()
            corr_data[var.replace("_", " ").title()[:20]] = corr_q

    if not corr_data:
        return None

    corr_df = pd.DataFrame(corr_data).dropna(how="all")

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    sns.heatmap(
        corr_df.T,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        xticklabels=[d.strftime("%Y-Q%q") if hasattr(d, "strftime") else str(d)[:7]
                     for d in corr_df.index],
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Rolling {window}w Correlation with EUR/USD"},
    )

    ax.set_title(
        f"Rolling {window}-Week Correlation Regime Map\n"
        f"(How factor correlations with EUR/USD shift over time)",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "rolling_correlation_heatmap.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# SCENARIO ANALYSIS CHART
# ──────────────────────────────────────────────────────────────

def plot_scenarios(scenario_df, save=True, show=False):
    """
    Horizontal bar chart showing EUR/USD under different scenarios.
    """
    if scenario_df is None or scenario_df.empty:
        print("[!] Scenario data not available")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    scenarios = scenario_df.sort_values("change_pct")

    colors = [COLORS["positive"] if c > 0 else COLORS["negative"]
              for c in scenarios["change_pct"]]

    bars = ax.barh(range(len(scenarios)), scenarios["change_pct"],
                    color=colors, alpha=0.7, edgecolor="white", linewidth=0.5,
                    height=0.6)

    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios["scenario"].str.replace("_", " ").str.title(),
                        fontsize=11)

    # Value labels
    for i, (_, row) in enumerate(scenarios.iterrows()):
        chg = row["change_pct"]
        fc = row["forecast_eurusd"]
        x_pos = chg + (0.3 if chg >= 0 else -0.3)
        ha = "left" if chg >= 0 else "right"
        ax.annotate(
            f"{chg:+.1f}%  →  {fc:.4f}",
            xy=(x_pos, i), fontsize=10, ha=ha, va="center",
            fontweight="bold",
        )

    ax.axvline(x=0, color="gray", linewidth=1.5)

    # Current level
    if "current_eurusd" in scenarios.columns:
        current = scenarios["current_eurusd"].iloc[0]
        ax.annotate(
            f"Current: {current:.4f}",
            xy=(0, -0.7), fontsize=10, color="gray",
            ha="center", fontweight="bold",
        )

    ax.set_xlabel("EUR/USD Change (%)", fontsize=12, fontweight="bold")
    ax.set_title("Scenario Analysis: EUR/USD Outcomes Under Different Conditions",
                 fontsize=14, fontweight="bold", color=COLORS["text"], pad=15)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "scenario_analysis.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"[✓] Saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────
# RESIDUAL DIAGNOSTICS
# ──────────────────────────────────────────────────────────────

def plot_residual_diagnostics(var_results, save=True, show=False):
    """
    Four-panel residual diagnostic plot for the VAR EUR/USD equation.
    """
    var_model = var_results.get("model")
    if var_model is None or var_model.results is None:
        print("[!] VAR model not available for diagnostics")
        return None

    try:
        resid = var_model.results.resid
        if "eurusd_close" not in resid.columns:
            return None
        eur_resid = resid["eurusd_close"].dropna()
    except Exception:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS["bg"])

    # Panel 1: Residual time series
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLORS["bg"])
    ax1.plot(eur_resid.index, eur_resid.values, color=COLORS["var"],
             linewidth=0.8, alpha=0.7)
    ax1.axhline(y=0, color="gray", linewidth=1)
    ax1.fill_between(eur_resid.index, eur_resid.values, 0,
                      where=(eur_resid.values > 0),
                      color=COLORS["positive"], alpha=0.1)
    ax1.fill_between(eur_resid.index, eur_resid.values, 0,
                      where=(eur_resid.values <= 0),
                      color=COLORS["negative"], alpha=0.1)
    ax1.set_title("Residuals Over Time", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Residual", fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Histogram + normal overlay
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLORS["bg"])
    ax2.hist(eur_resid.values, bins=40, density=True, color=COLORS["var"],
             alpha=0.6, edgecolor="white")
    # Normal distribution overlay
    from scipy.stats import norm
    x_range = np.linspace(eur_resid.min(), eur_resid.max(), 100)
    ax2.plot(x_range, norm.pdf(x_range, eur_resid.mean(), eur_resid.std()),
             color=COLORS["negative"], linewidth=2, label="Normal fit")
    ax2.set_title("Residual Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Residual", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Panel 3: QQ plot
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLORS["bg"])
    from scipy.stats import probplot
    probplot(eur_resid.values, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Normal)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.2)
    ax3.get_lines()[0].set_color(COLORS["var"])
    ax3.get_lines()[1].set_color(COLORS["negative"])

    # Panel 4: ACF
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLORS["bg"])
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(eur_resid.values, ax=ax4, lags=30, alpha=0.05,
             color=COLORS["var"], vlines_kwargs={"colors": COLORS["var"]})
    ax4.set_title("Autocorrelation Function (ACF)", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.2)

    fig.suptitle("VAR Model Residual Diagnostics (EUR/USD Equation)",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02)

    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_DIR, "residual_diagnostics.png")
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
    print("Model plots module loaded. Run from main.py for full pipeline.")