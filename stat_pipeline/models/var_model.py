# eur_usd_pipeline/models/var_model.py
"""
Vector Autoregression (VAR) model for EUR/USD multi-factor analysis.

The VAR captures linear dynamic interdependencies between EUR/USD
and its macro drivers simultaneously. Every variable is treated as
endogenous — each depends on its own lags AND the lags of all others.

Key outputs:
  - Optimal lag selection (AIC / BIC)
  - Fitted VAR model with diagnostics
  - 12-month and 24-month forecasts
  - Impulse Response Functions (IRFs)
  - Forecast Error Variance Decomposition (FEVD)
  - Granger causality integration

KNOWN ISSUES & FIXES:
  - statsmodels VAR requires all series to be stationary.
    We difference as needed and invert after forecasting.
  - VAR with too many variables relative to observations
    can be unstable. We cap endogenous variables at ~8-10
    and use AIC to select lag order within MODEL_CONFIG limits.
  - statsmodels IRF plots have a known rendering issue with
    tight_layout on some matplotlib backends. We use our own
    plotting wrapper instead.
"""

import warnings
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as StatsVAR
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, FORECAST_HORIZONS
from utils.helpers import (
    make_stationary, save_dataframe, ensure_directories,
    test_stationarity,
)


# ──────────────────────────────────────────────────────────────
# VAR VARIABLE SELECTION
# ──────────────────────────────────────────────────────────────

# Core endogenous variables for the VAR system.
# These represent the key macro channels that drive EUR/USD.
# Ordered by theoretical importance.

VAR_CORE_VARIABLES = [
    "eurusd_close",           # Target: EUR/USD spot
    "yield_spread_2y",        # Yield differential (DE-US)
    "brent_close",            # Oil / Terms of Trade
    "net_spec_position",      # COT speculative positioning
    "rr_25d_proxy",           # Options market sentiment
    "proxy_sentiment",        # News / market sentiment proxy
    "vix_close",              # Global risk appetite
]

# Extended set if enough observations are available
VAR_EXTENDED_VARIABLES = [
    "carry_index",            # Carry trade attractiveness
    "macro_composite_score",  # Composite macro conditions
    "eur_rsi_14",             # Technical momentum
]

# Minimum observations per variable for inclusion
MIN_OBS_PER_VAR = 100


def select_var_variables(df, core=None, extended=None, max_vars=10):
    """
    Select variables for the VAR model based on data availability.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset.
    core : list, optional
        Core variable names (default: VAR_CORE_VARIABLES).
    extended : list, optional
        Extended variable names (default: VAR_EXTENDED_VARIABLES).
    max_vars : int
        Maximum number of endogenous variables.

    Returns
    -------
    list of str
        Selected variable names.
    """
    if core is None:
        core = VAR_CORE_VARIABLES
    if extended is None:
        extended = VAR_EXTENDED_VARIABLES

    selected = []

    # Add core variables that exist and have enough data
    for var in core:
        if var in df.columns:
            n_obs = df[var].dropna().shape[0]
            if n_obs >= MIN_OBS_PER_VAR:
                selected.append(var)
            else:
                print(f"  [·] Skipping '{var}': only {n_obs} obs (need {MIN_OBS_PER_VAR})")
        else:
            print(f"  [·] '{var}' not in dataset")

    # Add extended if room and data available
    for var in extended:
        if len(selected) >= max_vars:
            break
        if var in df.columns and var not in selected:
            n_obs = df[var].dropna().shape[0]
            if n_obs >= MIN_OBS_PER_VAR:
                selected.append(var)

    # Ensure eurusd_close is always first (target)
    if "eurusd_close" in selected:
        selected.remove("eurusd_close")
        selected.insert(0, "eurusd_close")

    print(f"[✓] VAR variables ({len(selected)}): {selected}")
    return selected


# ──────────────────────────────────────────────────────────────
# VAR MODEL FITTING
# ──────────────────────────────────────────────────────────────

class VARModel:
    """
    Wrapper around statsmodels VAR with additional diagnostics,
    forecasting, and impulse response analysis.
    """

    def __init__(self, max_lags=None, ic=None):
        """
        Parameters
        ----------
        max_lags : int
            Maximum lag order to test (default from config).
        ic : str
            Information criterion for lag selection ('aic' or 'bic').
        """
        self.max_lags = max_lags or MODEL_CONFIG.get("var_max_lags", 12)
        self.ic = ic or MODEL_CONFIG.get("var_ic", "aic")
        self.model = None
        self.results = None
        self.variables = None
        self.diff_record = None
        self.last_levels = None  # For inverting differences
        self.data_stationary = None
        self.data_original = None
        self.optimal_lags = None

    def fit(self, df, variables=None):
        """
        Fit the VAR model.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature dataset (will be subsetted and made stationary).
        variables : list of str, optional
            Variables to include. Auto-selected if None.

        Returns
        -------
        self
        """
        print("\n══════════════════════════════════════════════")
        print("   FITTING VAR MODEL")
        print("══════════════════════════════════════════════\n")

        # ── Select variables ──
        if variables is None:
            variables = select_var_variables(df)
        self.variables = variables

        # ── Subset and clean ──
        var_data = df[variables].copy()
        var_data = var_data.ffill(limit=4).dropna()
        self.data_original = var_data.copy()

        # Store last level values for forecast inversion
        self.last_levels = var_data.iloc[-1].copy()

        print(f"[i] Data shape: {var_data.shape}")
        print(f"[i] Date range: {var_data.index.min().date()} → {var_data.index.max().date()}")

        # ── Make stationary ──
        var_stationary, self.diff_record = make_stationary(var_data, verbose=True)
        self.data_stationary = var_stationary

        if len(var_stationary) < 50:
            raise RuntimeError(
                f"Insufficient observations after differencing ({len(var_stationary)}). "
                f"Need at least 50."
            )

        # ── Fit VAR ──
        print(f"\n[...] Fitting VAR with max_lags={self.max_lags}, ic='{self.ic}'")

        self.model = StatsVAR(var_stationary)

        # Lag selection
        try:
            lag_order = self.model.select_order(maxlags=self.max_lags)
            self.optimal_lags = getattr(lag_order, self.ic, None)

            # Fallback if AIC selection returns None or 0
            if self.optimal_lags is None or self.optimal_lags == 0:
                self.optimal_lags = min(4, self.max_lags)

            print(f"\n[✓] Lag selection results:")
            print(f"    AIC: {lag_order.aic}")
            print(f"    BIC: {lag_order.bic}")
            print(f"    Selected ({self.ic.upper()}): {self.optimal_lags} lags")

        except Exception as e:
            print(f"  [!] Lag selection failed ({e}), using default 4 lags")
            self.optimal_lags = 4

        # Fit with selected lag order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = self.model.fit(self.optimal_lags)

        # ── Diagnostics ──
        self._print_diagnostics()

        return self

    def _print_diagnostics(self):
        """Print VAR model diagnostics."""
        if self.results is None:
            return

        print(f"\n── VAR Model Diagnostics ──")
        print(f"  Lag order:       {self.optimal_lags}")
        print(f"  Observations:    {self.results.nobs}")
        print(f"  Variables:       {len(self.variables)}")
        print(f"  Parameters/eq:   {self.results.k_ar * len(self.variables) + 1}")
        print(f"  AIC:             {self.results.aic:.2f}")
        print(f"  BIC:             {self.results.bic:.2f}")

        # Check stability (all eigenvalues inside unit circle)
        try:
            roots = self.results.roots
            max_root = np.max(np.abs(roots))
            is_stable = max_root < 1.0
            print(f"  Max eigenvalue:  {max_root:.4f} ({'STABLE ✓' if is_stable else 'UNSTABLE ✗'})")
            if not is_stable:
                print("  [⚠] Model is not stable — forecasts may explode!")
        except Exception:
            print("  [!] Could not compute stability roots")

        # R-squared for EUR/USD equation
        if "eurusd_close" in self.variables:
            idx = self.variables.index("eurusd_close")
            # Access the specific equation's summary
            try:
                eq_summary = self.results.summary().tables[idx + 1]
                print(f"\n  EUR/USD equation included in VAR system")
            except (IndexError, Exception):
                pass

    def forecast(self, steps=52, horizon_label="12m"):
        """
        Produce point forecasts from the VAR.

        Parameters
        ----------
        steps : int
            Number of periods (weeks) to forecast.
        horizon_label : str
            Label for this forecast horizon.

        Returns
        -------
        pd.DataFrame
            Forecasted values in levels (differences inverted).
        """
        if self.results is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        print(f"\n[...] Generating {horizon_label} forecast ({steps} weeks ahead)")

        # Get the last `optimal_lags` observations for forecasting
        last_obs = self.data_stationary.values[-self.optimal_lags:]

        # Forecast in differenced space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.results.forecast(last_obs, steps=steps)

        fc_df = pd.DataFrame(
            fc,
            columns=self.variables,
            index=pd.date_range(
                start=self.data_stationary.index[-1] + pd.Timedelta(weeks=1),
                periods=steps,
                freq="W-FRI"
            ),
        )

        # ── Invert differencing to get back to levels ──
        fc_levels = self._invert_differences(fc_df)

        print(f"[✓] {horizon_label} forecast generated:")
        if "eurusd_close" in fc_levels.columns:
            start_val = self.last_levels.get("eurusd_close", np.nan)
            end_val = fc_levels["eurusd_close"].iloc[-1]
            change_pct = (end_val - start_val) / start_val * 100
            direction = "APPRECIATION" if change_pct > 0 else "DEPRECIATION"
            print(f"    EUR/USD: {start_val:.4f} → {end_val:.4f} ({change_pct:+.2f}% {direction})")

        return fc_levels

    def _invert_differences(self, fc_diff):
        """
        Convert differenced forecasts back to levels.

        Uses cumulative summation starting from the last observed level.
        """
        fc_levels = fc_diff.copy()

        for col in fc_levels.columns:
            d = self.diff_record.get(col, 0)
            if d > 0:
                # Cumsum of differences + last observed level
                last_level = self.last_levels.get(col, 0)
                fc_levels[col] = fc_diff[col].cumsum() + last_level

        return fc_levels

    def forecast_confidence_interval(self, steps=52, alpha=0.05):
        """
        Generate forecast with confidence intervals.

        Parameters
        ----------
        steps : int
        alpha : float
            Significance level (0.05 = 95% CI).

        Returns
        -------
        dict with keys: 'point', 'lower', 'upper'
            Each is a pd.DataFrame.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted.")

        last_obs = self.data_stationary.values[-self.optimal_lags:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_result = self.results.forecast_interval(last_obs, steps=steps, alpha=alpha)

        fc_point, fc_lower, fc_upper = fc_result

        date_idx = pd.date_range(
            start=self.data_stationary.index[-1] + pd.Timedelta(weeks=1),
            periods=steps,
            freq="W-FRI"
        )

        point_df = pd.DataFrame(fc_point, columns=self.variables, index=date_idx)
        lower_df = pd.DataFrame(fc_lower, columns=self.variables, index=date_idx)
        upper_df = pd.DataFrame(fc_upper, columns=self.variables, index=date_idx)

        # Invert differences for all three
        point_levels = self._invert_differences(point_df)
        lower_levels = self._invert_differences(lower_df)
        upper_levels = self._invert_differences(upper_df)

        return {
            "point": point_levels,
            "lower": lower_levels,
            "upper": upper_levels,
        }

    def impulse_response(self, periods=26, orth=True):
        """
        Compute Impulse Response Functions (IRFs).

        Shows how a one-standard-deviation shock to each variable
        propagates through the system over time.

        Parameters
        ----------
        periods : int
            Horizon for IRF computation.
        orth : bool
            Use orthogonalised IRFs (Cholesky decomposition).
            Variable ordering matters — we place EUR/USD last
            so it responds to all shocks but doesn't contemporaneously
            affect others (most conservative for our target).

        Returns
        -------
        irf_result : statsmodels IRF object
        irf_data : dict of pd.DataFrames
            IRF data for each impulse→response pair.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted.")

        print(f"\n[...] Computing Impulse Response Functions ({periods} periods)")

        irf_result = self.results.irf(periods=periods)

        # Extract IRF data into a more usable format
        irf_data = {}
        for i, impulse_var in enumerate(self.variables):
            for j, response_var in enumerate(self.variables):
                key = f"{impulse_var} → {response_var}"
                irf_data[key] = pd.Series(
                    irf_result.irfs[:, j, i],
                    index=range(periods + 1),
                    name=key,
                )

        print(f"[✓] IRFs computed: {len(irf_data)} impulse-response pairs")

        return irf_result, irf_data

    def variance_decomposition(self, periods=52):
        """
        Forecast Error Variance Decomposition (FEVD).

        Shows what percentage of the forecast error variance for
        EUR/USD is explained by shocks to each variable.

        Parameters
        ----------
        periods : int
            Horizon for FEVD computation.

        Returns
        -------
        fevd_result : statsmodels FEVD object
        fevd_table : pd.DataFrame
            FEVD for EUR/USD at selected horizons.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted.")

        print(f"\n[...] Computing Variance Decomposition ({periods} periods)")

        fevd_result = self.results.fevd(periods=periods)

        # Extract EUR/USD FEVD
        if "eurusd_close" in self.variables:
            eur_idx = self.variables.index("eurusd_close")
            fevd_data = fevd_result.decomp[:, eur_idx, :]

            fevd_table = pd.DataFrame(
                fevd_data,
                columns=self.variables,
                index=range(1, periods + 1),
            )
            fevd_table.index.name = "horizon_weeks"

            # Print key horizons
            for h in [4, 13, 26, 52]:
                if h <= periods:
                    row = fevd_table.loc[h]
                    print(f"\n    FEVD at {h}-week horizon:")
                    for var, pct in row.items():
                        bar = "█" * int(pct * 40)
                        print(f"      {var:30s} {pct*100:5.1f}% {bar}")
        else:
            fevd_table = pd.DataFrame()

        print(f"\n[✓] FEVD computed for {periods} periods")

        return fevd_result, fevd_table

    def save(self, filepath=None):
        """Save model results and metadata."""
        if filepath is None:
            from config import MODEL_DIR
            filepath = os.path.join(MODEL_DIR, "var_model.pkl")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "variables": self.variables,
            "optimal_lags": self.optimal_lags,
            "diff_record": self.diff_record,
            "last_levels": self.last_levels,
            "ic": self.ic,
            "aic": self.results.aic if self.results else None,
            "bic": self.results.bic if self.results else None,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"[✓] VAR model metadata saved to {filepath}")

    def summary(self):
        """Return the full statsmodels summary."""
        if self.results is None:
            raise RuntimeError("Model not fitted.")
        return self.results.summary()


# ──────────────────────────────────────────────────────────────
# GRANGER CAUSALITY ANALYSIS
# ──────────────────────────────────────────────────────────────

def run_granger_causality(df, variables=None, target="eurusd_close",
                          max_lag=12, alpha=0.05):
    """
    Run pairwise Granger causality tests.

    Tests whether each variable Granger-causes EUR/USD
    (i.e., whether lags of X improve prediction of EUR/USD
    beyond EUR/USD's own lags).

    Parameters
    ----------
    df : pd.DataFrame
    variables : list of str
        Variables to test. If None, uses VAR_CORE_VARIABLES.
    target : str
        Target variable.
    max_lag : int
        Maximum lag order to test.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        Granger causality results with p-values and significance.
    """
    print(f"\n[...] Running Granger Causality Tests (target: {target})")

    if variables is None:
        variables = [v for v in VAR_CORE_VARIABLES if v in df.columns and v != target]

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame")

    results = []

    for var in variables:
        if var == target or var not in df.columns:
            continue

        # Prepare pairwise data (must be stationary)
        pair_data = df[[target, var]].dropna()

        if len(pair_data) < max_lag * 3:
            print(f"  [·] Skipping '{var}': insufficient data ({len(pair_data)} rows)")
            continue

        # Test stationarity and difference if needed
        pair_stationary = pair_data.copy()
        for col in pair_stationary.columns:
            stat_result = test_stationarity(pair_stationary[col], name=col, verbose=False)
            if not stat_result["is_stationary"]:
                pair_stationary[col] = pair_stationary[col].diff()
        pair_stationary = pair_stationary.dropna()

        if len(pair_stationary) < max_lag * 3:
            continue

        try:
            # statsmodels granger test
            # Column order: [target, cause] — tests whether cause Granger-causes target
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_result = grangercausalitytests(
                    pair_stationary[[target, var]],
                    maxlag=max_lag,
                    verbose=False,
                )

            # Extract minimum p-value across all lags (most significant)
            min_p = 1.0
            best_lag = 1
            all_p = {}

            for lag in range(1, max_lag + 1):
                if lag in gc_result:
                    # Use F-test p-value
                    p_val = gc_result[lag][0]["ssr_ftest"][1]
                    all_p[lag] = p_val
                    if p_val < min_p:
                        min_p = p_val
                        best_lag = lag

            is_significant = min_p < alpha

            results.append({
                "variable": var,
                "granger_causes": target,
                "min_p_value": min_p,
                "best_lag": best_lag,
                "significant": is_significant,
                "significance": "***" if min_p < 0.001 else "**" if min_p < 0.01 else "*" if min_p < 0.05 else "",
            })

            status = "✓ SIGNIFICANT" if is_significant else "· not significant"
            print(f"  [{status}] {var:30s} → {target}: p={min_p:.4f} (lag={best_lag})")

        except Exception as e:
            print(f"  [!] Granger test failed for '{var}': {e}")

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("min_p_value")

        print(f"\n[✓] Granger causality: {results_df['significant'].sum()} of "
              f"{len(results_df)} variables significantly cause {target}")

    return results_df


def run_full_granger_matrix(df, variables=None, max_lag=8, alpha=0.05):
    """
    Run Granger causality between ALL pairs of variables
    to build a full causality matrix.

    Returns
    -------
    pd.DataFrame
        Matrix where cell (i, j) is the p-value for
        'variable j Granger-causes variable i'.
    """
    print(f"\n[...] Building full Granger causality matrix")

    if variables is None:
        variables = [v for v in VAR_CORE_VARIABLES if v in df.columns]

    n = len(variables)
    matrix = pd.DataFrame(
        np.ones((n, n)),
        index=variables,
        columns=variables,
    )

    for target in variables:
        for cause in variables:
            if target == cause:
                matrix.loc[target, cause] = np.nan
                continue

            pair = df[[target, cause]].dropna()
            if len(pair) < max_lag * 3:
                continue

            # Quick stationarity fix
            pair_s = pair.copy()
            for col in pair_s.columns:
                sr = test_stationarity(pair_s[col], name=col, verbose=False)
                if not sr["is_stationary"]:
                    pair_s[col] = pair_s[col].diff()
            pair_s = pair_s.dropna()

            if len(pair_s) < max_lag * 3:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc = grangercausalitytests(pair_s[[target, cause]], maxlag=max_lag, verbose=False)

                min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1) if lag in gc)
                matrix.loc[target, cause] = min_p

            except Exception:
                pass

    print(f"[✓] Granger matrix: {n}×{n} ({(matrix < alpha).sum().sum()} significant pairs)")

    return matrix


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: RUN FULL VAR PIPELINE
# ──────────────────────────────────────────────────────────────

def run_var_pipeline(df, save=True):
    """
    Execute the complete VAR analysis pipeline.

    Steps:
      1. Select variables
      2. Fit VAR model
      3. Run Granger causality
      4. Generate 12m and 24m forecasts
      5. Compute IRFs and FEVD
      6. Save results

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset.

    Returns
    -------
    dict with all VAR outputs.
    """
    ensure_directories()

    print("\n" + "=" * 60)
    print("   VAR MODEL PIPELINE")
    print("=" * 60 + "\n")

    # ── Step 1–2: Fit model ──
    var = VARModel()
    var.fit(df)

    # ── Step 3: Granger causality ──
    gc_results = run_granger_causality(df, variables=var.variables)
    gc_matrix = run_full_granger_matrix(df, variables=var.variables)

    # ── Step 4: Forecasts ──
    forecasts = {}
    forecast_cis = {}

    for horizon in FORECAST_HORIZONS:
        steps = horizon * 4  # Convert months to approximate weeks
        label = f"{horizon}m"

        fc = var.forecast(steps=steps, horizon_label=label)
        forecasts[label] = fc

        ci = var.forecast_confidence_interval(steps=steps)
        forecast_cis[label] = ci

    # ── Step 5: IRFs and FEVD ──
    irf_result, irf_data = var.impulse_response(periods=52)
    fevd_result, fevd_table = var.variance_decomposition(periods=52)

    # ── Step 6: Save ──
    if save:
        var.save()

        for label, fc in forecasts.items():
            save_dataframe(fc, f"var_forecast_{label}.csv", subdir="models")

        if not gc_results.empty:
            save_dataframe(gc_results, "granger_causality.csv", subdir="models")

        save_dataframe(gc_matrix, "granger_matrix.csv", subdir="models")

        if not fevd_table.empty:
            save_dataframe(fevd_table, "fevd_table.csv", subdir="models")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("   VAR PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Variables:      {len(var.variables)}")
    print(f"   Optimal lags:   {var.optimal_lags}")
    print(f"   Observations:   {var.results.nobs}")

    for label, fc in forecasts.items():
        if "eurusd_close" in fc.columns:
            current = var.last_levels.get("eurusd_close", np.nan)
            final = fc["eurusd_close"].iloc[-1]
            chg = (final - current) / current * 100
            print(f"   {label} EUR/USD:   {current:.4f} → {final:.4f} ({chg:+.2f}%)")

    print("=" * 60 + "\n")

    return {
        "model": var,
        "forecasts": forecasts,
        "forecast_cis": forecast_cis,
        "granger_results": gc_results,
        "granger_matrix": gc_matrix,
        "irf_result": irf_result,
        "irf_data": irf_data,
        "fevd_result": fevd_result,
        "fevd_table": fevd_table,
    }


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from features.engineer import run_feature_engineering

    print("Running feature engineering first...")
    fe_result = run_feature_engineering()

    print("\nRunning VAR pipeline...")
    var_result = run_var_pipeline(fe_result["full_df"])