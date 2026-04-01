# eur_usd_pipeline/models/glm_model.py
"""
Generalised Linear Model (GLM) for EUR/USD prediction.

While the VAR captures linear dynamic interdependencies across
the full system, the GLM focuses on the EUR/USD equation alone
and can incorporate non-linear relationships through:

  - Interaction terms (yield × positioning, oil × carry, etc.)
  - Polynomial features (squared, cubed drivers)
  - Flexible link functions (log, identity, inverse)
  - Multiple distribution families (Gaussian, Gamma, Inverse Gaussian)

This lets us compare:
  VAR  → linear multi-equation system dynamics
  GLM  → non-linear single-equation predictive power

We fit multiple GLM specifications and select the best via
out-of-sample performance metrics.

KNOWN ISSUES & FIXES:
  - statsmodels GLM with Gamma family requires strictly positive y.
    For EUR/USD returns (which can be negative), we use Gaussian family
    with a log link as the non-linear variant, or shift returns to
    be strictly positive for Gamma.
  - Perfect separation / quasi-separation can cause GLM convergence
    failure with binary targets. We use regularisation (penalised
    IRLS) as a fallback.
  - Multicollinearity among interaction terms can inflate standard
    errors. We apply VIF screening before fitting.
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM as StatsGLM
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy import stats as sp_stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, FORECAST_HORIZONS
from utils.helpers import (
    ensure_directories, save_dataframe, scale_features,
)


# ──────────────────────────────────────────────────────────────
# GLM FEATURE SELECTION
# ──────────────────────────────────────────────────────────────

# Features grouped by category for interpretability.
# The GLM uses lagged features to predict forward returns.

GLM_FEATURE_GROUPS = {
    "yield_spread": [
        "yld_spread_level", "yld_spread_chg_4w", "yld_spread_zscore",
        "yld_spread_momentum", "yld_spread_narrowing",
    ],
    "terms_of_trade": [
        "tot_divergence", "tot_inv_zscore", "tot_relative_momentum",
        "oil_shock_cumulative",
    ],
    "positioning": [
        "pos_zscore", "pos_contrarian_signal", "pos_momentum_4w",
        "squeeze_score",
    ],
    "risk_reversal": [
        "rr_level", "rr_zscore", "rr_momentum_4w",
    ],
    "sentiment": [
        "proxy_sentiment", "risk_appetite", "momentum_sentiment",
    ],
    "technical": [
        "eur_rsi_14", "eur_macd_histogram", "eur_bb_pct_b",
        "eur_trend_strength", "eur_hurst",
    ],
    "volatility": [
        "eur_rvol_13w", "vol_term_structure", "vix_zscore",
    ],
    "carry": [
        "carry_index", "carry_zscore",
    ],
    "composite": [
        "macro_composite_score",
    ],
    "interactions": [
        "interact_yield_x_pos", "interact_oil_shock_x_carry",
        "interact_rr_x_momentum", "interact_sentiment_x_pos",
        "interact_yield_chg_x_highvol", "interact_contrarian_x_highvol",
    ],
    "nonlinear": [
        "yld_spread_level_sq", "carry_index_sq", "pos_zscore_sq",
        "proxy_sentiment_sq", "rr_level_sq",
        "yld_spread_level_cb", "carry_index_cb", "pos_zscore_cb",
    ],
}

# Target variables (forward returns at different horizons)
GLM_TARGETS = {
    "4w": "target_return_4w",
    "13w": "target_return_13w",
    "26w": "target_return_26w",
    "52w": "target_return_52w",
}


def select_glm_features(df, groups=None, min_coverage=0.7):
    """
    Select GLM features based on data availability and VIF screening.

    Parameters
    ----------
    df : pd.DataFrame
    groups : dict, optional
        Feature groups (default: GLM_FEATURE_GROUPS).
    min_coverage : float
        Minimum non-NaN fraction.

    Returns
    -------
    list of str
        Selected feature names.
    dict
        Feature group membership for interpretation.
    """
    if groups is None:
        groups = GLM_FEATURE_GROUPS

    selected = []
    group_map = {}

    for group_name, features in groups.items():
        for feat in features:
            if feat in df.columns:
                coverage = df[feat].notna().mean()
                if coverage >= min_coverage:
                    selected.append(feat)
                    group_map[feat] = group_name

    print(f"[✓] GLM features selected: {len(selected)} from {len(groups)} groups")

    # ── VIF screening (remove features with VIF > 10) ──
    selected = _vif_screen(df, selected, threshold=10.0)

    return selected, group_map


def _vif_screen(df, features, threshold=10.0):
    """
    Screen features using Variance Inflation Factor.
    Iteratively removes the feature with highest VIF until
    all are below threshold.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
    threshold : float
        Maximum acceptable VIF.

    Returns
    -------
    list of str
        Screened features.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    remaining = list(features)
    data = df[remaining].dropna()

    if len(data) < len(remaining) + 10:
        print("  [!] Insufficient data for VIF screening, skipping")
        return remaining

    max_iterations = 20
    iteration = 0

    while iteration < max_iterations and len(remaining) > 2:
        try:
            X = data[remaining].values.astype(float)

            # Check for constant columns
            std = X.std(axis=0)
            constant_mask = std < 1e-10
            if constant_mask.any():
                to_remove = [remaining[i] for i in range(len(remaining)) if constant_mask[i]]
                remaining = [f for f in remaining if f not in to_remove]
                data = df[remaining].dropna()
                continue

            vifs = []
            for i in range(len(remaining)):
                try:
                    vif = variance_inflation_factor(X, i)
                    vifs.append(vif)
                except Exception:
                    vifs.append(0)

            max_vif = max(vifs)
            if max_vif <= threshold or np.isinf(max_vif):
                break

            # Remove feature with highest VIF
            worst_idx = vifs.index(max_vif)
            removed = remaining.pop(worst_idx)
            data = df[remaining].dropna()
            iteration += 1

        except Exception as e:
            print(f"  [!] VIF screening error: {e}")
            break

    if iteration > 0:
        print(f"  [·] VIF screening removed {iteration} features (threshold={threshold})")

    return remaining


# ──────────────────────────────────────────────────────────────
# GLM MODEL CLASS
# ──────────────────────────────────────────────────────────────

class GLMModel:
    """
    GLM wrapper for EUR/USD prediction with multiple specifications.
    """

    def __init__(self, family="gaussian", link=None):
        """
        Parameters
        ----------
        family : str
            GLM family: 'gaussian', 'gamma', 'inverse_gaussian'.
        link : str or None
            Link function: 'identity', 'log', 'inverse'.
            If None, uses the canonical link for the family.
        """
        self.family_name = family
        self.link_name = link
        self.family = self._get_family(family, link)
        self.model = None
        self.results = None
        self.features = None
        self.target = None
        self.group_map = None
        self.train_metrics = None
        self.scaler = None

    def _get_family(self, family, link):
        """Construct statsmodels family object."""
        link_map = {
            "identity": sm.families.links.Identity(),
            "log": sm.families.links.Log(),
            "inverse": sm.families.links.InversePower(),
        }

        link_obj = link_map.get(link) if link else None

        if family == "gaussian":
            return sm.families.Gaussian(link=link_obj) if link_obj else sm.families.Gaussian()
        elif family == "gamma":
            return sm.families.Gamma(link=link_obj) if link_obj else sm.families.Gamma()
        elif family == "inverse_gaussian":
            return sm.families.InverseGaussian(link=link_obj) if link_obj else sm.families.InverseGaussian()
        else:
            print(f"  [!] Unknown family '{family}', defaulting to Gaussian")
            return sm.families.Gaussian()

    def fit(self, df, target_col="target_return_13w", features=None,
            feature_groups=None, scale=True):
        """
        Fit the GLM model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and target.
        target_col : str
            Target variable name.
        features : list of str, optional
            Feature names. Auto-selected if None.
        feature_groups : dict, optional
            Feature group definitions.
        scale : bool
            Whether to z-score standardise features.

        Returns
        -------
        self
        """
        print(f"\n══════════════════════════════════════════════")
        print(f"   FITTING GLM ({self.family_name.upper()}, link={self.link_name or 'canonical'})")
        print(f"   Target: {target_col}")
        print(f"══════════════════════════════════════════════\n")

        self.target = target_col

        # ── Feature selection ──
        if features is None:
            features, self.group_map = select_glm_features(df, groups=feature_groups)
        else:
            self.group_map = {f: "manual" for f in features}

        self.features = features

        # ── Prepare data ──
        required_cols = features + [target_col]
        available = [c for c in required_cols if c in df.columns]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            print(f"  [!] Missing columns: {missing}")
            features = [f for f in features if f in df.columns]
            self.features = features

        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not in DataFrame")

        data = df[features + [target_col]].dropna()
        print(f"[i] Training data: {len(data)} rows × {len(features)} features")

        if len(data) < len(features) + 20:
            raise RuntimeError(
                f"Insufficient training data ({len(data)} rows) for "
                f"{len(features)} features"
            )

        y = data[target_col].values.astype(float)
        X = data[features].values.astype(float)

        # ── Handle Gamma family (requires y > 0) ──
        if self.family_name == "gamma":
            if (y <= 0).any():
                shift = abs(y.min()) + 0.001
                y = y + shift
                print(f"  [·] Shifted target by {shift:.4f} for Gamma family (requires y > 0)")

        # ── Scale features ──
        if scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # ── Add constant ──
        X_const = sm.add_constant(X)

        # ── Fit GLM ──
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = StatsGLM(y, X_const, family=self.family)
                self.results = self.model.fit(maxiter=100)

        except Exception as e:
            print(f"  [!] GLM fit failed ({e}), trying regularised fit")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = StatsGLM(y, X_const, family=self.family)
                    self.results = self.model.fit_regularized(
                        alpha=0.1, L1_wt=0.5, maxiter=200
                    )
            except Exception as e2:
                raise RuntimeError(f"GLM fitting failed: {e2}")

        # ── Compute training metrics ──
        y_pred = self.results.predict(X_const)
        self.train_metrics = self._compute_metrics(y, y_pred, "Training")

        # ── Print results ──
        self._print_results(data.index)

        return self

    def _compute_metrics(self, y_true, y_pred, label=""):
        """Compute regression performance metrics."""
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "directional_accuracy": np.mean(np.sign(y_true) == np.sign(y_pred)),
            "correlation": np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else 0,
        }

        if label:
            print(f"\n  {label} Metrics:")
            print(f"    RMSE:                  {metrics['rmse']:.6f}")
            print(f"    MAE:                   {metrics['mae']:.6f}")
            print(f"    R²:                    {metrics['r2']:.4f}")
            print(f"    Directional accuracy:  {metrics['directional_accuracy']:.1%}")
            print(f"    Correlation:           {metrics['correlation']:.4f}")

        return metrics

    def _print_results(self, index):
        """Print GLM fitting results and coefficient analysis."""
        if self.results is None:
            return

        print(f"\n── GLM Coefficient Analysis ──")
        print(f"  Family: {self.family_name}, Link: {self.link_name or 'canonical'}")

        # Extract coefficients
        params = self.results.params
        pvalues = None

        # Check if pvalues are available (not for regularised fits)
        try:
            pvalues = self.results.pvalues
        except (AttributeError, Exception):
            pass

        # Build coefficient table
        coef_names = ["const"] + list(self.features)
        coef_data = []

        for i, name in enumerate(coef_names):
            if i < len(params):
                entry = {
                    "feature": name,
                    "coefficient": params[i],
                    "abs_coef": abs(params[i]),
                }
                if pvalues is not None and i < len(pvalues):
                    entry["p_value"] = pvalues[i]
                    entry["significant"] = pvalues[i] < 0.05
                else:
                    entry["p_value"] = np.nan
                    entry["significant"] = np.nan

                coef_data.append(entry)

        coef_df = pd.DataFrame(coef_data)

        if not coef_df.empty:
            # Sort by absolute coefficient magnitude
            coef_df = coef_df.sort_values("abs_coef", ascending=False)

            print(f"\n  Top 15 features by coefficient magnitude:")
            for _, row in coef_df.head(15).iterrows():
                name = row["feature"]
                coef = row["coefficient"]
                sig = ""
                if pd.notna(row.get("p_value")):
                    p = row["p_value"]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                # Feature group
                group = self.group_map.get(name, "")
                if group:
                    group = f" [{group}]"

                direction = "↑ EUR" if coef > 0 else "↓ EUR"
                print(f"    {name:40s} {coef:+.6f} {sig:3s} {direction}{group}")

        # Model fit statistics
        try:
            print(f"\n  Model Fit:")
            print(f"    AIC:       {self.results.aic:.2f}")
            print(f"    BIC:       {self.results.bic:.2f}")
            print(f"    Deviance:  {self.results.deviance:.4f}")
            print(f"    Pearson χ²: {self.results.pearson_chi2:.4f}")
        except (AttributeError, Exception):
            pass

    def predict(self, df):
        """
        Generate predictions from fitted model.

        Parameters
        ----------
        df : pd.DataFrame
            Data with required feature columns.

        Returns
        -------
        pd.Series
            Predictions indexed by df's index.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted.")

        available_features = [f for f in self.features if f in df.columns]
        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            print(f"  [!] Missing features for prediction: {missing}")

        data = df[available_features].copy()

        # Handle missing features by filling with 0 (neutral)
        for f in self.features:
            if f not in data.columns:
                data[f] = 0

        data = data[self.features]  # Ensure correct order

        # Handle NaNs
        nan_mask = data.isna().any(axis=1)
        data_clean = data.fillna(0)

        X = data_clean.values.astype(float)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        X_const = sm.add_constant(X, has_constant="add")

        preds = self.results.predict(X_const)

        result = pd.Series(preds, index=df.index, name=f"glm_pred_{self.target}")
        result[nan_mask] = np.nan

        return result

    def evaluate(self, test_df):
        """
        Evaluate model on test data.

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        dict
            Test metrics.
        pd.DataFrame
            Actual vs predicted.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted.")

        if self.target not in test_df.columns:
            raise ValueError(f"Target '{self.target}' not in test DataFrame")

        data = test_df[self.features + [self.target]].dropna()

        if len(data) < 5:
            print("[!] Insufficient test data for evaluation")
            return {}, pd.DataFrame()

        y_true = data[self.target].values
        predictions = self.predict(data)
        y_pred = predictions.values

        # Handle Gamma shift
        test_metrics = self._compute_metrics(y_true, y_pred, "Test")

        # Actual vs predicted DataFrame
        comparison = pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred,
            "residual": y_true - y_pred,
            "direction_correct": np.sign(y_true) == np.sign(y_pred),
        }, index=data.index)

        return test_metrics, comparison

    def feature_importance(self):
        """
        Compute feature importance based on standardised coefficients.

        Returns
        -------
        pd.DataFrame
            Feature importance table.
        """
        if self.results is None:
            raise RuntimeError("Model not fitted.")

        params = self.results.params[1:]  # Exclude intercept

        importance = pd.DataFrame({
            "feature": self.features,
            "coefficient": params,
            "abs_importance": np.abs(params),
            "direction": ["EUR_POSITIVE" if p > 0 else "EUR_NEGATIVE" for p in params],
            "group": [self.group_map.get(f, "unknown") for f in self.features],
        })

        importance = importance.sort_values("abs_importance", ascending=False)
        importance["rank"] = range(1, len(importance) + 1)
        importance["pct_of_total"] = (
            importance["abs_importance"] / importance["abs_importance"].sum() * 100
        )

        return importance

    def save(self, filepath=None):
        """Save model metadata."""
        if filepath is None:
            from config import MODEL_DIR
            filepath = os.path.join(MODEL_DIR, "glm_model.pkl")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "family": self.family_name,
            "link": self.link_name,
            "features": self.features,
            "target": self.target,
            "group_map": self.group_map,
            "train_metrics": self.train_metrics,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"[✓] GLM model metadata saved to {filepath}")


# ──────────────────────────────────────────────────────────────
# MULTI-SPECIFICATION COMPARISON
# ──────────────────────────────────────────────────────────────

def compare_glm_specifications(train_df, test_df, target_col="target_return_13w"):
    """
    Fit multiple GLM specifications and compare performance.

    Specifications tested:
      1. Gaussian + Identity link (standard linear)
      2. Gaussian + Log link (non-linear via log transform)
      3. Reduced model (only significant features from spec 1)
      4. Interaction-heavy model

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
    target_col : str

    Returns
    -------
    pd.DataFrame
        Comparison table of all specifications.
    dict
        Fitted models keyed by specification name.
    """
    print("\n" + "=" * 60)
    print("   GLM SPECIFICATION COMPARISON")
    print("=" * 60 + "\n")

    specifications = {
        "gaussian_identity": {"family": "gaussian", "link": "identity"},
        "gaussian_log": {"family": "gaussian", "link": "log"},
    }

    models = {}
    comparison = []

    for spec_name, spec_params in specifications.items():
        print(f"\n{'─' * 40}")
        print(f"  Specification: {spec_name}")
        print(f"{'─' * 40}")

        try:
            glm = GLMModel(
                family=spec_params["family"],
                link=spec_params.get("link"),
            )

            # For log link, target must be positive
            if spec_params.get("link") == "log":
                if target_col in train_df.columns:
                    min_val = train_df[target_col].min()
                    if min_val <= 0:
                        print(f"  [·] Log link requires y > 0, shifting target")
                        shift = abs(min_val) + 0.001
                        train_shifted = train_df.copy()
                        test_shifted = test_df.copy()
                        train_shifted[target_col] = train_shifted[target_col] + shift
                        test_shifted[target_col] = test_shifted[target_col] + shift

                        glm.fit(train_shifted, target_col=target_col)
                        test_metrics, test_comparison = glm.evaluate(test_shifted)
                    else:
                        glm.fit(train_df, target_col=target_col)
                        test_metrics, test_comparison = glm.evaluate(test_df)
                else:
                    continue
            else:
                glm.fit(train_df, target_col=target_col)
                test_metrics, test_comparison = glm.evaluate(test_df)

            models[spec_name] = glm

            row = {
                "specification": spec_name,
                "family": spec_params["family"],
                "link": spec_params.get("link", "canonical"),
                "n_features": len(glm.features),
                "train_r2": glm.train_metrics.get("r2", np.nan),
                "train_rmse": glm.train_metrics.get("rmse", np.nan),
                "train_dir_acc": glm.train_metrics.get("directional_accuracy", np.nan),
                "test_r2": test_metrics.get("r2", np.nan),
                "test_rmse": test_metrics.get("rmse", np.nan),
                "test_dir_acc": test_metrics.get("directional_accuracy", np.nan),
            }

            try:
                row["aic"] = glm.results.aic
                row["bic"] = glm.results.bic
            except (AttributeError, Exception):
                row["aic"] = np.nan
                row["bic"] = np.nan

            comparison.append(row)

        except Exception as e:
            print(f"  [!] Specification '{spec_name}' failed: {e}")

    # ── Reduced model (significant features only from best full model) ──
    if "gaussian_identity" in models:
        try:
            print(f"\n{'─' * 40}")
            print(f"  Specification: reduced_model")
            print(f"{'─' * 40}")

            base_model = models["gaussian_identity"]
            if hasattr(base_model.results, "pvalues"):
                pvals = base_model.results.pvalues[1:]  # Skip intercept
                sig_mask = pvals < 0.10  # 10% threshold for inclusion
                sig_features = [f for f, s in zip(base_model.features, sig_mask) if s]

                if len(sig_features) >= 3:
                    glm_reduced = GLMModel(family="gaussian", link="identity")
                    glm_reduced.fit(train_df, target_col=target_col, features=sig_features)
                    test_metrics, _ = glm_reduced.evaluate(test_df)

                    models["reduced_model"] = glm_reduced

                    row = {
                        "specification": "reduced_model",
                        "family": "gaussian",
                        "link": "identity",
                        "n_features": len(sig_features),
                        "train_r2": glm_reduced.train_metrics.get("r2", np.nan),
                        "train_rmse": glm_reduced.train_metrics.get("rmse", np.nan),
                        "train_dir_acc": glm_reduced.train_metrics.get("directional_accuracy", np.nan),
                        "test_r2": test_metrics.get("r2", np.nan),
                        "test_rmse": test_metrics.get("rmse", np.nan),
                        "test_dir_acc": test_metrics.get("directional_accuracy", np.nan),
                    }

                    try:
                        row["aic"] = glm_reduced.results.aic
                        row["bic"] = glm_reduced.results.bic
                    except (AttributeError, Exception):
                        row["aic"] = np.nan
                        row["bic"] = np.nan

                    comparison.append(row)
                else:
                    print(f"  [·] Only {len(sig_features)} significant features — skipping reduced model")

        except Exception as e:
            print(f"  [!] Reduced model failed: {e}")

    # ── Interaction-only model ──
    try:
        print(f"\n{'─' * 40}")
        print(f"  Specification: interaction_model")
        print(f"{'─' * 40}")

        interaction_features = [
            f for f in train_df.columns
            if ("interact_" in f or "_sq" in f or "_cb" in f)
            and train_df[f].notna().mean() > 0.7
        ]

        if len(interaction_features) >= 3:
            glm_interact = GLMModel(family="gaussian", link="identity")
            glm_interact.fit(train_df, target_col=target_col, features=interaction_features)
            test_metrics, _ = glm_interact.evaluate(test_df)

            models["interaction_model"] = glm_interact

            row = {
                "specification": "interaction_model",
                "family": "gaussian",
                "link": "identity",
                "n_features": len(interaction_features),
                "train_r2": glm_interact.train_metrics.get("r2", np.nan),
                "train_rmse": glm_interact.train_metrics.get("rmse", np.nan),
                "train_dir_acc": glm_interact.train_metrics.get("directional_accuracy", np.nan),
                "test_r2": test_metrics.get("r2", np.nan),
                "test_rmse": test_metrics.get("rmse", np.nan),
                "test_dir_acc": test_metrics.get("directional_accuracy", np.nan),
            }

            try:
                row["aic"] = glm_interact.results.aic
                row["bic"] = glm_interact.results.bic
            except (AttributeError, Exception):
                row["aic"] = np.nan
                row["bic"] = np.nan

            comparison.append(row)

    except Exception as e:
        print(f"  [!] Interaction model failed: {e}")

    # ── Summary table ──
    comp_df = pd.DataFrame(comparison)

    if not comp_df.empty:
        print("\n" + "=" * 60)
        print("   GLM COMPARISON SUMMARY")
        print("=" * 60)
        for _, row in comp_df.iterrows():
            print(f"\n  {row['specification']:25s}")
            print(f"    Features:    {row['n_features']}")
            print(f"    Train R²:    {row['train_r2']:.4f}   Test R²:    {row.get('test_r2', np.nan):.4f}")
            print(f"    Train RMSE:  {row['train_rmse']:.6f}   Test RMSE:  {row.get('test_rmse', np.nan):.6f}")
            print(f"    Dir. Acc:    {row['train_dir_acc']:.1%}          {row.get('test_dir_acc', np.nan):.1%}")

        # Best model by test R²
        if comp_df["test_r2"].notna().any():
            best = comp_df.loc[comp_df["test_r2"].idxmax()]
            print(f"\n  ★ Best specification (test R²): {best['specification']}")

        print("=" * 60 + "\n")

    return comp_df, models


# ──────────────────────────────────────────────────────────────
# MULTI-HORIZON GLM
# ──────────────────────────────────────────────────────────────

def fit_multi_horizon_glm(train_df, test_df, save=True):
    """
    Fit GLMs across multiple forecast horizons.

    Returns
    -------
    dict
        Results keyed by horizon.
    """
    print("\n" + "=" * 60)
    print("   MULTI-HORIZON GLM ANALYSIS")
    print("=" * 60 + "\n")

    results = {}

    for horizon_label, target_col in GLM_TARGETS.items():
        if target_col not in train_df.columns:
            print(f"  [·] Target '{target_col}' not available — skipping {horizon_label}")
            continue

        n_valid = train_df[target_col].notna().sum()
        if n_valid < 50:
            print(f"  [·] Target '{target_col}' has only {n_valid} obs — skipping")
            continue

        print(f"\n{'═' * 40}")
        print(f"  HORIZON: {horizon_label} ({target_col})")
        print(f"{'═' * 40}")

        try:
            comp_df, models = compare_glm_specifications(
                train_df, test_df, target_col=target_col
            )

            # Get best model
            best_model = None
            if models:
                if not comp_df.empty and comp_df["test_r2"].notna().any():
                    best_name = comp_df.loc[comp_df["test_r2"].idxmax(), "specification"]
                    best_model = models.get(best_name)
                else:
                    best_model = list(models.values())[0]

            results[horizon_label] = {
                "comparison": comp_df,
                "models": models,
                "best_model": best_model,
                "target": target_col,
            }

            if save and best_model is not None:
                best_model.save(
                    os.path.join("output", "models", f"glm_best_{horizon_label}.pkl")
                )

                importance = best_model.feature_importance()
                if not importance.empty:
                    save_dataframe(
                        importance,
                        f"glm_importance_{horizon_label}.csv",
                        subdir="models"
                    )

        except Exception as e:
            print(f"  [!] Horizon '{horizon_label}' failed: {e}")

    return results


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: RUN FULL GLM PIPELINE
# ──────────────────────────────────────────────────────────────

def run_glm_pipeline(train_df, test_df, save=True):
    """
    Execute the complete GLM analysis pipeline.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        From feature engineering pipeline.

    Returns
    -------
    dict
        All GLM results.
    """
    ensure_directories()

    results = fit_multi_horizon_glm(train_df, test_df, save=save)

    # ── Save comparison tables ──
    if save:
        all_comparisons = []
        for horizon, res in results.items():
            comp = res.get("comparison", pd.DataFrame())
            if not comp.empty:
                comp["horizon"] = horizon
                all_comparisons.append(comp)

        if all_comparisons:
            full_comp = pd.concat(all_comparisons, ignore_index=True)
            save_dataframe(full_comp, "glm_all_comparisons.csv", subdir="models")

    print("\n" + "=" * 60)
    print("   GLM PIPELINE COMPLETE")
    print("=" * 60)

    for horizon, res in results.items():
        best = res.get("best_model")
        if best and best.train_metrics:
            print(f"   {horizon:6s}: R²={best.train_metrics.get('r2', 0):.4f}, "
                  f"Dir.Acc={best.train_metrics.get('directional_accuracy', 0):.1%}")

    print("=" * 60 + "\n")

    return results


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from features.engineer import run_feature_engineering

    print("Running feature engineering first...")
    fe_result = run_feature_engineering()

    print("\nRunning GLM pipeline...")
    glm_results = run_glm_pipeline(fe_result["train_df"], fe_result["test_df"])