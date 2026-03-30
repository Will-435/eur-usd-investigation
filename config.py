# eur_usd_pipeline/config.py
"""
Global configuration for the EUR/USD macro pipeline.
All API keys, date ranges, and shared parameters live here.
"""

# ──────────────────────────────────────────────────────────────
# API KEYS — replace placeholder strings with your real keys
# ──────────────────────────────────────────────────────────────

# Free from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"  # <-- paste your FRED key

# GNews — free tier (100 requests/day, no key required for basic use)
# If you register at https://gnews.io/ you get higher limits
GNEWS_API_KEY = "YOUR_GNEWS_API_KEY_HERE"  # <-- paste if you have one, otherwise leave as-is

# ──────────────────────────────────────────────────────────────
# DATE RANGE
# ──────────────────────────────────────────────────────────────

START_DATE = "2017-01-01"   # Extended back to 2017 to avoid COVID/Ukraine skew
END_DATE = "2026-03-30"     # Current date — will auto-clip to last available data

# ──────────────────────────────────────────────────────────────
# FORECAST HORIZONS (months)
# ──────────────────────────────────────────────────────────────

FORECAST_HORIZONS = [12, 24]  # 1-year and 2-year forward projections

# ──────────────────────────────────────────────────────────────
# YFINANCE TICKER MAP
# ──────────────────────────────────────────────────────────────

TICKERS = {
    "eurusd": "EURUSD=X",          # EUR/USD spot rate
    "brent":  "BZ=F",              # Brent Crude front-month future
    "us2y":   "2YY=F",             # 2-Year US Treasury yield future
    "us10y":  "^TNX",              # 10-Year US Treasury yield (backup)
    "dxy":    "DX-Y.NYB",          # US Dollar Index
    "euro_index": "EURUSD=X",      # Proxy — true Euro index not on yfinance
    "vix":    "^VIX",              # CBOE Volatility Index (risk proxy)
}

# ──────────────────────────────────────────────────────────────
# FRED SERIES IDS
# ──────────────────────────────────────────────────────────────

FRED_SERIES = {
    "us_2y_yield":   "DGS2",       # 2-Year US Treasury constant maturity
    "de_2y_yield":   "DFII5",      # German 2Y not directly on FRED — see note below
    "eu_hicp":       "CP0000EZ19M086NEST",  # Eurozone HICP inflation
    "us_cpi":        "CPIAUCSL",   # US CPI Urban Consumers
    "eu_pmi":        "MPMIEUMMA",  # Eurozone Manufacturing PMI (if available)
    "us_pmi":        "MPMIUS00MA", # US Manufacturing PMI (if available)
    "ecb_rate":      "ECBMRRFR",   # ECB Main Refinancing Rate
    "fed_rate":      "DFEDTARU",   # Fed Funds Upper Target
}

# NOTE ON GERMAN 2Y BUND:
# FRED does not carry the German 2-year Bund yield directly.
# We use Investing.com scraping or a synthetic from the ECB Statistical
# Data Warehouse as a fallback. See data/fetch_yields.py for implementation.
# A reliable FRED proxy: "IRLTLT01DEM156N" (long-term) is available but
# is 10Y, not 2Y. The fetch_yields module handles the workaround.

# ──────────────────────────────────────────────────────────────
# COT REPORT SETTINGS
# ──────────────────────────────────────────────────────────────

COT_CONFIG = {
    # CFTC bulk CSV URL template — financial futures (disaggregated)
    "base_url": "https://www.cftc.gov/files/dea/history/fin_fut_disagg_txt_{year}.zip",
    # Euro FX contract code in CFTC data
    "euro_fx_code": "099741",
    # Years to pull (will be clipped to START_DATE/END_DATE)
    "years": list(range(2017, 2027)),
}

# ──────────────────────────────────────────────────────────────
# SENTIMENT SETTINGS
# ──────────────────────────────────────────────────────────────

SENTIMENT_CONFIG = {
    "search_queries": [
        "EUR USD forecast",
        "Euro Dollar outlook",
        "ECB monetary policy",
        "Federal Reserve interest rate",
        "Eurozone economy",
    ],
    "max_articles_per_query": 20,
    "lookback_days": 90,   # Rolling sentiment window
}

# ──────────────────────────────────────────────────────────────
# MODEL PARAMETERS
# ──────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "var_max_lags": 12,         # AIC/BIC will select optimal within this
    "var_ic": "aic",            # Information criterion for lag selection
    "glm_family": "gaussian",   # Gaussian with log-link for non-linearity
    "train_test_split": 0.85,   # 85% train, 15% test
    "rolling_window": 52,       # 52-week rolling for correlations
    "stationarity_alpha": 0.05, # ADF test significance level
}

# ──────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "output"
PLOT_DIR = f"{OUTPUT_DIR}/plots"
DATA_DIR = f"{OUTPUT_DIR}/data"
MODEL_DIR = f"{OUTPUT_DIR}/models"