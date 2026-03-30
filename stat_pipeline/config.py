# eur_usd_pipeline/config.py
"""
Global configuration for the EUR/USD macro pipeline.
All API keys, date ranges, and shared parameters live here.

To check all listed below.
Double check for spelling mistakes and typos that could cause critical crash's 
Triple check to avoid what happened with the oil-vol pipeline
"""

# ──────────────────────────────────────────────────────────────
# API KEYS — I will replace the real APIs when the approval comes back from the fed. 
# I have used placeholders for now.
# ──────────────────────────────────────────────────────────────

FRED_API_KEY = "FRED_API-KEY" 

# GNews - shit but does the job
GNEWS_API_KEY = "gnews_API"

# ──────────────────────────────────────────────────────────────
# DATE RANGE
# ──────────────────────────────────────────────────────────────

# 2017 might seem random, but it gives data for pre ukraine and covid
START_DATE = "2017-01-01"   
END_DATE = "2026-03-30"    # Will need to update when final run 

# ──────────────────────────────────────────────────────────────
# FORECAST HORIZONS (months)
# ──────────────────────────────────────────────────────────────

FORECAST_HORIZONS = [12, 24]  # We havent decided on the horizon yet. MAybe 1, maybe 2 years (12, 24 months)

# ──────────────────────────────────────────────────────────────
# YFINANCE TICKER MAP
# ──────────────────────────────────────────────────────────────

TICKERS = {
    "eurusd": "EURUSD=X",          # EUR/USD spot rate
    "brent":  "BZ=F",              # Brent Crude front month future price
    "us2y":   "2YY=F",             # 2-Year US Treasury yield future 
    "us10y":  "^TNX",              # 10-Year uS Treasury yield  - this one is just in case 2 year gilts are corrupt. this will slow the model massively
    "dxy":    "DX-Y.NYB",          # US Dollar Index
    "euro_index": "EURUSD=X",      # Proxy — the real EUR index isnt on yfinance, and I dont know how to get a better source, but this is a valid replica
    "vix":    "^VIX",              # CBOE Volatility Index - from oil pipeline
}

# ──────────────────────────────────────────────────────────────
# FRED SERIES IDS
# ──────────────────────────────────────────────────────────────
# These are from chat - need to double check they exist and are live
# COnfirm here after I have checked - Confirmed?: 

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
    # CFTC bulk CSV URL template — financial futures, csv needs to be cleaned (sepr file later on)
    "base_url": "https://www.cftc.gov/files/dea/history/fin_fut_disagg_txt_{year}.zip",
    # Euro FX contract code in CFTC data
    "euro_fx_code": "099741",
    # Years to pull (will be clipped to aformentioned start and end dtaes above^)
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
