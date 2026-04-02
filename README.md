# EUR/USD Macro Analysis Pipeline

A comprehensive Python pipeline that examines whether the Euro will appreciate
against the US Dollar over a 1–2 year horizon, using multi-factor macro analysis,
time series modelling, and NLP sentiment scoring.

### Note

Part of this README.md file has been edited using AI, but all revisions have been checked and correct

---

## Thesis

> Can we identify and model the key macro drivers of EUR/USD to determine
> whether the Euro is likely to appreciate over the next 12–24 months?

The pipeline analyses five core channels:

1. **Yield Spreads** — 2Y German Bund vs US Treasury spread dynamics
2. **Terms of Trade** — Brent Crude oil impact on Eurozone trade balance
3. **Speculative Positioning** — CFTC COT report analysis for squeeze setups
4. **Options Sentiment** — 25Δ risk reversal proxy for call/put skew
5. **Macro Fundamentals** — Inflation, rates, PMI, current account differentials

---

## Project Structure
```
 The ascii tree for the pipeline is laid out here
 ** This is still te first draft, layouts may have been changed, files might have been added/removed to improve the pipeline
 Will edit as needed throughout the coding process

eur_usd_pipeline/
│
├── config.py                          # API keys date ranges and global paramaters
├── main.py                            # Orchestrates full pipeline
├── requirements.txt
├── README.md                          # Note done yet - leave until end though
│
├── data/
│   ├── __init__.py
│   ├── fetch_fx.py                    # EUR/USD spot using yfinance
│   ├── fetch_yields.py                # US 2Y Treasury & German 2Y Bund (bc ECB doesnt issue, Ger dominates EU economically)
│   ├── fetch_oil.py                   # Brent Crude prices
│   ├── fetch_cot.py                   # CFTC Commitments of Traders
│   ├── fetch_risk_reversals.py        # 25-delta risk reversals (synthetic)
│   └── fetch_macro.py                 # CPI differentials, PMI anf rate diffs
│
├── features/
│   ├── __init__.py
│   ├── sentiment.py                   # News sentiment via APIs (standard free ones, easy point of upgrade for future?)
│   ├── technical.py                   # RSI, MACD, Bollinger, momentum
│   ├── spreads.py                     # Yield spreads, ToT construction
│   └── engineer.py                    # Master feature builder/merger
│
├── models/
│   ├── __init__.py
│   ├── var_model.py                   # Vector Autoregression model
│   ├── glm_model.py                   # GLM (Gamma/Gaussian families) - Generalised Linear Model
│   ├── granger.py                     # Granger causality tests - Need to improve understanding before writing the code structure
│   └── forecast.py                    # Forecast comparison & diagnostics
│
├── visualization/
│   ├── __init__.py
│   ├── yield_spread_plot.py           # Yield spread vs EUR/USD spot
│   ├── terms_of_trade_plot.py         # Brent (inverted) vs Euro index
│   ├── cot_plot.py                    # Net speculative positioning
│   ├── risk_reversal_plot.py          # 25 delta RR overlay
│   ├── model_plots.py                 # VAR vs GLM comparison visuals - Ho mcuh information can we capture assuming linear relationships between variables?
│   └── dashboard.py                   # Combined summary dashboard
│
└── utils/
    ├── __init__.py
    └── helpers.py                     # Date alignment, stationarity, I/O
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Edit `config.py` and replace the placeholder strings:
```python
FRED_API_KEY = "fred api key"   # this will need to be requested from the fred website
GNEWS_API_KEY = "gnews api key" # gnews is basic but has a working and free api, which is ideal for me
```

The FRED API key is **required** for US Treasury yield and macro data.
Register: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

### 3. Run the pipeline
```bash
# Full pipeline (data + models + plots)
python main.py

# Skip plot generation (faster)
python main.py --skip-plots

# Data and features only
python main.py --data-only
```

Estimated runtime: **5–15 minutes** (depends on internet speed and CFTC download).

---

## Data Sources

| Source | Data | Auth Required |
|--------|------|---------------|
| yfinance | EUR/USD, Brent, DXY, VIX, US 2Y futures | No |
| FRED API | US 2Y yield, CPI, Fed Funds, ECB rate, PMI | Yes (free key) |
| ECB SDW | German 2Y Bund yield, Eurozone HICP, current account | No |
| CFTC | Commitments of Traders (COT) bulk CSVs | No |
| GNews | Financial news articles for sentiment | Optional |

---

## Models

### Vector Autoregression (VAR)
- Multi-equation system capturing dynamic interdependencies
- AIC-optimised lag selection (up to 12 lags)
- Impulse Response Functions (IRFs) showing shock propagation
- Forecast Error Variance Decomposition (FEVD)
- 12-month and 24-month point forecasts with 95% confidence intervals

### Generalised Linear Model (GLM)
- Single-equation EUR/USD predictor with non-linear features
- Multiple specifications tested: Gaussian/identity, Gaussian/log, reduced, interaction-only
- Automatic VIF screening to manage multicollinearity
- Feature importance ranking by standardised coefficients
- Multi-horizon predictions (4w, 13w, 26w, 52w forward returns)

### Granger Causality
- Pairwise and bidirectional tests
- Rolling windows to detect time-varying causality
- Lag-specific profiles (at which lead time does each factor best predict EUR/USD?)
- Causality network with centrality analysis (leading vs lagging indicators)

### Additional Methods
- Hurst exponent (trend vs mean-reversion regime detection)
- Ensemble forecasting (VAR + GLM weighted combination)
- Scenario analysis (base, bull, bear, geopolitical shock)
- Composite macro scoring (multi-factor signal aggregation)

---

## Outputs

### CSV Data Files (`output/data/`)
- `fx_data.csv` — EUR/USD, DXY, VIX
- `yield_data.csv` — US 2Y, DE 2Y, yield spread
- `oil_data.csv` — Brent, inverted Brent, energy pressure index
- `cot_data.csv` — COT positioning metrics
- `risk_reversal_data.csv` — Synthetic 25Δ RR proxy
- `macro_data.csv` — Inflation, rates, PMI differentials
- `sentiment_data.csv` — NLP sentiment scores
- `full_features.csv` — Complete engineered feature set
- `train_features.csv` / `test_features.csv` — Model-ready splits

### Model Results (`output/models/`)
- `var_forecast_12m.csv` / `var_forecast_24m.csv`
- `granger_causality.csv` / `granger_matrix.csv`
- `fevd_table.csv`
- `glm_all_comparisons.csv` / `glm_importance_*.csv`
- `model_comparison.csv`
- `scenario_analysis.csv`
- `thesis_assessment.csv`

### Visualisations (`output/plots/`)
- `yield_spread_vs_eurusd.png` — Core yield spread thesis chart
- `yield_spread_correlation.png` — Rolling correlation regime analysis
- `terms_of_trade.png` — Oil/ToT vs EUR/USD multi-panel
- `energy_pressure.png` — Energy pressure index
- `cot_positioning.png` — COT with squeeze zones
- `dealer_vs_spec.png` — Smart money divergence
- `positioning_heatmap.png` — Historical percentile strip
- `risk_reversals.png` — 25Δ RR with regime bands
- `rr_vs_positioning.png` — Options vs futures cross-chart
- `var_forecast.png` — VAR projections with confidence bands
- `impulse_response.png` — IRF grid
- `fevd.png` — Variance decomposition stacked area
- `granger_heatmap.png` — Causality matrix
- `glm_importance_*.png` — Feature importance bar charts
- `rolling_correlation_heatmap.png` — Regime shift detection
- `residual_diagnostics.png` — Model health checks
- `scenario_analysis.png` — Bull/bear/shock outcomes
- `dashboard.png` — Combined multi-panel summary

---

## Key Limitations

1. **Synthetic Risk Reversals** — The 25Δ RR is a model-estimated proxy,
   not actual options market data. Real data requires Bloomberg or Refinitiv.

2. **German 2Y Bund Yield** — If the ECB SDW API is unreachable, a synthetic
   proxy (ECB rate + 50bp spread) is used. This is a rough approximation.

3. **Sentiment Coverage** — GNews only covers ~90 days of history. For the
   full 2017–2026 period, a market-revealed sentiment proxy is used instead.

4. **COT Data Granularity** — Weekly frequency only (Tuesday snapshot,
   Friday release). Intra-week positioning shifts are not captured.

5. **VAR Stationarity** — All series are differenced to achieve stationarity,
   which means the VAR models changes rather than levels. Forecasts are
   converted back to levels via cumulative summation.

6. **No Transaction Costs** — Directional accuracy metrics do not account
   for bid-ask spreads, slippage, or execution costs.

---

## Extending the Pipeline

- **Add Bloomberg data**: Replace the synthetic RR in `fetch_risk_reversals.py`
  with actual 25Δ RR from `blpapi`.
- **Add more currencies**: Extend `config.py` with GBP/USD, USD/JPY tickers
  and create parallel feature pipelines.
- **Machine learning**: Add `models/ml_model.py` with Random Forest or XGBoost
  for non-linear prediction comparison.
- **Real-time updates**: Wrap `main.py` in a scheduler (cron, Airflow) for
  weekly re-runs with fresh data.

---

## License

This project is for research and educational purposes. Data sourced from
public APIs — refer to each provider's terms of service for usage restrictions.
