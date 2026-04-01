# eur_usd_pipeline/data/fetch_macro.py
"""
Fetch macroeconomic data for EUR/USD analysis:
  - CPI differentials (US vs Eurozone)
  - Central bank policy rates (Fed vs ECB)
  - PMI differentials (manufacturing activity)
  - Real interest rate differentials
  - Current account / trade balance proxies

These macro fundamentals drive medium-to-long-term FX trends
and feed into both the VAR and GLM models.

Data sources:
  - FRED (US data + some Eurozone series)
  - ECB Statistical Data Warehouse (Eurozone-specific)
"""

import io
import warnings
import pandas as pd
import numpy as np
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE
from utils.helpers import clean_series, resample_to_weekly, save_dataframe


# ──────────────────────────────────────────────────────────────
# GENERIC FRED FETCHER
# ──────────────────────────────────────────────────────────────

def _fetch_fred_series(series_id, col_name, start=START_DATE, end=END_DATE):
    """
    Fetch a single FRED series with robust error handling.

    Tries fredapi first, falls back to REST API.

    Returns
    -------
    pd.DataFrame with single column `col_name`, datetime index.
    Returns empty DataFrame on failure (does not raise).
    """
    # ── Primary: fredapi ──
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        raw = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = raw.to_frame(name=col_name)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna()
        return df

    except Exception as e:
        pass

    # ── Fallback: REST API ──
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = [
            {"Date": obs["date"], col_name: obs["value"]}
            for obs in data.get("observations", [])
        ]
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna()
        return df

    except Exception as e:
        print(f"  [!] Could not fetch FRED series {series_id}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# ECB SDW GENERIC FETCHER
# ──────────────────────────────────────────────────────────────

def _fetch_ecb_series(series_key, col_name, start=START_DATE, end=END_DATE):
    """
    Fetch a series from the ECB Statistical Data Warehouse.

    Parameters
    ----------
    series_key : str
        Full ECB SDW series key (e.g. 'FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA').
    col_name : str
        Column name for the output DataFrame.

    Returns
    -------
    pd.DataFrame or empty DataFrame on failure.
    """
    # ECB SDW series keys contain the dataset as first segment
    parts = series_key.split(".", 1)
    if len(parts) < 2:
        print(f"  [!] Invalid ECB series key: {series_key}")
        return pd.DataFrame()

    dataset = parts[0]
    key = parts[1]
    url = f"https://data-api.ecb.europa.eu/service/data/{dataset}/{key}"

    headers = {"Accept": "text/csv"}
    params = {
        "startPeriod": start[:7],
        "endPeriod": end[:7],
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()

        csv_data = io.StringIO(resp.text)
        raw = pd.read_csv(csv_data)

        if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Date": pd.to_datetime(raw["TIME_PERIOD"]),
            col_name: pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
        })
        df = df.set_index("Date").sort_index().dropna()
        return df

    except Exception as e:
        print(f"  [!] ECB SDW fetch failed for {series_key}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# INFLATION DIFFERENTIAL
# ──────────────────────────────────────────────────────────────

def fetch_inflation_differential(start=START_DATE, end=END_DATE):
    """
    Compute US CPI vs Eurozone HICP inflation differential.

    Higher US inflation relative to Eurozone → USD weakening
    (purchasing power parity argument) → EUR positive.

    Returns
    -------
    pd.DataFrame
        Columns: ['us_cpi_yoy', 'eu_hicp_yoy', 'inflation_diff',
                   'inflation_diff_direction']
    """
    print("[...] Fetching inflation differential (US CPI vs EU HICP)")

    # ── US CPI ──
    us_cpi = _fetch_fred_series("CPIAUCSL", "us_cpi", start, end)

    # ── Eurozone HICP ──
    # Try FRED first, then ECB
    eu_hicp = _fetch_fred_series(
        FRED_SERIES.get("eu_hicp", "CP0000EZ19M086NEST"),
        "eu_hicp", start, end
    )

    if eu_hicp.empty:
        print("  [·] Trying ECB SDW for Eurozone HICP...")
        eu_hicp = _fetch_ecb_series(
            "ICP.M.U2.N.000000.4.ANR",  # HICP all items annual rate
            "eu_hicp", start, end
        )

    # ── Compute YoY rates ──
    df = pd.DataFrame()

    if not us_cpi.empty:
        # CPI is an index level — compute YoY % change
        us_cpi_monthly = us_cpi.resample("MS").last()
        df["us_cpi_yoy"] = us_cpi_monthly["us_cpi"].pct_change(periods=12) * 100
    else:
        print("  [!] US CPI not available")
        df["us_cpi_yoy"] = np.nan

    if not eu_hicp.empty:
        eu_hicp_monthly = eu_hicp.resample("MS").last()
        # ECB HICP may already be YoY rate — check magnitude
        if eu_hicp_monthly["eu_hicp"].mean() > 50:
            # It's an index level, compute YoY
            df["eu_hicp_yoy"] = eu_hicp_monthly["eu_hicp"].pct_change(periods=12) * 100
        else:
            # Already a YoY rate
            df = df.join(eu_hicp_monthly.rename(columns={"eu_hicp": "eu_hicp_yoy"}), how="outer")
    else:
        print("  [!] Eurozone HICP not available")
        df["eu_hicp_yoy"] = np.nan

    df = df.dropna(how="all")

    if not df.empty:
        # Inflation differential: US - EU
        # Positive → US inflation higher → USD should weaken → EUR positive
        df["inflation_diff"] = df["us_cpi_yoy"] - df["eu_hicp_yoy"]
        df["inflation_diff_direction"] = np.where(
            df["inflation_diff"] > 0, "USD_WEAKENING_PRESSURE", "EUR_WEAKENING_PRESSURE"
        )

    print(f"[✓] Inflation data: {len(df)} rows")
    return df


# ──────────────────────────────────────────────────────────────
# CENTRAL BANK RATE DIFFERENTIAL
# ──────────────────────────────────────────────────────────────

def fetch_rate_differential(start=START_DATE, end=END_DATE):
    """
    Compute Fed Funds Rate vs ECB Main Refinancing Rate differential.

    Higher US rates relative to ECB → capital flows to USD → EUR negative.
    Narrowing differential → EUR positive.

    Returns
    -------
    pd.DataFrame
        Columns: ['fed_rate', 'ecb_rate', 'rate_diff', 'rate_diff_change']
    """
    print("[...] Fetching central bank rate differential")

    # ── Fed Funds (upper target) ──
    fed = _fetch_fred_series(
        FRED_SERIES.get("fed_rate", "DFEDTARU"),
        "fed_rate", start, end
    )

    # ── ECB Main Refinancing Rate ──
    ecb = _fetch_fred_series(
        FRED_SERIES.get("ecb_rate", "ECBMRRFR"),
        "ecb_rate", start, end
    )

    if ecb.empty:
        print("  [·] Trying ECB SDW for refinancing rate...")
        ecb = _fetch_ecb_series(
            "FM.D.U2.EUR.4F.KR.MRR_FR.LEV",
            "ecb_rate", start, end
        )

    # Combine
    df = pd.DataFrame()

    if not fed.empty:
        df = df.join(fed.resample("MS").last(), how="outer")
    else:
        df["fed_rate"] = np.nan

    if not ecb.empty:
        df = df.join(ecb.resample("MS").last(), how="outer")
    else:
        df["ecb_rate"] = np.nan

    df = df.ffill().dropna(how="all")

    if not df.empty and "fed_rate" in df.columns and "ecb_rate" in df.columns:
        # Rate differential: Fed - ECB
        # Positive → US rates higher → USD attractive → EUR headwind
        # Narrowing → EUR tailwind
        df["rate_diff"] = df["fed_rate"] - df["ecb_rate"]
        df["rate_diff_change"] = df["rate_diff"].diff()
        df["rate_diff_direction"] = np.where(
            df["rate_diff_change"] < 0,
            "NARROWING_EUR_POSITIVE",
            "WIDENING_EUR_NEGATIVE"
        )

    print(f"[✓] Rate differential: {len(df)} rows")
    return df


# ──────────────────────────────────────────────────────────────
# PMI DIFFERENTIAL (MANUFACTURING ACTIVITY)
# ──────────────────────────────────────────────────────────────

def fetch_pmi_differential(start=START_DATE, end=END_DATE):
    """
    Fetch US vs Eurozone Manufacturing PMI differential.

    PMI > 50 = expansion. Higher relative PMI → stronger economy → currency positive.

    KNOWN ISSUE: FRED PMI series are sometimes discontinued or
    have restricted access. We use ISM Manufacturing (US) and
    attempt Eurozone composite from multiple sources.

    Returns
    -------
    pd.DataFrame
        Columns: ['us_pmi', 'eu_pmi', 'pmi_diff']
    """
    print("[...] Fetching PMI differential")

    # ── US ISM Manufacturing ──
    us_pmi = _fetch_fred_series("MANEMP", "us_pmi_proxy", start, end)
    if us_pmi.empty:
        # Try alternative: NAPM index
        us_pmi = _fetch_fred_series("NAPMPI", "us_pmi_proxy", start, end)
    if us_pmi.empty:
        # Last resort: industrial production as proxy
        us_pmi = _fetch_fred_series("INDPRO", "us_pmi_proxy", start, end)
        if not us_pmi.empty:
            # Convert index to pseudo-PMI (YoY change + 50)
            us_pmi_monthly = us_pmi.resample("MS").last()
            us_pmi_monthly["us_pmi"] = us_pmi_monthly["us_pmi_proxy"].pct_change(12) * 100 + 50
            us_pmi = us_pmi_monthly[["us_pmi"]]
            print("  [⚠] Using Industrial Production as US PMI proxy")

    if "us_pmi" not in us_pmi.columns and "us_pmi_proxy" in us_pmi.columns:
        us_pmi = us_pmi.rename(columns={"us_pmi_proxy": "us_pmi"})

    # ── Eurozone PMI ──
    eu_pmi = _fetch_fred_series(
        FRED_SERIES.get("eu_pmi", "MPMIEUMMA"),
        "eu_pmi", start, end
    )

    if eu_pmi.empty:
        # Try Eurozone industrial production as proxy
        eu_pmi = _fetch_fred_series("EA19PRINTO01GYSAM", "eu_pmi_proxy", start, end)
        if not eu_pmi.empty:
            eu_pmi_monthly = eu_pmi.resample("MS").last()
            eu_pmi_monthly["eu_pmi"] = eu_pmi_monthly["eu_pmi_proxy"] + 50
            eu_pmi = eu_pmi_monthly[["eu_pmi"]]
            print("  [⚠] Using Eurozone Industrial Production as PMI proxy")

    if "eu_pmi" not in eu_pmi.columns and "eu_pmi_proxy" in eu_pmi.columns:
        eu_pmi = eu_pmi.rename(columns={"eu_pmi_proxy": "eu_pmi"})

    # Combine
    df = pd.DataFrame()
    if not us_pmi.empty:
        df = df.join(us_pmi.resample("MS").last(), how="outer")
    if not eu_pmi.empty:
        df = df.join(eu_pmi.resample("MS").last(), how="outer")

    df = df.ffill(limit=3).dropna(how="all")

    if "us_pmi" in df.columns and "eu_pmi" in df.columns:
        # PMI diff: EU - US → positive = Eurozone outperforming → EUR positive
        df["pmi_diff"] = df["eu_pmi"] - df["us_pmi"]

    print(f"[✓] PMI data: {len(df)} rows")
    return df


# ──────────────────────────────────────────────────────────────
# REAL INTEREST RATE DIFFERENTIAL
# ──────────────────────────────────────────────────────────────

def compute_real_rate_differential(rate_df, inflation_df):
    """
    Compute real interest rate differential (nominal rate - inflation).

    Real rates drive capital flows more accurately than nominal rates.
    Higher US real rate → USD attractive → EUR negative.

    Parameters
    ----------
    rate_df : pd.DataFrame
        From fetch_rate_differential(), must have 'fed_rate', 'ecb_rate'.
    inflation_df : pd.DataFrame
        From fetch_inflation_differential(), must have 'us_cpi_yoy', 'eu_hicp_yoy'.

    Returns
    -------
    pd.DataFrame
        Columns: ['us_real_rate', 'eu_real_rate', 'real_rate_diff']
    """
    print("[...] Computing real interest rate differential")

    merged = rate_df.join(inflation_df, how="inner")
    merged = merged.ffill(limit=3).dropna(
        subset=["fed_rate", "ecb_rate"],
        how="all"
    )

    df = pd.DataFrame(index=merged.index)

    # Real rate = nominal - inflation
    if "fed_rate" in merged.columns and "us_cpi_yoy" in merged.columns:
        df["us_real_rate"] = merged["fed_rate"] - merged["us_cpi_yoy"]
    else:
        df["us_real_rate"] = np.nan

    if "ecb_rate" in merged.columns and "eu_hicp_yoy" in merged.columns:
        df["eu_real_rate"] = merged["ecb_rate"] - merged["eu_hicp_yoy"]
    else:
        df["eu_real_rate"] = np.nan

    df = df.dropna(how="all")

    if "us_real_rate" in df.columns and "eu_real_rate" in df.columns:
        # Real rate diff: EU - US → positive = Eurozone real rates higher → EUR positive
        df["real_rate_diff"] = df["eu_real_rate"] - df["us_real_rate"]
        df["real_rate_direction"] = np.where(
            df["real_rate_diff"].diff() > 0,
            "EUR_IMPROVING",
            "EUR_DETERIORATING"
        )

    print(f"[✓] Real rate differential: {len(df)} rows")
    return df


# ──────────────────────────────────────────────────────────────
# EUROZONE CURRENT ACCOUNT (TRADE BALANCE PROXY)
# ──────────────────────────────────────────────────────────────

def fetch_current_account(start=START_DATE, end=END_DATE):
    """
    Fetch Eurozone current account balance from ECB SDW.

    Persistent surplus → capital inflows → EUR supportive.
    Deteriorating balance → EUR headwind.

    Returns
    -------
    pd.DataFrame
        Columns: ['ez_current_account', 'ca_trend']
    """
    print("[...] Fetching Eurozone current account balance")

    # ECB SDW: Eurozone balance of payments — current account total
    df = _fetch_ecb_series(
        "BP6.Q.N.I8.W1.S1.S1.T.B.CA._Z._Z._Z.EUR._T._X.N",
        "ez_current_account", start, end
    )

    if df.empty:
        # Fallback: try FRED
        df = _fetch_fred_series("BPBLTT01EZQ659S", "ez_current_account", start, end)

    if df.empty:
        print("  [!] Current account data not available from any source")
        return pd.DataFrame()

    # 4-quarter rolling sum for annualised view
    df_q = df.resample("QS").last().dropna()
    df_q["ca_rolling_4q"] = df_q["ez_current_account"].rolling(4).sum()

    # Trend direction
    df_q["ca_trend"] = np.where(
        df_q["ca_rolling_4q"].diff() > 0,
        "IMPROVING",
        "DETERIORATING"
    )

    print(f"[✓] Current account: {len(df_q)} rows")
    return df_q


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FETCH ALL MACRO DATA
# ──────────────────────────────────────────────────────────────

def fetch_all_macro(start=START_DATE, end=END_DATE, save=True):
    """
    Fetch all macroeconomic indicators and combine.

    Returns
    -------
    dict of pd.DataFrames
        Keys: 'inflation', 'rates', 'pmi', 'real_rates', 'current_account'
    Also returns a merged monthly DataFrame.
    """
    print("\n═══ MACRO DATA FETCH ═══")

    inflation = fetch_inflation_differential(start=start, end=end)
    rates = fetch_rate_differential(start=start, end=end)
    pmi = fetch_pmi_differential(start=start, end=end)

    # Real rates need both nominal rates and inflation
    if not rates.empty and not inflation.empty:
        real_rates = compute_real_rate_differential(rates, inflation)
    else:
        real_rates = pd.DataFrame()
        print("  [!] Skipping real rate differential (missing inputs)")

    current_account = fetch_current_account(start=start, end=end)

    # ── Merge all into a single monthly DataFrame ──
    print("\n[...] Merging macro indicators")

    # Resample everything to monthly
    frames = {}
    for name, df in [
        ("inflation", inflation),
        ("rates", rates),
        ("pmi", pmi),
        ("real_rates", real_rates),
    ]:
        if not df.empty:
            monthly = df.resample("MS").last()
            frames[name] = monthly

    if frames:
        combined = pd.DataFrame()
        for name, df in frames.items():
            # Avoid duplicate columns
            new_cols = [c for c in df.columns if c not in combined.columns]
            if new_cols:
                combined = combined.join(df[new_cols], how="outer") if not combined.empty else df[new_cols].copy()

        combined = combined.ffill(limit=3)
        print(f"[✓] Combined macro data: {len(combined)} rows, {len(combined.columns)} columns")
    else:
        combined = pd.DataFrame()
        print("  [!] No macro data available to combine")

    # Save individual and combined
    if save and not combined.empty:
        save_dataframe(combined, "macro_data.csv")

    result = {
        "inflation": inflation,
        "rates": rates,
        "pmi": pmi,
        "real_rates": real_rates,
        "current_account": current_account,
        "combined": combined,
    }

    return result


# ──────────────────────────────────────────────────────────────
# MACRO SUMMARY
# ──────────────────────────────────────────────────────────────

def print_macro_summary(macro_dict):
    """Print a human-readable summary of macro conditions."""
    print("\n══════════════════════════════════════════════")
    print("   MACRO CONDITIONS SUMMARY")
    print("══════════════════════════════════════════════")

    # Inflation
    inf = macro_dict.get("inflation")
    if inf is not None and not inf.empty:
        latest = inf.iloc[-1]
        if pd.notna(latest.get("inflation_diff")):
            print(f"  Inflation diff (US-EU):  {latest['inflation_diff']:.2f}pp")
            print(f"  → {latest.get('inflation_diff_direction', 'N/A')}")

    # Rates
    rates = macro_dict.get("rates")
    if rates is not None and not rates.empty:
        latest = rates.iloc[-1]
        if pd.notna(latest.get("rate_diff")):
            print(f"  Rate diff (Fed-ECB):     {latest['rate_diff']:.2f}pp")
            print(f"  → {latest.get('rate_diff_direction', 'N/A')}")

    # Real rates
    rr = macro_dict.get("real_rates")
    if rr is not None and not rr.empty:
        latest = rr.iloc[-1]
        if pd.notna(latest.get("real_rate_diff")):
            print(f"  Real rate diff (EU-US):  {latest['real_rate_diff']:.2f}pp")
            print(f"  → {latest.get('real_rate_direction', 'N/A')}")

    # PMI
    pmi = macro_dict.get("pmi")
    if pmi is not None and not pmi.empty:
        latest = pmi.iloc[-1]
        if pd.notna(latest.get("pmi_diff")):
            print(f"  PMI diff (EU-US):        {latest['pmi_diff']:.1f}")
            eu_status = "expanding" if latest.get("eu_pmi", 0) > 50 else "contracting"
            print(f"  Eurozone manufacturing:  {eu_status}")

    print("══════════════════════════════════════════════\n")


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    macro = fetch_all_macro()
    print_macro_summary(macro)