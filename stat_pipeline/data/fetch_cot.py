# eur_usd_pipeline/data/fetch_cot.py
"""
Fetch and parse CFTC Commitments of Traders (COT) report data.

The COT report shows how different market participants are positioned
in Euro FX futures. Key insight: if speculators are heavily short EUR,
any positive catalyst can trigger a short squeeze → rapid EUR rally.
If already massively long, the thesis may be a crowded trade.

Data source: CFTC bulk CSV downloads (disaggregated futures reports).
Updated weekly (Tuesday snapshot, released Friday).

KNOWN ISSUES & FIXES:
  - CFTC changed column names in 2017 vs earlier years.
    We normalise column names across all vintages.
  - ZIP files from CFTC occasionally contain multiple CSV files.
    We filter for the disaggregated futures-only file.
  - The 2024+ files may use a new URL pattern. We handle both
    the legacy and current patterns with fallback logic.
"""

import io
import os
import sys
import zipfile
import warnings
import requests
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COT_CONFIG, START_DATE, END_DATE
from utils.helpers import clean_series, resample_to_weekly, save_dataframe


# ──────────────────────────────────────────────────────────────
# COT URL PATTERNS
# ──────────────────────────────────────────────────────────────

# CFTC uses two URL patterns depending on the year:
#   Current year:  fut_disagg_txt_{year}.zip
#   Historical:    fin_fut_disagg_txt_{year}.zip
# We try both for resilience.

URL_PATTERNS = [
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
    "https://www.cftc.gov/files/dea/history/fin_fut_disagg_txt_{year}.zip",
    # Some years use 'dea_fut_xls' with CSV inside
    "https://www.cftc.gov/files/dea/history/deafut_txt_{year}.zip",
]

# Euro FX futures contract identifiers used by CFTC
EURO_FX_CODES = ["099741", "99741"]
EURO_FX_NAMES = ["EURO FX", "EURO FX - CHICAGO MERCANTILE EXCHANGE"]


# ──────────────────────────────────────────────────────────────
# COLUMN NAME NORMALISATION
# ──────────────────────────────────────────────────────────────

# CFTC column names vary across years and report types.
# We map the most common variants to canonical names.

COLUMN_MAP = {
    # Date
    "As of Date in Form YYYY-MM-DD": "date",
    "As_of_Date_In_Form_YYYY-MM-DD": "date",
    "Report_Date_as_YYYY-MM-DD": "date",
    "As of Date in Form YYMMDD": "date_short",

    # Contract identifier
    "CFTC Contract Market Code": "cftc_code",
    "CFTC_Contract_Market_Code": "cftc_code",
    "Market_and_Exchange_Names": "market_name",
    "Market and Exchange Names": "market_name",

    # Asset manager / institutional
    "Asset Mgr Positions-Long (All)": "am_long",
    "Asset_Mgr_Positions_Long_All": "am_long",
    "Asset Mgr Positions-Short (All)": "am_short",
    "Asset_Mgr_Positions_Short_All": "am_short",
    "Asset Mgr Positions-Spreading (All)": "am_spread",
    "Asset_Mgr_Positions_Spread_All": "am_spread",

    # Leveraged funds (hedge funds)
    "Lev Money Positions-Long (All)": "lev_long",
    "Lev_Money_Positions_Long_All": "lev_long",
    "Lev Money Positions-Short (All)": "lev_short",
    "Lev_Money_Positions_Short_All": "lev_short",
    "Lev Money Positions-Spreading (All)": "lev_spread",
    "Lev_Money_Positions_Spread_All": "lev_spread",

    # Dealer / intermediary
    "Dealer Positions-Long (All)": "dealer_long",
    "Dealer_Positions_Long_All": "dealer_long",
    "Dealer Positions-Short (All)": "dealer_short",
    "Dealer_Positions_Short_All": "dealer_short",

    # Non-commercial (legacy format)
    "NonComm_Positions_Long_All": "noncomm_long",
    "Noncommercial Positions-Long (All)": "noncomm_long",
    "NonComm_Positions_Short_All": "noncomm_short",
    "Noncommercial Positions-Short (All)": "noncomm_short",

    # Open interest
    "Open Interest (All)": "open_interest",
    "Open_Interest_All": "open_interest",
}


# ──────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────

def _download_cot_year(year):
    """
    Download and extract COT data for a single year.

    Tries multiple URL patterns. Returns a DataFrame or None.
    """
    for pattern in URL_PATTERNS:
        url = pattern.format(year=year)
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                continue

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                # Find the main CSV (largest file, or one containing 'disagg' or 'fut')
                csv_files = [f for f in zf.namelist() if f.lower().endswith(".txt") or f.lower().endswith(".csv")]
                if not csv_files:
                    continue

                # Prefer the disaggregated file
                target = csv_files[0]
                for f in csv_files:
                    if "disagg" in f.lower() or "fin_fut" in f.lower():
                        target = f
                        break

                with zf.open(target) as csv_file:
                    df = pd.read_csv(csv_file, low_memory=False)
                    return df

        except (requests.RequestException, zipfile.BadZipFile, Exception) as e:
            continue

    return None


def _normalise_columns(df):
    """Map variant CFTC column names to canonical names."""
    rename = {}
    for orig_col in df.columns:
        stripped = orig_col.strip()
        if stripped in COLUMN_MAP:
            rename[orig_col] = COLUMN_MAP[stripped]
    df = df.rename(columns=rename)
    return df


def _filter_euro_fx(df):
    """Filter for Euro FX futures contract rows only."""
    filtered = pd.DataFrame()

    # Try filtering by CFTC code
    if "cftc_code" in df.columns:
        code_col = df["cftc_code"].astype(str).str.strip()
        mask = code_col.isin(EURO_FX_CODES)
        if mask.any():
            filtered = df[mask].copy()

    # Fallback: filter by market name
    if filtered.empty and "market_name" in df.columns:
        name_col = df["market_name"].astype(str).str.upper()
        mask = name_col.apply(
            lambda x: any(name in x for name in EURO_FX_NAMES)
        )
        if mask.any():
            filtered = df[mask].copy()

    return filtered


def fetch_cot_data(start=START_DATE, end=END_DATE, save=True):
    """
    Fetch and parse COT data for Euro FX futures across all configured years.

    Returns
    -------
    pd.DataFrame
        Weekly COT positioning data with computed net positioning metrics.
    """
    print("[...] Fetching CFTC Commitments of Traders data")

    start_year = int(start[:4])
    end_year = int(end[:4])
    years = list(range(start_year, end_year + 1))

    all_frames = []

    for year in years:
        print(f"  [·] Downloading COT {year}...", end=" ")
        raw = _download_cot_year(year)

        if raw is None:
            print("FAILED (no data)")
            continue

        raw = _normalise_columns(raw)
        euro = _filter_euro_fx(raw)

        if euro.empty:
            print(f"OK ({len(raw)} rows total, 0 Euro FX)")
            continue

        print(f"OK ({len(euro)} Euro FX rows)")
        all_frames.append(euro)

    if not all_frames:
        raise RuntimeError(
            "Could not retrieve any COT data. Check internet connection "
            "and CFTC URL availability."
        )

    # Combine all years
    combined = pd.concat(all_frames, ignore_index=True)

    # Parse date
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    elif "date_short" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date_short"], format="%y%m%d", errors="coerce")
    else:
        raise RuntimeError("No date column found in COT data")

    combined = combined.set_index("date").sort_index()
    combined = combined.loc[start:end]

    # Drop duplicate dates (keep last reported)
    combined = combined[~combined.index.duplicated(keep="last")]

    # Compute positioning metrics
    combined = _compute_positioning(combined)

    print(f"[✓] COT data: {len(combined)} rows ({combined.index.min().date()} → {combined.index.max().date()})")

    if save:
        save_dataframe(combined, "cot_data.csv")

    return combined


# ──────────────────────────────────────────────────────────────
# POSITIONING CALCULATIONS
# ──────────────────────────────────────────────────────────────

def _compute_positioning(df):
    """
    Compute net speculative positioning metrics from raw COT data.

    Key metrics:
    - net_spec_position: leveraged funds net long/short
    - net_am_position: asset manager net long/short
    - positioning_index: percentile rank of current positioning
    - crowding_score: how extreme current positioning is (z-score)
    """

    # ── Net leveraged fund (hedge fund) positioning ──
    if "lev_long" in df.columns and "lev_short" in df.columns:
        df["lev_long"] = pd.to_numeric(df["lev_long"], errors="coerce")
        df["lev_short"] = pd.to_numeric(df["lev_short"], errors="coerce")
        df["net_lev_position"] = df["lev_long"] - df["lev_short"]
    else:
        df["net_lev_position"] = np.nan

    # ── Net asset manager positioning ──
    if "am_long" in df.columns and "am_short" in df.columns:
        df["am_long"] = pd.to_numeric(df["am_long"], errors="coerce")
        df["am_short"] = pd.to_numeric(df["am_short"], errors="coerce")
        df["net_am_position"] = df["am_long"] - df["am_short"]
    else:
        df["net_am_position"] = np.nan

    # ── Net dealer positioning ──
    if "dealer_long" in df.columns and "dealer_short" in df.columns:
        df["dealer_long"] = pd.to_numeric(df["dealer_long"], errors="coerce")
        df["dealer_short"] = pd.to_numeric(df["dealer_short"], errors="coerce")
        df["net_dealer_position"] = df["dealer_long"] - df["dealer_short"]
    else:
        df["net_dealer_position"] = np.nan

    # ── Fallback: non-commercial (legacy report) ──
    if "noncomm_long" in df.columns and "noncomm_short" in df.columns:
        df["noncomm_long"] = pd.to_numeric(df["noncomm_long"], errors="coerce")
        df["noncomm_short"] = pd.to_numeric(df["noncomm_short"], errors="coerce")
        df["net_noncomm_position"] = df["noncomm_long"] - df["noncomm_short"]

    # ── Composite speculative positioning ──
    # Prefer leveraged funds; fall back to non-commercial
    if df["net_lev_position"].notna().sum() > 10:
        df["net_spec_position"] = df["net_lev_position"]
    elif "net_noncomm_position" in df.columns:
        df["net_spec_position"] = df["net_noncomm_position"]
    else:
        df["net_spec_position"] = np.nan

    # ── Open interest ──
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

    # ── Positioning as % of open interest ──
    if "open_interest" in df.columns:
        df["spec_pct_oi"] = df["net_spec_position"] / df["open_interest"] * 100
    else:
        df["spec_pct_oi"] = np.nan

    # ── Percentile rank (historical context) ──
    # Where does current positioning sit in the full history?
    spec = df["net_spec_position"].dropna()
    if len(spec) > 20:
        df["positioning_percentile"] = spec.rank(pct=True) * 100
    else:
        df["positioning_percentile"] = np.nan

    # ── Crowding z-score ──
    # Z-score of net spec position using rolling 2-year (104-week) window
    rolling_mean = df["net_spec_position"].rolling(104, min_periods=52).mean()
    rolling_std = df["net_spec_position"].rolling(104, min_periods=52).std()
    df["crowding_zscore"] = (df["net_spec_position"] - rolling_mean) / rolling_std

    # ── Short squeeze probability proxy ──
    # When positioning is extremely short (z < -1.5) and starting to cover,
    # short squeeze risk is elevated
    df["squeeze_risk"] = np.where(
        (df["crowding_zscore"] < -1.5) & (df["net_spec_position"].diff() > 0),
        "HIGH",
        np.where(
            df["crowding_zscore"] < -1.0,
            "ELEVATED",
            np.where(
                df["crowding_zscore"] > 1.5,
                "CROWDED_LONG",
                "NEUTRAL"
            )
        )
    )

    # ── Weekly change in positioning ──
    df["spec_position_change"] = df["net_spec_position"].diff()
    df["spec_position_change_pct"] = df["net_spec_position"].pct_change()

    # ── Select output columns ──
    output_cols = [
        "net_spec_position", "net_lev_position", "net_am_position",
        "net_dealer_position", "open_interest", "spec_pct_oi",
        "positioning_percentile", "crowding_zscore", "squeeze_risk",
        "spec_position_change", "spec_position_change_pct",
    ]

    # Add non-commercial if available
    if "net_noncomm_position" in df.columns:
        output_cols.append("net_noncomm_position")

    # Keep only columns that exist and have data
    output_cols = [c for c in output_cols if c in df.columns]

    return df[output_cols]


# ──────────────────────────────────────────────────────────────
# POSITIONING SUMMARY
# ──────────────────────────────────────────────────────────────

def print_positioning_summary(cot_df):
    """Print a human-readable summary of current COT positioning."""
    if cot_df.empty:
        print("[!] No COT data to summarise")
        return

    latest = cot_df.iloc[-1]
    print("\n══════════════════════════════════════════════")
    print("   COT POSITIONING SUMMARY (Latest Report)")
    print("══════════════════════════════════════════════")

    if pd.notna(latest.get("net_spec_position")):
        net = latest["net_spec_position"]
        direction = "LONG" if net > 0 else "SHORT"
        print(f"  Net speculative position:  {net:,.0f} contracts ({direction})")

    if pd.notna(latest.get("spec_pct_oi")):
        print(f"  As % of open interest:     {latest['spec_pct_oi']:.1f}%")

    if pd.notna(latest.get("positioning_percentile")):
        print(f"  Historical percentile:     {latest['positioning_percentile']:.0f}th")

    if pd.notna(latest.get("crowding_zscore")):
        z = latest["crowding_zscore"]
        print(f"  Crowding z-score:          {z:.2f}", end="")
        if z > 2:
            print("  ← EXTREMELY CROWDED LONG")
        elif z > 1:
            print("  ← Moderately long")
        elif z < -2:
            print("  ← EXTREMELY SHORT (squeeze risk)")
        elif z < -1:
            print("  ← Moderately short")
        else:
            print("  ← Neutral range")

    if pd.notna(latest.get("squeeze_risk")):
        print(f"  Short squeeze risk:        {latest['squeeze_risk']}")

    if pd.notna(latest.get("spec_position_change")):
        chg = latest["spec_position_change"]
        direction = "added" if chg > 0 else "reduced"
        print(f"  Weekly change:             {chg:+,.0f} ({direction})")

    print("══════════════════════════════════════════════\n")


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_cot_data()
    print_positioning_summary(df)
    print("\nSample (last 5 rows):")
    print(df.tail())