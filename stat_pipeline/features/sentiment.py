# eur_usd_pipeline/features/sentiment.py
"""
News sentiment scoring for EUR/USD-relevant headlines.

Sources:
  - GNews API (free tier: 100 requests/day, no key needed for basic use)
  - Fallback: RSS feeds from major financial outlets

NLP engines:
  - VADER (Valence Aware Dictionary for sEntiment Reasoning)
    → tuned for social media / news headlines, handles negation well
  - TextBlob
    → general-purpose, useful as a second opinion

We produce a composite sentiment score that captures the market's
mood toward the Euro, ECB policy, and transatlantic macro themes.

KNOWN ISSUES & FIXES:
  - GNews sometimes returns duplicate articles across queries.
    We deduplicate by URL before scoring.
  - VADER can mis-score financial jargon ('cut' = negative in general
    English but 'rate cut' is context-dependent). We apply a
    financial-domain adjustment layer.
  - TextBlob requires the 'punkt' tokenizer from NLTK on first run.
    We handle the download gracefully.
"""

import warnings
import hashlib
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GNEWS_API_KEY, SENTIMENT_CONFIG, START_DATE, END_DATE
from utils.helpers import save_dataframe


# ──────────────────────────────────────────────────────────────
# NLP ENGINE SETUP
# ──────────────────────────────────────────────────────────────

def _init_vader():
    """Initialise VADER sentiment analyser."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        print("  [!] vaderSentiment not installed. Run: pip install vaderSentiment")
        return None


def _init_textblob():
    """Initialise TextBlob, downloading corpora if needed."""
    try:
        from textblob import TextBlob
        # Test that corpora are available
        try:
            _ = TextBlob("test").sentiment
        except Exception:
            print("  [·] Downloading TextBlob corpora...")
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
        return TextBlob
    except ImportError:
        print("  [!] textblob not installed. Run: pip install textblob")
        return None


# ──────────────────────────────────────────────────────────────
# FINANCIAL DOMAIN ADJUSTMENTS
# ──────────────────────────────────────────────────────────────

# Words that have different sentiment in finance vs general English
FINANCE_OVERRIDES = {
    # Phrase → (EUR sentiment adjustment, confidence)
    # Positive for EUR
    "euro rises": (0.5, 0.8),
    "euro gains": (0.5, 0.8),
    "euro rallies": (0.6, 0.8),
    "euro strengthens": (0.5, 0.8),
    "ecb hawkish": (0.4, 0.7),
    "ecb rate hike": (0.5, 0.7),
    "eurozone growth": (0.4, 0.7),
    "eurozone recovery": (0.4, 0.7),
    "eurozone surplus": (0.3, 0.6),
    "dollar weakens": (0.4, 0.7),
    "dollar falls": (0.4, 0.7),
    "fed dovish": (0.3, 0.6),
    "fed rate cut": (0.3, 0.6),
    "fed pause": (0.2, 0.5),
    "short squeeze euro": (0.5, 0.7),

    # Negative for EUR
    "euro falls": (-0.5, 0.8),
    "euro drops": (-0.5, 0.8),
    "euro weakens": (-0.5, 0.8),
    "euro slides": (-0.5, 0.8),
    "ecb dovish": (-0.4, 0.7),
    "ecb rate cut": (-0.4, 0.7),
    "eurozone recession": (-0.5, 0.8),
    "eurozone contraction": (-0.4, 0.7),
    "eurozone crisis": (-0.6, 0.8),
    "dollar strengthens": (-0.4, 0.7),
    "dollar rallies": (-0.4, 0.7),
    "dollar surges": (-0.5, 0.7),
    "fed hawkish": (-0.3, 0.6),
    "fed rate hike": (-0.3, 0.6),
    "trade war": (-0.3, 0.6),
    "tariff": (-0.2, 0.5),
}


def _apply_finance_adjustment(text, base_score):
    """
    Adjust a sentiment score based on financial domain phrases.

    Parameters
    ----------
    text : str
        Headline or article text (lowercase).
    base_score : float
        Original sentiment score [-1, 1].

    Returns
    -------
    float
        Adjusted score.
    """
    text_lower = text.lower()
    adjustment = 0.0
    total_confidence = 0.0

    for phrase, (adj, conf) in FINANCE_OVERRIDES.items():
        if phrase in text_lower:
            adjustment += adj * conf
            total_confidence += conf

    if total_confidence > 0:
        # Blend: weight original score vs finance-specific signal
        blended = (base_score * 0.4) + (adjustment / total_confidence * 0.6)
        return np.clip(blended, -1.0, 1.0)

    return base_score


# ──────────────────────────────────────────────────────────────
# ARTICLE FETCHING (GNEWS)
# ──────────────────────────────────────────────────────────────

def _fetch_gnews_articles(query, max_results=20, lookback_days=90):
    """
    Fetch articles from GNews API.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int
        Maximum articles to return.
    lookback_days : int
        How far back to search.

    Returns
    -------
    list of dict
        Each dict has 'title', 'description', 'url', 'publishedAt'.
    """
    try:
        from gnews import GNews

        gn = GNews(
            language="en",
            country="US",
            period=f"{lookback_days}d",
            max_results=max_results,
        )

        articles = gn.get_news(query)

        if articles is None:
            return []

        results = []
        for article in articles:
            results.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "published_at": article.get("published date", ""),
                "source": article.get("publisher", {}).get("title", "unknown"),
                "query": query,
            })

        return results

    except Exception as e:
        print(f"  [!] GNews fetch failed for '{query}': {e}")
        return []


def _fetch_gnews_api_direct(query, max_results=10, lookback_days=90):
    """
    Direct GNews REST API call (requires API key for higher limits).

    Fallback if the gnews Python package has issues.
    """
    import requests

    if GNEWS_API_KEY == "YOUR_GNEWS_API_KEY_HERE":
        return []

    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": min(max_results, 10),
        "from": from_date,
        "apikey": GNEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for article in data.get("articles", []):
            results.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "unknown"),
                "query": query,
            })
        return results

    except Exception as e:
        print(f"  [!] GNews API direct call failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# ARTICLE SCORING
# ──────────────────────────────────────────────────────────────

def score_articles(articles):
    """
    Score a list of articles using VADER + TextBlob with finance adjustments.

    Parameters
    ----------
    articles : list of dict
        Each must have 'title' and optionally 'description'.

    Returns
    -------
    pd.DataFrame
        Scored articles with columns:
        ['title', 'source', 'published_at', 'query',
         'vader_score', 'textblob_score', 'finance_score',
         'composite_score', 'sentiment_label']
    """
    vader = _init_vader()
    TextBlobClass = _init_textblob()

    scored = []

    for article in articles:
        title = article.get("title", "")
        desc = article.get("description", "")
        text = f"{title}. {desc}".strip()

        if not text or text == ".":
            continue

        record = {
            "title": title,
            "source": article.get("source", ""),
            "published_at": article.get("published_at", ""),
            "query": article.get("query", ""),
            "url": article.get("url", ""),
        }

        # ── VADER score ──
        if vader:
            vs = vader.polarity_scores(text)
            record["vader_score"] = vs["compound"]  # [-1, 1]
        else:
            record["vader_score"] = 0.0

        # ── TextBlob score ──
        if TextBlobClass:
            try:
                blob = TextBlobClass(text)
                record["textblob_score"] = blob.sentiment.polarity  # [-1, 1]
            except Exception:
                record["textblob_score"] = 0.0
        else:
            record["textblob_score"] = 0.0

        # ── Finance-domain adjusted score ──
        # Base: average of VADER and TextBlob
        base = (record["vader_score"] + record["textblob_score"]) / 2
        record["finance_score"] = _apply_finance_adjustment(text, base)

        # ── Composite ──
        # Weighted: 40% VADER, 20% TextBlob, 40% finance-adjusted
        record["composite_score"] = (
            0.40 * record["vader_score"]
            + 0.20 * record["textblob_score"]
            + 0.40 * record["finance_score"]
        )

        # ── Label ──
        cs = record["composite_score"]
        if cs > 0.15:
            record["sentiment_label"] = "BULLISH_EUR"
        elif cs < -0.15:
            record["sentiment_label"] = "BEARISH_EUR"
        else:
            record["sentiment_label"] = "NEUTRAL"

        scored.append(record)

    return pd.DataFrame(scored)


# ──────────────────────────────────────────────────────────────
# AGGREGATE SENTIMENT TIME SERIES
# ──────────────────────────────────────────────────────────────

def build_sentiment_timeseries(scored_df):
    """
    Aggregate article-level scores into a time series.

    Groups by date and computes daily / weekly sentiment metrics.

    Parameters
    ----------
    scored_df : pd.DataFrame
        From score_articles().

    Returns
    -------
    pd.DataFrame
        Weekly sentiment time series.
    """
    if scored_df.empty:
        print("  [!] No scored articles to build time series from")
        return pd.DataFrame()

    df = scored_df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    # Daily aggregation
    daily = df.groupby(df.index.date).agg(
        sentiment_mean=("composite_score", "mean"),
        sentiment_median=("composite_score", "median"),
        sentiment_std=("composite_score", "std"),
        n_articles=("composite_score", "count"),
        bullish_pct=("sentiment_label", lambda x: (x == "BULLISH_EUR").mean()),
        bearish_pct=("sentiment_label", lambda x: (x == "BEARISH_EUR").mean()),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"

    # Weekly aggregation
    weekly = daily.resample("W-FRI").agg({
        "sentiment_mean": "mean",
        "sentiment_median": "median",
        "sentiment_std": "mean",
        "n_articles": "sum",
        "bullish_pct": "mean",
        "bearish_pct": "mean",
    })

    # Sentiment momentum (2-week change in average sentiment)
    weekly["sentiment_momentum"] = weekly["sentiment_mean"].diff(2)

    # Bull-bear spread
    weekly["bull_bear_spread"] = weekly["bullish_pct"] - weekly["bearish_pct"]

    weekly = weekly.dropna(how="all")

    return weekly


# ──────────────────────────────────────────────────────────────
# HISTORICAL SENTIMENT PROXY (WHEN LIVE NEWS IS LIMITED)
# ──────────────────────────────────────────────────────────────

def build_historical_sentiment_proxy(eurusd_df, vix_df=None):
    """
    Build a proxy sentiment indicator from market data when
    live news coverage is limited (GNews only covers ~90 days).

    Uses:
    - EUR/USD momentum as a revealed-preference sentiment proxy
    - VIX as a risk-appetite proxy
    - Return dispersion as an uncertainty proxy

    This extends sentiment coverage back to 2017.

    Parameters
    ----------
    eurusd_df : pd.DataFrame
        Must have 'eurusd_close' and 'eurusd_return' columns.
    vix_df : pd.DataFrame, optional
        Must have 'vix_close' column.

    Returns
    -------
    pd.DataFrame
        Columns: ['proxy_sentiment', 'risk_appetite', 'uncertainty_proxy']
    """
    print("[...] Building historical sentiment proxy from market data")

    df = eurusd_df.copy()

    # ── Momentum-based sentiment ──
    # 4-week return z-scored → positive = bullish sentiment revealed
    ret_4w = df["eurusd_close"].pct_change(4)
    ret_mean = ret_4w.rolling(52, min_periods=26).mean()
    ret_std = ret_4w.rolling(52, min_periods=26).std()
    df["momentum_sentiment"] = (ret_4w - ret_mean) / ret_std

    # ── Mean reversion signal ──
    # Distance from 52-week mean → extreme moves suggest sentiment overshoot
    ma_52 = df["eurusd_close"].rolling(52).mean()
    std_52 = df["eurusd_close"].rolling(52).std()
    df["mean_reversion_signal"] = (df["eurusd_close"] - ma_52) / std_52

    # ── Risk appetite (from VIX if available) ──
    if vix_df is not None and "vix_close" in vix_df.columns:
        merged = df.join(vix_df[["vix_close"]], how="left")
        merged["vix_close"] = merged["vix_close"].ffill()

        # Invert and z-score: low VIX = high risk appetite = EUR positive
        vix_z = (
            (merged["vix_close"] - merged["vix_close"].rolling(52).mean())
            / merged["vix_close"].rolling(52).std()
        )
        df["risk_appetite"] = -vix_z  # Invert so positive = risk-on
    else:
        df["risk_appetite"] = 0

    # ── Uncertainty proxy ──
    # Rolling 13-week realised vol of weekly returns
    if "eurusd_return" in df.columns:
        df["uncertainty_proxy"] = df["eurusd_return"].rolling(13).std() * np.sqrt(52)
    else:
        df["uncertainty_proxy"] = np.nan

    # ── Composite proxy sentiment ──
    # Blend momentum and risk appetite
    df["proxy_sentiment"] = (
        0.5 * df["momentum_sentiment"].fillna(0)
        + 0.3 * df["risk_appetite"].fillna(0)
        + 0.2 * (-df["mean_reversion_signal"].fillna(0))  # Contrarian signal
    )

    output_cols = ["proxy_sentiment", "risk_appetite", "uncertainty_proxy",
                   "momentum_sentiment", "mean_reversion_signal"]
    output = df[[c for c in output_cols if c in df.columns]].dropna(how="all")

    print(f"[✓] Historical sentiment proxy: {len(output)} rows")
    return output


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FULL SENTIMENT PIPELINE
# ──────────────────────────────────────────────────────────────

def fetch_all_sentiment(eurusd_df=None, vix_df=None, save=True):
    """
    Run the full sentiment pipeline:
      1. Fetch recent news articles
      2. Score with NLP
      3. Build live sentiment time series
      4. Build historical proxy for full date range
      5. Combine

    Parameters
    ----------
    eurusd_df : pd.DataFrame, optional
        EUR/USD data for historical proxy. If None, proxy is skipped.
    vix_df : pd.DataFrame, optional
        VIX data for risk appetite component.

    Returns
    -------
    dict with keys: 'articles', 'live_sentiment', 'proxy_sentiment', 'combined'
    """
    print("\n═══ SENTIMENT ANALYSIS ═══")

    # ── Step 1: Fetch articles ──
    print("[...] Fetching news articles")
    all_articles = []
    queries = SENTIMENT_CONFIG.get("search_queries", [])
    max_per = SENTIMENT_CONFIG.get("max_articles_per_query", 20)
    lookback = SENTIMENT_CONFIG.get("lookback_days", 90)

    for query in queries:
        print(f"  [·] Searching: '{query}'")
        articles = _fetch_gnews_articles(query, max_results=max_per, lookback_days=lookback)

        # Fallback to direct API if gnews package returned nothing
        if not articles:
            articles = _fetch_gnews_api_direct(query, max_results=max_per, lookback_days=lookback)

        all_articles.extend(articles)
        print(f"      → {len(articles)} articles")

    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for a in all_articles:
        url_hash = hashlib.md5(a.get("url", "").encode()).hexdigest()
        if url_hash not in seen_urls:
            seen_urls.add(url_hash)
            unique_articles.append(a)

    print(f"[✓] Total unique articles: {len(unique_articles)}")

    # ── Step 2: Score ──
    scored_df = pd.DataFrame()
    if unique_articles:
        print("[...] Scoring articles with VADER + TextBlob")
        scored_df = score_articles(unique_articles)
        print(f"[✓] Scored {len(scored_df)} articles")

        if not scored_df.empty:
            bull = (scored_df["sentiment_label"] == "BULLISH_EUR").sum()
            bear = (scored_df["sentiment_label"] == "BEARISH_EUR").sum()
            neut = (scored_df["sentiment_label"] == "NEUTRAL").sum()
            print(f"    Bullish: {bull}  |  Bearish: {bear}  |  Neutral: {neut}")
            print(f"    Mean composite score: {scored_df['composite_score'].mean():.3f}")

    # ── Step 3: Live sentiment time series ──
    live_sentiment = pd.DataFrame()
    if not scored_df.empty:
        live_sentiment = build_sentiment_timeseries(scored_df)

    # ── Step 4: Historical proxy ──
    proxy_sentiment = pd.DataFrame()
    if eurusd_df is not None:
        proxy_sentiment = build_historical_sentiment_proxy(eurusd_df, vix_df)

    # ── Step 5: Combine ──
    # Use proxy for historical, overlay live where available
    combined = pd.DataFrame()

    if not proxy_sentiment.empty:
        combined = proxy_sentiment.copy()

    if not live_sentiment.empty:
        # Rename to avoid collision, then merge
        live_cols = {
            "sentiment_mean": "live_sentiment",
            "bull_bear_spread": "live_bull_bear",
            "sentiment_momentum": "live_momentum",
        }
        live_renamed = live_sentiment.rename(columns=live_cols)
        cols_to_add = [c for c in live_renamed.columns if c in live_cols.values()]
        if cols_to_add:
            combined = combined.join(live_renamed[cols_to_add], how="outer")

    if save and not combined.empty:
        save_dataframe(combined, "sentiment_data.csv")

    if save and not scored_df.empty:
        save_dataframe(scored_df, "scored_articles.csv")

    result = {
        "articles": scored_df,
        "live_sentiment": live_sentiment,
        "proxy_sentiment": proxy_sentiment,
        "combined": combined,
    }

    return result


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = fetch_all_sentiment()
    if not result["combined"].empty:
        print("\nCombined sentiment (last 5 rows):")
        print(result["combined"].tail())
    if not result["articles"].empty:
        print("\nSample scored articles:")
        print(result["articles"][["title", "composite_score", "sentiment_label"]].head(10))