"""
Mock data generator for AI Disclosure → Trading Lag Predictor.

Generates realistic DataFrames matching the datathon CSV schemas.
Stock prices use geometric Brownian motion. Events inject small return
bumps at random lags (5-40 days) so the model has a learnable signal.

When real CSVs exist in data/, data_loader() switches automatically.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
    "NFLX", "CRM", "INTC", "CSCO", "PEP", "COST", "AVGO", "TXN", "QCOM",
    "AMD", "ORCL", "IBM", "GE", "CAT", "BA", "MMM", "HON", "LMT", "GD",
    "RTX", "DE", "UPS", "FDX", "LOW", "TGT", "MCD", "SBUX", "NKE",
    "CVX", "XOM", "COP", "SLB",
]

SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "META": "Technology", "NVDA": "Technology",
    "TSLA": "Consumer Discretionary", "JPM": "Financials", "V": "Financials",
    "JNJ": "Healthcare", "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "UNH": "Healthcare", "HD": "Consumer Discretionary", "MA": "Financials",
    "DIS": "Communication Services", "PYPL": "Financials", "ADBE": "Technology",
    "NFLX": "Communication Services", "CRM": "Technology", "INTC": "Technology",
    "CSCO": "Technology", "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "AVGO": "Technology", "TXN": "Technology", "QCOM": "Technology",
    "AMD": "Technology", "ORCL": "Technology", "IBM": "Technology",
    "GE": "Industrials", "CAT": "Industrials", "BA": "Industrials",
    "MMM": "Industrials", "HON": "Industrials", "LMT": "Industrials",
    "GD": "Industrials", "RTX": "Industrials", "DE": "Industrials",
    "UPS": "Industrials", "FDX": "Industrials", "LOW": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "CVX": "Energy", "XOM": "Energy", "COP": "Energy", "SLB": "Energy",
}

INDUSTRIES = {
    "Technology": ["Software", "Semiconductors", "IT Services", "Hardware"],
    "Financials": ["Banks", "Payments", "Insurance"],
    "Healthcare": ["Pharma", "Health Insurance", "Medical Devices"],
    "Consumer Discretionary": ["Retail", "Auto", "Restaurants", "Apparel"],
    "Consumer Staples": ["Food & Beverage", "Household Products"],
    "Industrials": ["Aerospace & Defense", "Machinery", "Transportation"],
    "Communication Services": ["Media", "Entertainment", "Telecom"],
    "Energy": ["Oil & Gas", "Oilfield Services"],
}

USE_CASES = [
    "Customer Service Chatbot", "Code Generation", "Content Creation",
    "Predictive Analytics", "Document Processing", "Search Enhancement",
    "Recommendation Engine", "Fraud Detection", "Supply Chain Optimization",
    "Drug Discovery",
]

AGENT_TYPES = ["Chatbot", "Copilot", "Autonomous Agent", "Analytics", "Other"]

AI_VENDORS = ["OpenAI", "Google", "Microsoft", "Anthropic", "AWS", "Meta AI", None]
AI_MODELS = ["GPT-4", "Gemini", "Claude", "Llama", "DALL-E", "Copilot", None]

GENAI_RELEASES = [
    ("GPT-3.5", "2022-11-30", "OpenAI"),
    ("GPT-4", "2023-03-14", "OpenAI"),
    ("GPT-4 Turbo", "2023-11-06", "OpenAI"),
    ("GPT-4o", "2024-05-13", "OpenAI"),
    ("Claude 2", "2023-07-11", "Anthropic"),
    ("Claude 3", "2024-03-04", "Anthropic"),
    ("Gemini 1.0", "2023-12-06", "Google"),
    ("Gemini 1.5", "2024-02-15", "Google"),
    ("Llama 2", "2023-07-18", "Meta"),
    ("Llama 3", "2024-04-18", "Meta"),
    ("Mistral Large", "2024-02-26", "Mistral"),
    ("DALL-E 3", "2023-10-03", "OpenAI"),
    ("Sora", "2024-02-15", "OpenAI"),
    ("Stable Diffusion 3", "2024-02-22", "Stability AI"),
    ("Copilot", "2023-03-16", "Microsoft"),
    ("Gemini 2.0", "2024-12-11", "Google"),
    ("Claude 3.5 Sonnet", "2024-06-20", "Anthropic"),
    ("o1", "2024-09-12", "OpenAI"),
    ("o1 Pro", "2024-12-05", "OpenAI"),
    ("DeepSeek-R1", "2025-01-20", "DeepSeek"),
    ("Grok-2", "2024-08-13", "xAI"),
    ("Claude 4 Sonnet", "2025-05-22", "Anthropic"),
    ("GPT-4.1", "2025-04-14", "OpenAI"),
    ("Llama 4", "2025-04-05", "Meta"),
    ("Gemini 2.5 Pro", "2025-03-25", "Google"),
    ("Claude 4.5 Sonnet", "2025-10-22", "Anthropic"),
    ("GPT-5", "2025-06-10", "OpenAI"),
]

SP_INDICES = ["S&P 500", "S&P 400"]


# ---------------------------------------------------------------------------
# Geometric Brownian Motion price simulator
# ---------------------------------------------------------------------------
def _simulate_gbm(n_days: int, mu: float = 0.0003, sigma: float = 0.02,
                  s0: float = 100.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate daily close prices via geometric Brownian motion."""
    if rng is None:
        rng = np.random.default_rng()
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * rng.standard_normal(n_days))
    prices = s0 * np.cumprod(daily_returns)
    return prices


# ---------------------------------------------------------------------------
# Mock generators
# ---------------------------------------------------------------------------
def generate_trading_dates(start: str = "2015-01-02", end: str = "2025-06-30") -> pd.DatetimeIndex:
    """Generate business-day trading calendar."""
    return pd.bdate_range(start, end)


def generate_stock_prices(tickers: list[str] | None = None,
                          seed: int = 42) -> pd.DataFrame:
    """Generate ~50 tickers × ~2500 days of OHLCV data via GBM."""
    rng = np.random.default_rng(seed)
    tickers = tickers or TICKERS
    dates = generate_trading_dates()
    rows = []
    for ticker in tickers:
        s0 = rng.uniform(30, 500)
        mu = rng.uniform(0.0001, 0.0005)
        sigma = rng.uniform(0.005, 0.012)
        close = _simulate_gbm(len(dates), mu=mu, sigma=sigma, s0=s0, rng=rng)
        high = close * (1 + rng.uniform(0, 0.02, len(dates)))
        low = close * (1 - rng.uniform(0, 0.02, len(dates)))
        opn = low + (high - low) * rng.uniform(0.3, 0.7, len(dates))
        volume = rng.integers(500_000, 50_000_000, len(dates))
        for i, dt in enumerate(dates):
            rows.append({
                "Date": dt,
                "Open": round(opn[i], 2),
                "High": round(high[i], 2),
                "Low": round(low[i], 2),
                "Close": round(close[i], 2),
                "Adj_Close": round(close[i], 2),
                "Volume": int(volume[i]),
                "ticker": ticker,
            })
    return pd.DataFrame(rows)


def generate_spy_prices(seed: int = 42) -> pd.DataFrame:
    """Generate SPY benchmark prices."""
    rng = np.random.default_rng(seed + 100)
    dates = generate_trading_dates()
    close = _simulate_gbm(len(dates), mu=0.0003, sigma=0.01, s0=250, rng=rng)
    return pd.DataFrame({"Date": dates, "Adj_Close": np.round(close, 2)})


def generate_events(stock_prices_df: pd.DataFrame,
                    n_events: int = 200, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Generate AI adoption events with injected return bumps.

    Returns (events_df, injection_map) where injection_map stores the
    true peak lag per event for validation.
    """
    rng = np.random.default_rng(seed + 200)
    tickers = stock_prices_df["ticker"].unique()
    dates = sorted(stock_prices_df["Date"].unique())

    # Pick event dates that leave room for a 60-day post-event window
    # Concentrate events in 2023-2025 (post-GenAI era) for realistic data
    genai_start = pd.Timestamp("2023-01-01")
    eligible_start = max(dates[60], genai_start)
    eligible_end = dates[-70]
    eligible_dates = [d for d in dates if eligible_start <= d <= eligible_end]

    events = []
    injection_map = {}

    # Make true lag depend on features so the model has a learnable signal
    _use_case_lag = {uc: int(rng.integers(8, 35)) for uc in USE_CASES}
    _sector_lag_offset = {s: int(rng.integers(-5, 6)) for s in set(SECTORS.values())}
    _agent_lag_offset = {a: int(rng.integers(-3, 4)) for a in AGENT_TYPES}

    for i in range(n_events):
        ticker = rng.choice(tickers)
        ann_date = pd.Timestamp(rng.choice(eligible_dates))
        use_case = rng.choice(USE_CASES)
        agent_type = rng.choice(AGENT_TYPES)
        vendor = rng.choice(AI_VENDORS)
        model = rng.choice(AI_MODELS)
        sector = SECTORS.get(ticker, "Other")
        sp_index = rng.choice(SP_INDICES)

        # True lag depends systematically on features (learnable signal)
        base_lag = _use_case_lag[use_case]
        sector_offset = _sector_lag_offset.get(sector, 0)
        agent_offset = _agent_lag_offset[agent_type]
        vendor_offset = -3 if vendor else 2  # named vendor → faster reaction
        noise = int(rng.integers(-3, 4))
        true_lag = int(np.clip(base_lag + sector_offset + agent_offset + vendor_offset + noise, 5, 40))
        injection_map[i] = {"ticker": ticker, "date": ann_date, "true_lag": true_lag}

        events.append({
            "ticker": ticker,
            "announcement_date": ann_date,
            "use_case": use_case,
            "agent_type": agent_type,
            "ai_vendor": vendor if vendor else np.nan,
            "ai_model": model if model else np.nan,
            "industry": sector,
            "sp_index": sp_index,
        })

    events_df = pd.DataFrame(events)

    # Inject return bumps into stock prices at the true lag
    _inject_return_bumps(stock_prices_df, injection_map, rng)

    return events_df, injection_map


def _inject_return_bumps(stock_prices_df: pd.DataFrame,
                         injection_map: dict,
                         rng: np.random.Generator) -> None:
    """Inject small positive abnormal returns around the true peak lag day."""
    stock_prices_df.sort_values(["ticker", "Date"], inplace=True)
    stock_prices_df.reset_index(drop=True, inplace=True)

    for _event_id, info in injection_map.items():
        ticker = info["ticker"]
        ann_date = info["date"]
        true_lag = info["true_lag"]

        mask = stock_prices_df["ticker"] == ticker
        ticker_df = stock_prices_df.loc[mask].copy()
        ticker_dates = ticker_df["Date"].values

        # Find the announcement date index (convert to numpy datetime64 for searchsorted)
        date_idx = np.searchsorted(ticker_dates, np.datetime64(ann_date))
        if date_idx >= len(ticker_dates):
            continue

        # Inject a strong bump over a few days around the true lag
        bump_center = date_idx + true_lag
        for offset in range(-2, 3):
            idx = bump_center + offset
            if 0 <= idx < len(ticker_dates):
                # Strong bump at center, tapering at edges
                bump = rng.uniform(0.06, 0.12) * max(0.2, 1 - abs(offset) * 0.3)
                row_idx = ticker_df.index[idx]
                for col in ["Close", "Adj_Close", "High"]:
                    stock_prices_df.loc[row_idx, col] *= (1 + bump)


def generate_edgar_capex(tickers: list[str] | None = None,
                         seed: int = 42) -> pd.DataFrame:
    """Generate mock EDGAR CapEx / R&D fundamentals data."""
    rng = np.random.default_rng(seed + 300)
    tickers = tickers or TICKERS
    rows = []
    concepts = ["CapitalExpenditures", "ResearchAndDevelopmentExpense"]
    for ticker in tickers:
        for year in range(2015, 2026):
            for q in range(1, 5):
                for concept in concepts:
                    base_val = rng.uniform(50_000_000, 5_000_000_000)
                    growth = rng.uniform(-0.1, 0.3)
                    value = base_val * (1 + growth)
                    end_month = q * 3
                    end_date = pd.Timestamp(year, min(end_month, 12), 28)
                    rows.append({
                        "ticker": ticker,
                        "concept": concept,
                        "value": round(value, 2),
                        "end_date": end_date,
                        "fiscal_year": year,
                        "fiscal_period": f"Q{q}",
                    })
    return pd.DataFrame(rows)


def generate_ticker_dim(tickers: list[str] | None = None,
                        seed: int = 42) -> pd.DataFrame:
    """Generate ticker dimension table."""
    rng = np.random.default_rng(seed + 400)
    tickers = tickers or TICKERS
    rows = []
    for ticker in tickers:
        sector = SECTORS.get(ticker, "Other")
        industry_options = INDUSTRIES.get(sector, ["Other"])
        industry = rng.choice(industry_options)
        rows.append({"ticker": ticker, "sector": sector, "industry": industry})
    return pd.DataFrame(rows)


def generate_genai_releases() -> pd.DataFrame:
    """Generate GenAI model releases dimension table."""
    rows = []
    for name, date, company in GENAI_RELEASES:
        rows.append({
            "model_name": name,
            "release_date": pd.Timestamp(date),
            "company": company,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loader — auto-detect real CSVs or fall back to mock
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Map of logical name → possible CSV filenames
_CSV_MAP = {
    "events": ["enterprise_ai_adoption_internet_events.csv"],
    "stock_prices": [
        "sp500_prices_all_since_2015.csv",
        "sp400_prices_all_since_2015.csv",
    ],
    "spy_prices": ["etfs_prices_all_since_2015.csv"],
    "edgar_capex": ["sp_edgar_fundamentals.csv"],
    "ticker_dim": ["ticker_dimension.csv", "ticker_dimens.csv"],
    "genai_releases": ["genai_dimension.csv"],
}


def _find_csv(name: str) -> Path | None:
    """Find the first existing CSV for a logical dataset name."""
    for fname in _CSV_MAP.get(name, []):
        p = DATA_DIR / fname
        if p.exists():
            return p
    return None


def data_loader(seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Load data — real CSVs if available, otherwise mock.

    Returns dict with keys:
        events, stock_prices, spy_prices, edgar_capex, ticker_dim, genai_releases
    Also includes 'injection_map' (None when using real data).
    """
    result = {}
    using_real = {}

    # --- Stock prices ---
    csv_path = _find_csv("stock_prices")
    if csv_path:
        stock_prices = pd.read_csv(csv_path, parse_dates=["Date"])
        # If multiple price files, concat them
        for fname in _CSV_MAP["stock_prices"]:
            p = DATA_DIR / fname
            if p.exists() and p != csv_path:
                extra = pd.read_csv(p, parse_dates=["Date"])
                stock_prices = pd.concat([stock_prices, extra], ignore_index=True)
        # Standardize column names
        if "Adj Close" in stock_prices.columns:
            stock_prices.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
        using_real["stock_prices"] = True
    else:
        stock_prices = generate_stock_prices(seed=seed)
        using_real["stock_prices"] = False
    result["stock_prices"] = stock_prices

    # --- SPY prices ---
    csv_path = _find_csv("spy_prices")
    if csv_path:
        spy_all = pd.read_csv(csv_path, parse_dates=["Date"])
        if "Adj Close" in spy_all.columns:
            spy_all.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
        # Filter to SPY only
        if "ticker" in spy_all.columns:
            spy_prices = spy_all[spy_all["ticker"] == "SPY"][["Date", "Adj_Close"]].copy()
        else:
            spy_prices = spy_all[["Date", "Adj_Close"]].copy()
        using_real["spy_prices"] = True
    else:
        spy_prices = generate_spy_prices(seed=seed)
        using_real["spy_prices"] = False
    result["spy_prices"] = spy_prices

    # --- Events ---
    csv_path = _find_csv("events")
    injection_map = None
    if csv_path:
        events = pd.read_csv(csv_path, parse_dates=["announcement_date"])
        using_real["events"] = True
    else:
        events, injection_map = generate_events(stock_prices, seed=seed)
        using_real["events"] = False
    result["events"] = events
    result["injection_map"] = injection_map

    # --- EDGAR ---
    csv_path = _find_csv("edgar_capex")
    if csv_path:
        edgar = pd.read_csv(csv_path, parse_dates=["end_date"])
        using_real["edgar_capex"] = True
    else:
        edgar = generate_edgar_capex(seed=seed)
        using_real["edgar_capex"] = False
    result["edgar_capex"] = edgar

    # --- Ticker dimension ---
    csv_path = _find_csv("ticker_dim")
    if csv_path:
        ticker_dim = pd.read_csv(csv_path)
        using_real["ticker_dim"] = True
    else:
        ticker_dim = generate_ticker_dim(seed=seed)
        using_real["ticker_dim"] = False
    result["ticker_dim"] = ticker_dim

    # --- GenAI releases ---
    csv_path = _find_csv("genai_releases")
    if csv_path:
        genai = pd.read_csv(csv_path, parse_dates=["release_date"])
        using_real["genai_releases"] = True
    else:
        genai = generate_genai_releases()
        using_real["genai_releases"] = False
    result["genai_releases"] = genai

    result["_using_real"] = using_real
    return result
