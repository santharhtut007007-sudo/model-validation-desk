"""Market data: adjusted daily closes from Yahoo Finance, cached to CSV.

The portfolio is a static, daily-rebalanced multi-asset book. Daily P&L is
hypothetical P&L (positions held fixed for one day), the quantity regulators
use for "clean" VaR backtesting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CACHE, PORTFOLIO, START


def load_prices(refresh: bool = False, cache: Path = CACHE) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker, business-day index."""
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf

    tickers = list(PORTFOLIO)
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
    px = raw[tickers].dropna(how="any")
    if px.empty:
        raise RuntimeError("Yahoo Finance returned no prices; check the connection")
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(cache)
    return px


def portfolio_pnl(prices: pd.DataFrame, weights: dict[str, float], notional: float) -> pd.DataFrame:
    """Asset returns plus portfolio return and dollar P&L for a daily-rebalanced book."""
    rets = prices[list(weights)].pct_change().dropna()
    w = pd.Series(weights)
    if abs(w.sum() - 1) > 1e-9:
        raise ValueError(f"weights sum to {w.sum():.4f}, expected 1")
    out = rets.copy()
    out["port_ret"] = rets @ w
    out["pnl"] = out["port_ret"] * notional
    return out
