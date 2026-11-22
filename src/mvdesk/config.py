"""Validation set-up: the book, the models in scope and the test thresholds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "prices.csv"
OUT = ROOT / "output"

# Hypothetical multi-asset book. Weights sum to 1; notional in USD.
PORTFOLIO = {
    "SPY": 0.35,  # US large-cap equity
    "EFA": 0.15,  # developed ex-US equity
    "TLT": 0.20,  # long US Treasuries
    "LQD": 0.20,  # investment-grade corporate credit
    "GLD": 0.10,  # gold
}
NOTIONAL = 100_000_000
START = "2005-01-01"
BACKTEST_START = "2008-01-02"  # leaves 3 years of history for the first estimate

VAR_LEVEL = 0.99   # Basel backtesting confidence
ES_LEVEL = 0.975   # FRTB expected-shortfall confidence
TRAFFIC_WINDOW = 250

MODEL_UNDER_REVIEW = "HS-250"
