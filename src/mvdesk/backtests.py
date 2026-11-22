"""Outcome analysis: the statistical tests a validator runs on VaR and ES forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Basel Committee (1996) traffic light for 250 observations at 99%.
BASEL_PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

# Acerbi & Szekely (2014) Z2 traffic light for 97.5% ES (approximately
# distribution-free): yellow below -0.70 (5% level), red below -1.80 (0.01% level).
Z2_YELLOW, Z2_RED = -0.70, -1.80


def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    """1 when the realised loss is larger than the VaR forecast."""
    return (r < -var).astype(int)


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


@dataclass
class TestResult:
    stat: float
    pvalue: float

    def verdict(self, alpha: float = 0.05) -> str:
        return "Reject" if self.pvalue < alpha else "Pass"


def kupiec_pof(hits: np.ndarray, p: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures likelihood ratio, chi2(1)."""
    n, x = len(hits), int(np.sum(hits))
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen_ind(hits: np.ndarray) -> TestResult:
    """Christoffersen (1998) independence test: is an exception more likely after an exception?"""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen_cc(hits: np.ndarray, p: float) -> TestResult:
    """Conditional coverage = POF + independence, chi2(2)."""
    lr = kupiec_pof(hits, p).stat + christoffersen_ind(hits).stat
    return TestResult(lr, float(stats.chi2.sf(lr, 2)))


def basel_zone(n_exceptions: int) -> str:
    if n_exceptions <= 4:
        return "Green"
    if n_exceptions <= 9:
        return "Yellow"
    return "Red"


def basel_plus_factor(n_exceptions: int) -> float:
    if n_exceptions <= 4:
        return 0.0
    return BASEL_PLUS_FACTOR.get(n_exceptions, 1.0)


def rolling_exceptions(hits: pd.Series, window: int = 250) -> pd.Series:
    return hits.rolling(window).sum().dropna().astype(int)


def acerbi_szekely_z2(r: pd.Series, var975: pd.Series, es975: pd.Series, alpha: float = 0.025) -> float:
    """Z2 = sum(X_t I_t / (T alpha ES_t)) + 1. Expected value 0; negative means ES is too low."""
    hits = r < -var975
    return float((r[hits] / es975[hits]).sum() / (len(r) * alpha) + 1)


def z2_zone(z2: float) -> str:
    if z2 < Z2_RED:
        return "Red"
    if z2 < Z2_YELLOW:
        return "Yellow"
    return "Green"


def tick_loss(r: pd.Series, var: pd.Series, p: float) -> pd.Series:
    """Quantile (pinball) loss of the VaR forecast; the scoring rule VaR is elicitable under."""
    q = -var
    return ((r < q).astype(float) - p) * (q - r)


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lags: int | None = None) -> TestResult:
    """DM test on loss differential a - b with Newey-West variance. Negative stat: a is better."""
    d = (loss_a - loss_b).to_numpy()
    n = len(d)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return TestResult(float(stat), float(2 * stats.norm.sf(abs(stat))))


def ljung_box(x: np.ndarray, lags: int = 10) -> TestResult:
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    denom = x @ x
    acf = np.array([x[k:] @ x[:-k] / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    return TestResult(float(q), float(stats.chi2.sf(q, lags)))


def jarque_bera(x: np.ndarray) -> TestResult:
    res = stats.jarque_bera(x)
    return TestResult(float(res.statistic), float(res.pvalue))
