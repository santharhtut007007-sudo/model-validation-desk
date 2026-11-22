"""One-day-ahead VaR and ES models.

Every forecast for day t uses returns up to and including t-1 only. Outputs are
positive loss numbers in return units (0.021 = a 2.1% loss), with columns
`var99`, `var975` and `es975`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

LEVELS = {"var99": 0.99, "var975": 0.975}
ES_LEVEL = 0.975


# --------------------------------------------------------------------------- historical simulation
def historical_simulation(r: pd.Series, window: int = 250) -> pd.DataFrame:
    """Plain HS: empirical quantile of the trailing `window` returns."""
    x = r.to_numpy()
    win = sliding_window_view(x, window)[:-1]  # window ending at t-1 forecasts t
    idx = r.index[window:]
    out = {k: -np.quantile(win, 1 - a, axis=1) for k, a in LEVELS.items()}
    q = -out["var975"][:, None]
    tail = np.where(win <= q, win, np.nan)
    out["es975"] = -np.nanmean(tail, axis=1)
    return pd.DataFrame(out, index=idx)


# --------------------------------------------------------------------------- EWMA / RiskMetrics
def ewma_variance(r: pd.Series, lam: float = 0.94, seed: int = 30) -> pd.Series:
    """sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}  (forecast for day t)."""
    x = r.to_numpy()
    s2 = np.empty_like(x)
    s2[0] = np.var(x[:seed])
    for t in range(1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(s2, index=r.index)


def normal_var_es(sigma: np.ndarray | pd.Series) -> dict[str, np.ndarray]:
    out = {k: stats.norm.ppf(a) * np.asarray(sigma) for k, a in LEVELS.items()}
    z = stats.norm.ppf(ES_LEVEL)
    out["es975"] = np.asarray(sigma) * stats.norm.pdf(z) / (1 - ES_LEVEL)
    return out


def ewma(r: pd.Series, lam: float = 0.94, burn: int = 250) -> pd.DataFrame:
    s = np.sqrt(ewma_variance(r, lam)).iloc[burn:]
    return pd.DataFrame(normal_var_es(s), index=s.index)


# --------------------------------------------------------------------------- GARCH(1,1)
def std_t_quantile(p: float, nu: float) -> float:
    """Quantile of a Student-t rescaled to unit variance."""
    return stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)


def std_t_es(p: float, nu: float) -> float:
    """Lower-tail expected shortfall (as a positive number) of a unit-variance Student-t."""
    q = stats.t.ppf(p, nu)
    es = (nu + q**2) / (nu - 1) * stats.t.pdf(q, nu) / p
    return es * np.sqrt((nu - 2) / nu)


def garch_filter(eps: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    """Conditional variance forecasts; element t uses eps up to t-1."""
    s2 = np.empty(len(eps) + 1)
    s2[0] = s0
    for t in range(len(eps)):
        s2[t + 1] = omega + alpha * eps[t] ** 2 + beta * s2[t]
    return s2


def garch_forecasts(
    r: pd.Series, fit_window: int = 750, refit_every: int = 21, start: int | None = None
) -> pd.DataFrame:
    """Rolling GARCH(1,1) with Student-t innovations.

    Parameters are re-estimated every `refit_every` days on the trailing `fit_window`
    returns, and held fixed in between while the variance recursion updates daily.
    Returns mu, sigma, nu and the standardised residuals of the latest fit, which the
    filtered-HS model reuses.
    """
    from arch import arch_model

    x = r.to_numpy() * 100.0  # arch is better conditioned in percent
    n = len(x)
    start = fit_window if start is None else max(start, fit_window)
    rows, resid_store = [], {}
    for t0 in range(start, n, refit_every):
        sample = x[t0 - fit_window : t0]
        res = arch_model(sample, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
            disp="off", show_warning=False
        )
        mu, omega, alpha, beta, nu = (res.params[k] for k in ("mu", "omega", "alpha[1]", "beta[1]", "nu"))
        z = (sample - mu) / res.conditional_volatility
        t1 = min(t0 + refit_every, n)
        eps = x[t0 - fit_window : t1] - mu
        s2 = garch_filter(eps, omega, alpha, beta, float(np.var(sample)))
        for t in range(t0, t1):
            rows.append((r.index[t], mu / 100, np.sqrt(s2[t - (t0 - fit_window)]) / 100, nu, t0))
        resid_store[t0] = z
    df = pd.DataFrame(rows, columns=["date", "mu", "sigma", "nu", "fit_id"]).set_index("date")
    df.attrs["resid"] = resid_store
    return df


def garch_t(g: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k, a in LEVELS.items():
        q = np.array([std_t_quantile(1 - a, nu) for nu in g["nu"]])
        out[k] = -(g["mu"].to_numpy() + g["sigma"].to_numpy() * q)
    e = np.array([std_t_es(1 - ES_LEVEL, nu) for nu in g["nu"]])
    out["es975"] = -g["mu"].to_numpy() + g["sigma"].to_numpy() * e
    return pd.DataFrame(out, index=g.index)


def filtered_hs(g: pd.DataFrame) -> pd.DataFrame:
    """Filtered HS: GARCH volatility times the empirical quantile of standardised residuals."""
    resid = g.attrs["resid"]
    cache = {}
    for fid, z in resid.items():
        q975 = np.quantile(z, 1 - ES_LEVEL)
        cache[fid] = (np.quantile(z, 0.01), q975, z[z <= q975].mean())
    zq = np.array([cache[f] for f in g["fit_id"]])
    mu, s = g["mu"].to_numpy(), g["sigma"].to_numpy()
    return pd.DataFrame(
        {"var99": -(mu + s * zq[:, 0]), "var975": -(mu + s * zq[:, 1]), "es975": -(mu + s * zq[:, 2])},
        index=g.index,
    )


# --------------------------------------------------------------------------- registry
MODEL_INFO = {
    "HS-250": {
        "role": "Model under review",
        "method": "Historical simulation, 250-day equally weighted window",
        "assumption": "The last year of returns is a fair sample of tomorrow's distribution",
    },
    "EWMA-0.94": {
        "role": "Challenger",
        "method": "RiskMetrics EWMA volatility (lambda = 0.94), normal quantile",
        "assumption": "Returns are conditionally normal with exponentially decaying memory",
    },
    "GARCH-t": {
        "role": "Challenger",
        "method": "GARCH(1,1), Student-t innovations, 750-day window, monthly re-fit",
        "assumption": "Volatility clusters and mean-reverts; tails are fat but symmetric",
    },
    "FHS": {
        "role": "Challenger",
        "method": "Filtered HS: GARCH volatility x empirical quantile of standardised residuals",
        "assumption": "Standardised residuals are i.i.d.; no distributional shape is imposed",
    },
}


def run_all(r: pd.Series, backtest_start: str) -> dict[str, pd.DataFrame]:
    """Forecasts for every model, aligned to the common backtest period."""
    start_i = int(r.index.searchsorted(pd.Timestamp(backtest_start)))
    g = garch_forecasts(r, start=start_i)
    models = {
        "HS-250": historical_simulation(r, 250),
        "EWMA-0.94": ewma(r, 0.94),
        "GARCH-t": garch_t(g),
        "FHS": filtered_hs(g),
    }
    common = None
    for v in models.values():
        idx = v.index[v.index >= pd.Timestamp(backtest_start)]
        common = idx if common is None else common.intersection(idx)
    return {k: v.loc[common] for k, v in models.items()}
