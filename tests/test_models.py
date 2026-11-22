import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from mvdesk import models as m


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    return pd.Series(rng.standard_t(5, 1500) * 0.01, index=idx)


def test_hs_uses_only_past_data(returns):
    fc = m.historical_simulation(returns, 250)
    d = fc.index[10]
    i = returns.index.get_loc(d)
    expected = -np.quantile(returns.iloc[i - 250:i], 0.01)
    assert fc.at[d, "var99"] == pytest.approx(expected)
    shocked = returns.copy()
    shocked.loc[d] = -0.5  # a crash on day d must not move the forecast for day d
    assert m.historical_simulation(shocked, 250).at[d, "var99"] == pytest.approx(expected)


def test_hs_es_at_least_var(returns):
    fc = m.historical_simulation(returns, 250)
    assert (fc["es975"] >= fc["var975"] - 1e-12).all()
    assert (fc["var99"] >= fc["var975"]).all()


def test_ewma_recursion(returns):
    s2 = m.ewma_variance(returns, 0.94)
    t = 100
    assert s2.iloc[t] == pytest.approx(0.94 * s2.iloc[t - 1] + 0.06 * returns.iloc[t - 1] ** 2)


def test_normal_es_formula():
    out = m.normal_var_es(np.array([1.0]))
    assert out["var99"][0] == pytest.approx(2.326, abs=1e-3)
    assert out["es975"][0] == pytest.approx(2.338, abs=1e-3)  # Basel's 97.5% ES ~ 99% VaR


@pytest.mark.parametrize("nu", [4.5, 6, 10, 30])
def test_standardised_t_has_unit_variance_and_correct_es(nu):
    scale = np.sqrt((nu - 2) / nu)
    var = stats.t.var(nu) * scale**2
    assert var == pytest.approx(1.0)
    q = m.std_t_quantile(0.025, nu)
    tail, _ = integrate.quad(lambda x: x * stats.t.pdf(x / scale, nu) / scale, -np.inf, q)
    assert m.std_t_es(0.025, nu) == pytest.approx(-tail / 0.025, rel=1e-6)


def test_garch_filter_recursion():
    eps = np.array([0.1, -0.2, 0.3])
    s2 = m.garch_filter(eps, 0.01, 0.1, 0.85, 0.5)
    assert s2[1] == pytest.approx(0.01 + 0.1 * 0.01 + 0.85 * 0.5)
    assert s2[3] == pytest.approx(0.01 + 0.1 * 0.09 + 0.85 * s2[2])


def test_garch_models_cover_roughly_one_percent(returns):
    g = m.garch_forecasts(returns, fit_window=750, refit_every=63)
    r = returns.loc[g.index]
    for fc in (m.garch_t(g), m.filtered_hs(g)):
        rate = (r < -fc["var99"]).mean()
        assert 0.0 <= rate < 0.03
        assert (fc["es975"] > fc["var975"]).all()
