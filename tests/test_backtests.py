import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mvdesk import backtests as bt


def test_kupiec_matches_hand_calculation():
    hits = np.zeros(250, dtype=int)
    hits[:8] = 1  # 8 exceptions in 250 days at 1%
    n, x, p = 250, 8, 0.01
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
               - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    res = bt.kupiec_pof(hits, p)
    assert res.stat == pytest.approx(lr)
    assert res.pvalue == pytest.approx(stats.chi2.sf(lr, 1))


def test_kupiec_zero_exceptions_is_finite():
    res = bt.kupiec_pof(np.zeros(250, dtype=int), 0.01)
    assert np.isfinite(res.stat) and res.stat == pytest.approx(-2 * 250 * np.log(0.99))


def test_independence_detects_clustering():
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1
    assert bt.christoffersen_ind(spread).pvalue > 0.5
    assert bt.christoffersen_ind(clustered).pvalue < 0.001


def test_conditional_coverage_is_sum_of_parts():
    rng = np.random.default_rng(0)
    h = (rng.random(2000) < 0.02).astype(int)
    cc = bt.christoffersen_cc(h, 0.01)
    assert cc.stat == pytest.approx(bt.kupiec_pof(h, 0.01).stat + bt.christoffersen_ind(h).stat)


@pytest.mark.parametrize("n,zone,plus", [(0, "Green", 0), (4, "Green", 0), (5, "Yellow", 0.40),
                                          (9, "Yellow", 0.85), (10, "Red", 1.0), (15, "Red", 1.0)])
def test_basel_traffic_light(n, zone, plus):
    assert bt.basel_zone(n) == zone
    assert bt.basel_plus_factor(n) == plus


def test_z2_near_zero_for_correct_model_and_negative_for_understated_es():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(200_000) * 0.01)
    q = stats.norm.ppf(0.975) * 0.01
    es = 0.01 * stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025
    var = pd.Series(q, index=r.index)
    assert abs(bt.acerbi_szekely_z2(r, var, pd.Series(es, index=r.index))) < 0.05
    z_low = bt.acerbi_szekely_z2(r, var * 0.7, pd.Series(es * 0.7, index=r.index))
    assert z_low < bt.Z2_YELLOW


def test_tick_loss_is_minimised_by_true_quantile():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.standard_normal(100_000))
    true = -stats.norm.ppf(0.01)
    losses = {k: bt.tick_loss(r, pd.Series(k, index=r.index), 0.01).mean()
              for k in (true * 0.8, true, true * 1.2)}
    assert min(losses, key=losses.get) == true


def test_diebold_mariano_sign():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.random(2000))
    res = bt.diebold_mariano(a, a + 0.1)
    assert res.stat < 0 and res.pvalue < 0.001


def test_ljung_box_white_noise_vs_ar():
    rng = np.random.default_rng(4)
    e = rng.standard_normal(3000)
    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.6 * ar[t - 1] + e[t]
    assert bt.ljung_box(e).pvalue > 0.01
    assert bt.ljung_box(ar).pvalue < 1e-10
