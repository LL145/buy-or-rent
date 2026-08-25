import math

import pytest

from buyrent import (History, Scenario, breakeven_growth, breakeven_growth_real,
                     run, simulate, valuation_dev)
from buyrent.model import annuity_payment, loan_balance, with_growth


def cash_scenario(**kw):
    """全款、零成本的干净场景, 便于解析验证。"""
    base = dict(down=1.0, buy_cost=0.0, sell_cost=0.0, carry=0.0,
                rent_yield=0.03, g_rent=0.0, g_house=0.02, r_invest=0.05, hold=1)
    base.update(kw)
    return Scenario(**base)


def test_one_year_cash_analytic():
    # T=1 全款零成本: 买方期末 = P(1+g) + 租金(1+r); 租方期末 = P(1+r)
    s = cash_scenario()
    out = simulate(s)
    p = s.price
    assert out["wealth_buy"] == pytest.approx(
        p * (1 + s.g_house) + s.rent_yield * p * (1 + s.r_invest))
    assert out["wealth_rent"] == pytest.approx(p * (1 + s.r_invest))


def test_breakeven_matches_model():
    s = Scenario(rent_yield=0.02, hold=10)
    g = breakeven_growth(s)
    assert abs(simulate(with_growth(s, g))["gap"]) < 1e-4


def test_gap_monotone_in_growth():
    s = Scenario(hold=10)
    gaps = [simulate(with_growth(s, g))["gap"] for g in (-0.02, 0.0, 0.02, 0.05)]
    assert gaps == sorted(gaps)


def test_higher_invest_return_favors_renting():
    s1 = simulate(Scenario(hold=10, r_invest=0.04))
    s2 = simulate(Scenario(hold=10, r_invest=0.08))
    assert s2["gap"] < s1["gap"]


def test_short_hold_transaction_costs_kill_buying():
    # 持有3年、买卖成本合计7%: 即使房价与投资同速, 租房应胜
    s = Scenario(hold=3, g_house=0.05, r_invest=0.05, g_rent=0.05,
                 rent_yield=0.025)
    assert simulate(s)["gap"] < 0


def test_amortization():
    pay = annuity_payment(100.0, 0.05, 30)
    bal = 100.0
    for _ in range(30):
        bal = bal * 1.05 - pay
    assert bal == pytest.approx(0, abs=1e-9)
    assert loan_balance(100.0, 0.05, 30, 30) == 0
    assert loan_balance(100.0, 0.05, 30, 10) == pytest.approx(
        100 * 1.05 ** 10 - pay * (1.05 ** 10 - 1) / 0.05)


def test_valuation_dev_sign():
    # 当前租金收益率低于历史常态 → 现在更贵 → 偏离为正
    assert valuation_dev(0.015, 0.025) > 0
    assert valuation_dev(0.030, 0.025) < 0


def test_history_and_montecarlo():
    hist = History()
    p2, n = hist.prob_exceed(0.02, 10)
    p4, _ = hist.prob_exceed(0.04, 10)
    assert n > 1000 and 0 < p4 < p2 < 1
    # 高估组的条件概率应低于低估组
    assert hist.prob_exceed(0.02, 10, 2)[0] < hist.prob_exceed(0.02, 10, 0)[0]

    s = Scenario(rent_yield=0.02, mort_rate=0.045, hold=10)
    mc = run(s, hist, n=400, seed=1)
    assert 0.0 <= mc["p_buy_wins"] <= 1.0
    assert mc["gap_real_pct_of_price"]["p5"] < mc["gap_real_pct_of_price"]["p95"]

    # 租金收益率更高(更便宜)的城市, 买房胜率应更高
    s_cheap = Scenario(rent_yield=0.05, mort_rate=0.045, hold=10)
    mc_cheap = run(s_cheap, hist, n=400, seed=1)
    assert mc_cheap["p_buy_wins"] > mc["p_buy_wins"]
