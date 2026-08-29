import numpy as np
import pytest

from buyrent import (History, Scenario, breakeven_growth, breakeven_growth_real,
                     run, simulate, valuation_dev)
from buyrent.bootstrap import country_block_ci
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
    mc = run(s, hist, n_boot=200, seed=1)
    assert 0.0 <= mc["p_buy_wins"] <= 1.0
    assert mc["gap_real_pct_of_price"]["p5"] < mc["gap_real_pct_of_price"]["p95"]

    # 租金收益率更高(更便宜)的城市, 买房胜率应更高
    s_cheap = Scenario(rent_yield=0.05, mort_rate=0.045, hold=10)
    mc_cheap = run(s_cheap, hist, n_boot=200, seed=1)
    assert mc_cheap["p_buy_wins"] > mc["p_buy_wins"]


def test_montecarlo_is_exhaustive_not_sampled():
    """全量求值: 胜率必须可由窗口池精确复算, 且与自助次数无关。"""
    hist = History()
    s = Scenario(rent_yield=0.03, hold=10)
    a = run(s, hist, n_boot=0, seed=1)
    b = run(s, hist, n_boot=500, seed=7)
    assert a["p_buy_wins"] == b["p_buy_wins"]        # 无蒙特卡洛噪声
    assert "p_buy_wins_ci95" not in a                # n_boot=0 时不算区间
    assert a["n_windows"] == len(hist.pool(10))
    assert a["p_buy_wins"] * a["n_windows"] == pytest.approx(
        round(a["p_buy_wins"] * a["n_windows"]))     # 胜率是个分数


def test_country_block_ci_wider_than_naive():
    """整群自助的区间必须明显宽于按窗口 iid 抽样——这正是重叠窗口的代价。"""
    hist = History()
    sub = hist.pool(10)
    g = sub.g_house.values
    block = country_block_ci(sub.iso.values, g, lambda a: (a >= 0.02).mean(),
                             n_boot=600, seed=0)
    # 打散国家标签 = 假装窗口独立
    rng = np.random.default_rng(0)
    naive = country_block_ci(rng.permutation(sub.iso.values), g,
                             lambda a: (a >= 0.02).mean(), n_boot=600, seed=0)
    assert (block[1] - block[0]) > 1.5 * (naive[1] - naive[0])


def test_ci_brackets_point_estimate():
    hist = History()
    for terc in (None, 0, 2):
        p, _ = hist.prob_exceed(0.02, 10, terc)
        lo, hi = hist.prob_exceed_ci(0.02, 10, terc, n_boot=400)
        assert lo <= p <= hi


def test_tercile_conditioning_is_weak_but_signed():
    """高估组下跌概率更高, 但差异的95%区间含0——论文必须按这个强度表述。"""
    import json
    from pathlib import Path
    u = json.load(open(Path(__file__).resolve().parents[1]
                       / "data" / "derived" / "uncertainty.json"))
    c = u["g10_by_tercile_ci"]["contrast_dear_minus_cheap"]["p_le_0"]
    assert c["point"] > 0                    # 方向: 买贵了后面更容易跌
    assert c["ci95"][0] < 0 < c["ci95"][1]   # 但强度: 区间跨0
    assert 0.01 < c["p_le_0"] < 0.10         # 单边 p 在 1%–10% 之间


def test_sample_period_switch():
    """论文局限第1条承诺可切换样本期——必须真的能切。"""
    full, post = History(), History(since=1950)
    assert len(post.win) < len(full.win)
    assert post.win.year0.min() >= 1950
    # 三分位切点不随样本期变化: "相对自己历史贵不贵"是固定标准
    assert post.cutoffs == full.cutoffs
    # 战后窗口的房价涨幅整体更高
    assert post.prob_exceed(0.02, 10)[0] > full.prob_exceed(0.02, 10)[0]
    with pytest.raises(ValueError):
        History(since=2100)


def test_idio_offsets_scale_with_horizon():
    """特异性冲击折到年化涨幅上时按 1/√T 摊薄; 关闭时不改变任何结果。"""
    from buyrent.montecarlo import idio_offsets
    assert idio_offsets(0.0, 10).tolist() == [0.0]
    a, b = idio_offsets(0.10, 5), idio_offsets(0.10, 20)
    assert a.std() > b.std()                       # 持有越久摊得越薄
    assert a.mean() == pytest.approx(0, abs=1e-12)  # 均值为零, 不平移中位数
    assert len(a) == 9


def test_idio_risk_raises_win_rate_but_not_median():
    """特异性风险不利好租房: 它把胜率拉向五五开, 中位数几乎不动、左尾变厚。

    这是论文第9节第5条的实证依据——也是不能只看胜率的理由。
    """
    hist = History()
    s = Scenario(rent_yield=0.018, mort_rate=0.040, hold=10)
    base = run(s, hist, tercile=2, idio_sd=0.0, n_boot=0)
    wide = run(s, hist, tercile=2, idio_sd=0.15, n_boot=0)
    assert base["p_buy_wins"] < 0.5
    assert wide["p_buy_wins"] > base["p_buy_wins"] + 0.05   # 胜率被拉高
    assert abs(wide["gap_real_pct_of_price"]["median"]
               - base["gap_real_pct_of_price"]["median"]) < 0.02  # 中位数几乎不动
    assert wide["gap_real_pct_of_price"]["p5"] < base["gap_real_pct_of_price"]["p5"]
