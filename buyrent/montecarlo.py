"""蒙特卡洛: 历史窗口自助法。

不凭空假设分布——直接从 1870 年以来 18 国的真实历史窗口中抽样
(真实房价涨幅, 真实租金涨幅, 股/债真实回报, 通胀) 的联合实现值,
天然保留相关结构(通胀↑→名义租金/房价↑而月供不变 = 业主的通胀对冲)。

简化: 窗口内取年化常数速率(丢失路径内波动, 保留联合均值), 论文中说明。
"""
from dataclasses import replace

import numpy as np

from .history import History
from .model import Scenario, simulate


def nominal(real: float, infl: float) -> float:
    return (1 + real) * (1 + infl) - 1


def run(s: Scenario, hist: History, n: int = 5000, tercile: int | None = None,
        eq_share: float = 0.6, seed: int = 0) -> dict:
    """返回 P(买优于租) 及期末财富差(占房价%, 真实口径)的分布。

    eq_share: 租方替代资产中股票占比, 其余为长期国债(同窗口, 保留相关性)。
    tercile:  当前城市的估值三分位(None = 不加条件, 用全部历史)。
    """
    rng = np.random.default_rng(seed)
    w = hist.sample_windows(s.hold, n, rng, tercile)
    gaps = np.empty(n)
    wins = 0
    for i, row in w.iterrows():
        r_mix = eq_share * row.r_eq + (1 - eq_share) * row.r_bond
        si = replace(
            s,
            g_house=nominal(row.g_house, row.infl),
            g_rent=nominal(row.g_rent, row.infl),
            r_invest=nominal(r_mix, row.infl),
            infl=row.infl,
        )
        out = simulate(si)
        gaps[i] = out["gap_real"] / s.price
        wins += out["gap"] > 0
    q = np.percentile(gaps, [5, 25, 50, 75, 95])
    return {
        "p_buy_wins": wins / n,
        "gap_real_pct_of_price": {
            "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]},
        "n": n,
        "n_windows": int(len(hist._slice(s.hold, tercile).dropna(
            subset=["g_house", "g_rent", "r_eq", "r_bond", "infl"]))),
    }
