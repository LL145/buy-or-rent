"""历史窗口全量求值 + 以国家为整群的自助置信区间。

不凭空假设分布——直接用 1870 年以来 18 国的真实历史窗口
(真实房价涨幅, 真实租金涨幅, 股/债真实回报, 通胀) 的联合实现值,
天然保留相关结构(通胀↑→名义租金/房价↑而月供不变 = 业主的通胀对冲)。

两点方法约定:

1. **全量而非抽样。** 同持有期(可选同估值组)的窗口池只有数百到一千余个,
   逐个求值即得精确的 P(买优于租); 有放回抽 n 次只会给同一个数加一层
   蒙特卡洛噪声。
2. **不确定性以国家为整群。** 滚动窗口高度重叠(相邻10年窗口共享9年), 且各国
   受同一轮全球周期驱动, 独立样本量远小于窗口数。把整个国家作为 block 有放回
   抽样, 同时吸收窗口重叠与国别内相关, 给出 P(买优于租) 的 95% 区间。
   区间宽度本身是重要输出: 它说明这个胜率能被当作几位有效数字来读。

简化: 窗口内取年化常数速率(丢失路径内波动, 保留联合均值), 论文第9节说明。
"""
from dataclasses import replace

import numpy as np

from .bootstrap import N_BOOT, country_block_ci
from .history import History
from .model import Scenario, simulate


def nominal(real: float, infl: float) -> float:
    return (1 + real) * (1 + infl) - 1


def run(s: Scenario, hist: History, tercile: int | None = None,
        eq_share: float = 0.6, r_invest_real: float | None = None,
        infl_fixed: float | None = None, n_boot: int = N_BOOT,
        seed: int = 0) -> dict:
    """返回 P(买优于租)(含95%区间) 及期末财富差(占房价%, 真实口径)的分布。

    eq_share:       租方替代资产中股票占比, 其余为长期国债(同窗口, 保留相关性)。
    tercile:        当前城市的估值三分位(None = 不加条件, 用全部历史)。
    r_invest_real:  若给定, 租方投资收益率不再取自历史窗口, 而是固定为该真实
                    收益率(按窗口通胀换算名义)。用于可投资渠道受限的市场
                    (如中国居民: 存款/国债/理财的真实收益率 1–3%)。
    infl_fixed:     若给定, 窗口通胀被替换为该固定值(真实量的联合取值不变)。
                    用于浮动利率按揭的市场(如中国 LPR 定价): 名义月供锁定 +
                    历史通胀会凭空授予买方一份固定利率式的通胀对冲, 浮动利率下
                    应关闭该通道——用"固定低通胀 + 固定名义利率"近似
                    "任意通胀 + 恒定真实利率"。
    n_boot:         整群自助重复次数(0 = 不算区间)。
    """
    w = hist.pool(s.hold, tercile)
    gaps = np.empty(len(w))
    wins = np.empty(len(w), dtype=bool)
    for i, row in enumerate(w.itertuples(index=False)):
        infl = infl_fixed if infl_fixed is not None else row.infl
        r_mix = (r_invest_real if r_invest_real is not None
                 else eq_share * row.r_eq + (1 - eq_share) * row.r_bond)
        out = simulate(replace(
            s,
            g_house=nominal(row.g_house, infl),
            g_rent=nominal(row.g_rent, infl),
            r_invest=nominal(r_mix, infl),
            infl=infl,
        ))
        gaps[i] = out["gap_real"] / s.price
        wins[i] = out["gap"] > 0

    q = np.percentile(gaps, [5, 25, 50, 75, 95])
    res = {
        "p_buy_wins": float(wins.mean()),
        "gap_real_pct_of_price": {
            "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]},
        "n_windows": int(len(w)),
        "n_countries": int(w.iso.nunique()),
    }
    if n_boot:
        res["p_buy_wins_ci95"] = country_block_ci(
            w.iso.values, wins, np.mean, n_boot, seed)
    return res
