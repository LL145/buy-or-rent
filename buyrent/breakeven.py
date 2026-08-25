"""盈亏平衡涨幅 g*: 使 买 与 租 期末财富相等的名义房价年涨幅。"""
from scipy.optimize import brentq

from .model import Scenario, simulate, with_growth


def breakeven_growth(s: Scenario, lo: float = -0.20, hi: float = 0.25) -> float:
    """解 wealth_buy(g) = wealth_rent。gap 对 g 单调递增, 用 brentq。"""
    f = lambda g: simulate(with_growth(s, g))["gap"]
    if f(lo) > 0:       # 房价年跌 20% 买仍划算(几乎不可能出现)
        return lo
    if f(hi) < 0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-12))


def breakeven_growth_real(s: Scenario, **kw) -> float:
    """真实口径的 g*(与历史分布同口径)。"""
    g = breakeven_growth(s, **kw)
    return (1 + g) / (1 + s.infl) - 1
