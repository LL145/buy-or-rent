"""buyrent: 买房还是租房的概率化估算。

面向使用者的入口是网页计算器(web/, 见 README); 本包是它与全部论文数字背后的引擎。
方法说明见 docs/research-framework.md。
"""
from .breakeven import breakeven_growth, breakeven_growth_real
from .history import History, snap_horizon, valuation_dev
from .model import Scenario, simulate
from .montecarlo import run

__all__ = ["Scenario", "simulate", "breakeven_growth", "breakeven_growth_real",
           "History", "valuation_dev", "snap_horizon", "run"]
