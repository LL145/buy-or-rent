"""buyrent: 买房还是租房的概率化估算。

用法见 buyrent/cli.py 与 docs/research-framework.md。
"""
from .breakeven import breakeven_growth, breakeven_growth_real
from .history import History, snap_horizon, valuation_dev
from .model import Scenario, simulate
from .montecarlo import run

__all__ = ["Scenario", "simulate", "breakeven_growth", "breakeven_growth_real",
           "History", "valuation_dev", "snap_horizon", "run"]
