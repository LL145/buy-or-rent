"""历史查询层: 把"未来涨幅假设"换算成历史分位数。

数据来自 data/derived/windows.csv (由 analysis/stylized_facts.py 生成):
1870–2020 年 18 国所有 国家×起始年×持有期 窗口的联合年化实现值
(真实房价涨幅、真实租金涨幅、股/债真实回报、通胀、期初估值偏离)。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import N_BOOT, country_block_ci

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (5, 10, 15, 20)
TERCILE_NAMES = {0: "低估", 1: "中间", 2: "高估"}


def snap_horizon(hold: int) -> int:
    return min(HORIZONS, key=lambda k: abs(k - hold))


def valuation_dev(ry_now: float, ry_typical: float) -> float:
    """估值偏离 = log(当前租售比) − log(历史常态租售比) = log(ry_typical/ry_now)。

    ry_now: 当前租金收益率; ry_typical: 本市过去 ~20 年的典型租金收益率。
    正值 = 比历史贵(高估方向)。
    """
    return float(np.log(ry_typical / ry_now))


class History:
    def __init__(self, windows_csv: Path | None = None):
        base = windows_csv or ROOT / "data" / "derived" / "windows.csv"
        self.win = pd.read_csv(base)
        with open(ROOT / "data" / "derived" / "tercile_cutoffs.json") as f:
            self.cutoffs = {int(k): v for k, v in json.load(f).items()}

    def tercile_of(self, dev: float, horizon: int) -> int:
        q1, q2 = self.cutoffs[snap_horizon(horizon)]
        return 0 if dev <= q1 else (1 if dev <= q2 else 2)

    def _slice(self, horizon: int, tercile: int | None) -> pd.DataFrame:
        k = snap_horizon(horizon)
        sub = self.win[self.win.horizon == k]
        if tercile is not None:
            sub = sub[sub.tercile == tercile]
        return sub

    def prob_exceed(self, g_real: float, horizon: int,
                    tercile: int | None = None) -> tuple[float, int]:
        """历史上『真实房价年化涨幅 ≥ g_real 持续该持有期』的频率。"""
        x = self._slice(horizon, tercile).g_house.dropna()
        return float((x >= g_real).mean()), int(len(x))

    def prob_exceed_ci(self, g_real: float, horizon: int,
                       tercile: int | None = None, n_boot: int = N_BOOT,
                       seed: int = 0) -> list[float]:
        """上式频率的 95% 区间(以国家为整群自助)。

        频率本身也是估计量。点估计不带区间, 等于用一个看起来精确的数字
        掩盖它背后只有 16 个国家的事实。
        """
        sub = self._slice(horizon, tercile).dropna(subset=["g_house"])
        return country_block_ci(sub.iso.values, sub.g_house.values,
                                lambda a: (a >= g_real).mean(), n_boot, seed)

    def quantiles(self, horizon: int, tercile: int | None = None,
                  qs=(5, 25, 50, 75, 95)) -> dict:
        x = self._slice(horizon, tercile).g_house.dropna()
        return {q: float(np.percentile(x, q)) for q in qs}

    def pool(self, horizon: int, tercile: int | None = None) -> pd.DataFrame:
        """该持有期(可选该估值组)下全部可用的历史窗口。

        蒙特卡洛直接遍历这个池子而非从中有放回抽样: 池子只有数百到一千余个
        窗口, 全量求值既精确又更快, 抽样只会凭空添一层噪声。不确定性由
        `montecarlo.run` 以国家为整群的自助法给出(窗口重叠 + 国别内相关)。
        """
        return self._slice(horizon, tercile).dropna(
            subset=["g_house", "g_rent", "r_eq", "r_bond", "infl"])
