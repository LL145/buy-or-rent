"""二因素速查表: 租金收益率 × 持有期 → 买/租/中性。

其余参数固定为中国 2026 共同环境(按揭3.1%, 首付30%, 中国成本结构)。
每格跑两类投资者(保守 真实1.5% / 均衡 真实3%)的蒙特卡洛,
判定对两者同时稳健才给出方向:
    买   min(P保守, P均衡) >= 55%
    租   max(P保守, P均衡) <= 45%
    中性  其余(金融账接近打平, 非金融因素定夺)

用法: python3 analysis/lookup_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buyrent import History, Scenario, run  # noqa: E402

COMMON = dict(down=0.30, mort_rate=0.031, mort_years=30, buy_cost=0.025,
              sell_cost=0.015, carry=0.007, infl=0.005, g_rent=0.005,
              r_invest=0.02)
YIELDS = [0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
HOLDS = [3, 5, 10, 15, 20]
INVESTORS = [0.015, 0.030]

hist = History()
grid = {}
half_widths = []
for h in HOLDS:
    for ry in YIELDS:
        s = Scenario(rent_yield=ry, hold=h, **COMMON)
        # 每格也配 95% 区间: 论文说"格内数字的区间约 ±N 个百分点", N 必须算出来
        mcs = [run(s, hist, r_invest_real=rr, infl_fixed=0.005, seed=3)
               for rr in INVESTORS]
        ps = [m["p_buy_wins"] for m in mcs]
        lo, hi = min(ps), max(ps)
        verdict = "买" if lo >= 0.55 else ("租" if hi <= 0.45 else "中性")
        cell = {"p_cons": round(ps[0], 3), "p_bal": round(ps[1], 3),
                "p_cons_ci95": [round(v, 3) for v in mcs[0]["p_buy_wins_ci95"]],
                "p_bal_ci95": [round(v, 3) for v in mcs[1]["p_buy_wins_ci95"]],
                "verdict": verdict}
        for m in mcs:
            a, b = m["p_buy_wins_ci95"]
            half_widths.append((b - a) / 2)
        grid[f"{h}|{ry}"] = cell
        print(f"hold={h:2d} ry={ry:.1%}  保守={ps[0]:.0%}{cell['p_cons_ci95']} "
              f"均衡={ps[1]:.0%}{cell['p_bal_ci95']}  {verdict}")

# 区间半宽的中位数与最大值: 论文 8.4 引用的是这两个数, 不是拍脑袋的 ±7
grid["_ci_half_width_pp"] = {"median": round(100 * float(np.median(half_widths)), 1),
                             "max": round(100 * float(np.max(half_widths)), 1)}
print("区间半宽(百分点): 中位", grid["_ci_half_width_pp"]["median"],
      "最大", grid["_ci_half_width_pp"]["max"])

json.dump(grid, open(ROOT / "data" / "derived" / "lookup_table.json", "w"),
          indent=1, ensure_ascii=False)
print("saved data/derived/lookup_table.json")
