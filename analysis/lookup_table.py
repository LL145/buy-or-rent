"""8.4 节速查表组: 2026 年中国读者只需要查的一组表。

两张表共用同一套格子——纵轴租金收益率(年租金÷房价), 横轴持有期:
    表 A 判定  每格跑两类租房投资者(保守 真实1.5% / 均衡 真实3%)的全量求值,
              两者同向才给出方向:
                  买   min(P保守, P均衡) >= 55%
                  租   max(P保守, P均衡) <= 45%
                  中性  其余(金融账不裁决)
    表 B 代价  同一格子在三条命名路径(历史中位 / 房价横盘 / 日本路径 -2%/年)下
              的期末财富差占房价(保守投资者, 真实口径)——"中性"格子靠它定夺。
外加一条换算规则的依据: 按"高估组"条件化后每格胜率比不分组低多少, 折成
表格上要下移几行。

持有期从 5 年起: 红线 1(持有不足 5 年不买)优先于任何概率, 表格不该在
"3 年"列里给出"买"。其余参数固定为中国 2026 共同环境(按揭 3.1%, 首付 30%,
中国成本结构, 浮动利率口径)。

用法: python3 analysis/lookup_table.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buyrent import History, Scenario, run, simulate  # noqa: E402
from buyrent.montecarlo import nominal  # noqa: E402

COMMON = dict(down=0.30, mort_rate=0.031, mort_years=30, buy_cost=0.025,
              sell_cost=0.015, carry=0.007, infl=0.005, g_rent=0.005,
              r_invest=0.02)
YIELDS = [0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
HOLDS = [5, 10, 15, 20]
INVESTORS = [0.015, 0.030]
HIGH_TERCILE = 2        # 估值三分位: 2 = 租售比相对本地常态偏低(高估)
JAPAN_G = -0.02         # 与 analysis/stress.py 相同


def gap_under(s: Scenario, g_real: float) -> float:
    """给定真实房价年涨幅下的期末财富差(占房价, 真实口径)。与 stress.py 同一定义。"""
    out = simulate(replace(s, g_house=nominal(g_real, s.infl)))
    return round(out["gap_real"] / s.price, 4)


hist = History()
median_g = {h: float(np.median(hist.pool(h).g_house)) for h in HOLDS}
grid = {}
half_widths, shifts, flips = [], [], []
p_cons = {}
for h in HOLDS:
    for ry in YIELDS:
        s = Scenario(rent_yield=ry, hold=h, **COMMON)
        # 表 A: 两档投资者, 每格配 95% 区间(论文引用区间半宽的中位与最大值)
        mcs = [run(s, hist, r_invest_real=rr, infl_fixed=0.005, seed=3)
               for rr in INVESTORS]
        ps = [m["p_buy_wins"] for m in mcs]
        lo, hi = min(ps), max(ps)
        verdict = "买" if lo >= 0.55 else ("租" if hi <= 0.45 else "中性")
        # 换算规则的依据: 同一格子按高估组条件化后两档投资者的胜率与判定
        ps_high = [run(s, hist, tercile=HIGH_TERCILE, r_invest_real=rr,
                       infl_fixed=0.005, n_boot=0)["p_buy_wins"] for rr in INVESTORS]
        p_high = ps_high[0]
        verdict_high = ("买" if min(ps_high) >= 0.55
                        else ("租" if max(ps_high) <= 0.45 else "中性"))
        cell = {"p_cons": round(ps[0], 3), "p_bal": round(ps[1], 3),
                "p_cons_ci95": [round(v, 3) for v in mcs[0]["p_buy_wins_ci95"]],
                "p_bal_ci95": [round(v, 3) for v in mcs[1]["p_buy_wins_ci95"]],
                "verdict": verdict,
                "p_cons_high_tercile": round(p_high, 3),
                "p_bal_high_tercile": round(ps_high[1], 3),
                "verdict_high_tercile": verdict_high,
                # 表 B: 三条命名路径下的财富差占房价
                "paths": {"median": gap_under(s, median_g[h]),
                          "flat": gap_under(s, 0.0),
                          "japan": gap_under(s, JAPAN_G)}}
        for m in mcs:
            a, b = m["p_buy_wins_ci95"]
            half_widths.append((b - a) / 2)
        shifts.append((h, (ps[0] - p_high) * 100))
        if verdict_high != verdict:
            flips.append({"hold": h, "rent_yield": ry, "uncond": verdict,
                          "high_tercile": verdict_high})
        p_cons[(h, ry)] = ps[0]
        grid[f"{h}|{ry}"] = cell
        pa = cell["paths"]
        print(f"hold={h:2d} ry={ry:.1%}  保守={ps[0]:.0%}{cell['p_cons_ci95']} "
              f"均衡={ps[1]:.0%}{cell['p_bal_ci95']}  {verdict:2s} | 高估组 {p_high:.0%} "
              f"| 中位 {pa['median']:+.0%} 横盘 {pa['flat']:+.0%} 日本 {pa['japan']:+.0%}")

# 区间半宽的中位数与最大值: 论文 8.4 引用的是这两个数, 不是拍脑杆的 ±7
grid["_ci_half_width_pp"] = {"median": round(100 * float(np.median(half_widths)), 1),
                             "max": round(100 * float(np.max(half_widths)), 1)}
# 估值换算规则: 高估组条件化把胜率压低多少, 对比相邻两行(租金收益率差 0.5pp)的差
row_steps = [(p_cons[(h, YIELDS[i + 1])] - p_cons[(h, YIELDS[i])]) * 100
             for h in HOLDS for i in range(3)]        # 1.5→2.0→2.5→3.0 三步
sh = [v for _, v in shifts]
grid["_valuation_shift_pp"] = {
    "high_tercile_minus_uncond_median": round(-float(np.median(sh)), 1),
    "range": [round(-max(sh), 1), round(-min(sh), 1)],
    "by_hold_median": {str(h): round(-float(np.median([v for hh, v in shifts if hh == h])), 1)
                       for h in HOLDS},
    "adjacent_row_step_median": round(float(np.median(row_steps)), 1),
    "verdict_flips": flips,
}
grid["_median_g_real"] = {str(h): round(median_g[h], 4) for h in HOLDS}
grid["_japan_path_g_real"] = JAPAN_G
print("区间半宽(百分点): 中位", grid["_ci_half_width_pp"]["median"],
      "最大", grid["_ci_half_width_pp"]["max"])
print("高估组条件化的胜率变动(百分点):", grid["_valuation_shift_pp"])
print("各持有期历史中位真实涨幅:", grid["_median_g_real"])

json.dump(grid, open(ROOT / "data" / "derived" / "lookup_table.json", "w"),
          indent=1, ensure_ascii=False)
print("saved data/derived/lookup_table.json")
