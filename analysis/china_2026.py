"""2026 年中国分城市算例(论文第 8 节) + fig7。

参数来源(2026年8月检索, 详见论文脚注):
- 新发个人房贷加权平均利率 ~3.1% (2026Q2), 5年期LPR 3.5%
- 租金收益率: 北京1.64% 上海1.95% 广州1.87% 深圳1.82% (2025.11);
  二线平均 2.36% (2026.3, 杭苏~2.0, 成都武汉>2.3); 三四线 ~2.2%
- CPI ≈ 0, 10年期国债 1.73%; 2026H1 百城二手房价 -2.9%(跌幅收窄)

用法: python3 analysis/china_2026.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buyrent import (History, Scenario, breakeven_growth,  # noqa: E402
                     breakeven_growth_real, run, valuation_dev)

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": BASE,
    "font.size": 10,
})

# 中国 2026 共同参数: 按揭3.1%(30年), 首付30%, 契税+中介≈2.5%,
# 卖出≈1.5%(满五唯一), 无房产税→持有成本0.7%, 通胀0.5%, 租金短期零增长
COMMON = dict(down=0.30, mort_rate=0.031, mort_years=30, buy_cost=0.025,
              sell_cost=0.015, carry=0.007, infl=0.005, g_rent=0.005,
              r_invest=0.02, hold=10)

# (名称, 当前租金收益率, 本市25年常态租金收益率)
# 第三项是**假设**而非数据: 中国没有公开的城市级 25 年租金收益率序列可查。
# 它只通过估值分组(三分位)进入模型, 所以下面对每个城市另算"若落入中间组 /
# 不分组"的结果, 以及常态低到多少就会换组——论文第 8.2 节必须把这个边界写出来。
CITIES = [
    ("一线城市\n(京沪广深)", 0.018, 0.024),
    ("强二线\n(杭州苏州)", 0.020, 0.024),
    ("二线\n(成都武汉等)", 0.024, 0.028),
    ("三四线\n(人口流出)", 0.022, 0.030),
]
# 租方真实投资收益率: 保守(存款/国债/理财) vs 均衡(含权益/全球配置)
INVESTORS = [("保守投资者(真实1.5%)", 0.015), ("均衡投资者(真实3%)", 0.030)]

hist = History()
rows = []
for name, ry, ry_typ in CITIES:
    s = Scenario(rent_yield=ry, **COMMON)
    dev = valuation_dev(ry, ry_typ)
    tc = hist.tercile_of(dev, s.hold)
    g_star_n = breakeven_growth(s)
    g_star_r = breakeven_growth_real(s)
    p_u, n_u = hist.prob_exceed(g_star_r, s.hold)
    p_c, n_c = hist.prob_exceed(g_star_r, s.hold, tc)
    row = {"city": name.replace("\n", ""), "ry": ry, "ry_typical": ry_typ,
           "dev": round(dev, 3), "tercile": tc,
           "g_star_nominal": round(g_star_n, 4),
           "g_star_real": round(g_star_r, 4),
           "p_hist_uncond": round(p_u, 3), "p_hist_cond": round(p_c, 3),
           "p_hist_cond_ci95": [round(v, 3) for v in
                                hist.prob_exceed_ci(g_star_r, s.hold, tc)]}
    for label, rr in INVESTORS:
        mc = run(s, hist, tercile=tc, r_invest_real=rr, infl_fixed=0.005, seed=11)
        key = "conservative" if rr == 0.015 else "balanced"
        row[f"p_buy_{key}"] = round(mc["p_buy_wins"], 3)
        # 分城市结论同样只能读到十位数——区间与点估计一起入库
        row[f"p_buy_{key}_ci95"] = [round(v, 3) for v in mc["p_buy_wins_ci95"]]
        row[f"gap_median_{key}"] = round(
            mc["gap_real_pct_of_price"]["median"], 3)
        row[f"gap_p5_{key}"] = round(mc["gap_real_pct_of_price"]["p5"], 3)
        row[f"gap_p95_{key}"] = round(mc["gap_real_pct_of_price"]["p95"], 3)
    # 估值分组的敏感性(保守投资者): 当前组 / 中间组 / 不分组。
    # 常态租金收益率低于 ry·exp(q2) 时该城落入中间组——这就是假设的"翻转门槛"。
    q2 = hist.cutoffs[s.hold][1]
    sens = {}
    for label, t in (("mid", 1), ("uncond", None)):
        mc = run(s, hist, tercile=t, r_invest_real=0.015, infl_fixed=0.005,
                 seed=11)
        sens[label] = {"p_buy": round(mc["p_buy_wins"], 3),
                       "p_buy_ci95": [round(v, 3) for v in mc["p_buy_wins_ci95"]],
                       "p_hist": round(hist.prob_exceed(g_star_r, s.hold, t)[0], 3)}
    row["tercile_sensitivity"] = sens
    row["ry_typical_to_mid"] = round(float(ry * np.exp(q2)), 4)
    rows.append(row)
    print(row)

json.dump(rows, open(ROOT / "data" / "derived" / "china_2026.json", "w"),
          indent=2, ensure_ascii=False)

# ------- fig7: 城市×投资者类型 的买房胜率 + 盈亏平衡涨幅 -------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
ax = axes[0]
x = np.arange(len(CITIES))
wbar = 0.36
for j, (label, rr) in enumerate(INVESTORS):
    key = "conservative" if rr == 0.015 else "balanced"
    vals = [r[f"p_buy_{key}"] * 100 for r in rows]
    b = ax.bar(x + (j - 0.5) * wbar, vals, wbar * 0.94,
               color=[BLUE, ORANGE][j], edgecolor=SURF, lw=1, label=label)
    for xi, v in zip(x + (j - 0.5) * wbar, vals):
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", color=SEC, fontsize=9)
ax.axhline(50, color=BASE, lw=0.9, ls="--")
ax.set_xticks(x, [c[0] for c in CITIES], fontsize=9)
ax.set_ylim(0, 100)
ax.set_ylabel("P(买 优于 租) %")
ax.legend(frameon=False, fontsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("2026年中国: 持有10年的买房胜率(估值条件化)", color=INK, fontsize=12)

ax = axes[1]
gs = [r["g_star_real"] * 100 for r in rows]
ax.bar(x, gs, 0.5, color=BLUE, edgecolor=SURF, lw=1)
for xi, v, r in zip(x, gs, rows):
    ax.text(xi, v + 0.08, f"{v:.1f}%", ha="center", color=SEC, fontsize=9)
    ax.text(xi, -0.45, f"历史频率 {r['p_hist_cond']:.0%}", ha="center",
            color=MUT, fontsize=8.5)
ax.axhline(1.0, color=ORANGE, lw=1.4, ls="--")
ax.text(-0.42, 1.06, "1870年以来中位涨幅 1.0%", color=ORANGE, ha="left", fontsize=9)
ax.axhline(0, color=BASE, lw=0.8)
ax.set_ylim(-0.6, 1.9)
ax.set_xticks(x, [c[0] for c in CITIES], fontsize=9)
ax.set_ylabel("盈亏平衡真实涨幅 g* (%/年)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("买房打平所需涨幅 及其在高估起点后的历史频率", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig7_china_2026.png", dpi=150)
print("saved fig7")
