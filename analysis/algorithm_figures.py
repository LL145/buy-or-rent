"""fig6: 算法输出总览 —— 买房胜率如何随租金收益率与持有期变化。

用法: python3 analysis/algorithm_figures.py   (先跑 stylized_facts.py)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buyrent import History, Scenario, breakeven_growth_real, run  # noqa: E402

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": BASE,
    "font.size": 10,
})


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


hist = History()
HOLDS = [5, 10, 15, 20]
YIELDS = [0.015, 0.025, 0.035, 0.055]
LABELS = ["1.5%(极低,如一线核心区)", "2.5%", "3.5%", "5.5%(高,如美国多数城市)"]


def scen(ry, hold):
    return Scenario(rent_yield=ry, down=0.30, mort_rate=0.045, hold=hold)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
ax = axes[0]
for c, ry, lab in zip(COLS, YIELDS, LABELS):
    ps = [run(scen(ry, h), hist, seed=7, n_boot=0)["p_buy_wins"] for h in HOLDS]
    ax.plot(HOLDS, np.array(ps) * 100, color=c, lw=2, marker="o", ms=5,
            markeredgecolor=SURF, label=f"租金收益率 {lab}")
ax.axhline(50, color=BASE, lw=0.9, ls="--")
ax.text(20.2, 50, "五五开", color=MUT, va="center", fontsize=9)
ax.set_xticks(HOLDS)
ax.set_ylim(0, 100)
ax.set_xlabel("持有期(年)")
ax.set_ylabel("P(买 优于 租) %")
ax.legend(frameon=False, fontsize=9, loc="upper left")
style(ax)
ax.set_title("买房胜率 = f(租金收益率, 持有期)", color=INK, fontsize=12)

ax = axes[1]
rys = np.linspace(0.012, 0.065, 40)
for c, h in zip([COLS[0], COLS[1], COLS[2]], [5, 10, 20]):
    gs = [breakeven_growth_real(scen(r, h)) * 100 for r in rys]
    ax.plot(rys * 100, gs, color=c, lw=2, label=f"持有 {h} 年")
ax.axhline(1.0, color=BASE, lw=0.9, ls="--")
ax.text(6.4, 1.15, "历史中位涨幅 1.0%", color=MUT, ha="right", fontsize=9)
ax.set_xlabel("租金收益率(%)")
ax.set_ylabel("盈亏平衡真实涨幅 g* (%/年)")
ax.legend(frameon=False, fontsize=9)
style(ax)
ax.set_title("买房需要多高的涨幅才划算(按揭4.5%, 首付30%)", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig6_algorithm.png", dpi=150)
print("saved fig6")
