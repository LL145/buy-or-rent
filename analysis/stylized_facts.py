"""复现论文第 4 节的历史典型事实，并生成算法所需的派生数据。

输入:  data/JSTdatasetR6.xlsx  (Jordà-Schularick-Taylor Macrohistory Database R6,
       含 Rate of Return on Everything 回报序列; 下载见 data/download.sh)
输出:  figures/fig1..fig5*.png
       data/derived/windows.csv     — 所有 国家×起始年×持有期 的滚动窗口联合实现值
       data/derived/key_stats.json  — 论文引用的全部关键数字
       data/derived/tercile_cutoffs.json — 估值三分位切点(算法查询用)

用法:  python3 analysis/stylized_facts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DERIVED = ROOT / "data" / "derived"
FIG.mkdir(exist_ok=True)
DERIVED.mkdir(parents=True, exist_ok=True)

# ---- 图表样式(经过 CVD 校验的调色板, 见 docs/research-framework.md 引用的规范) ----
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": BASE, "font.size": 10,
})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def ann(x):  # 年化
    return float(np.round(x, 5))


# ================================ 数据准备 ================================
df = pd.read_excel(ROOT / "data" / "JSTdatasetR6.xlsx", sheet_name="Sheet1")
df = df.sort_values(["iso", "year"]).reset_index(drop=True)
g = df.groupby("iso", group_keys=False)

df["infl"] = g["cpi"].pct_change()
df["rhp"] = df["hpnom"] / df["cpi"]                      # 真实房价指数
df["ry"] = df["housing_rent_yd"]                          # 租金收益率 R/P
df["rent_real"] = df["ry"] * df["hpnom"] / df["cpi"]      # 真实租金指数
df["rwage"] = df["wage"] / df["cpi"]
for nom, real in [("housing_tr", "h_tr_r"), ("eq_tr", "eq_tr_r"),
                  ("bond_tr", "bond_tr_r"), ("housing_capgain", "h_cg_r")]:
    df[real] = (1 + df[nom]) / (1 + df["infl"]) - 1
# 恶性通胀年份(德国 1920-24 等)不进入回报统计
df["hyper"] = (df["infl"].abs() > 0.5) | df["infl"].isna()

# 估值代理: log(租售比) 相对过去25年(至少15年)均值的偏离 —— 决策时刻实时可算
df["lpr"] = np.log(1.0 / df["ry"])
df["dev"] = g["lpr"].apply(
    lambda s: s - s.rolling(25, min_periods=15).mean())

# ================================ 滚动窗口 ================================
HORIZONS = [5, 10, 15, 20]


def window_rows(sub):
    """一个国家的所有 起始年×持有期 窗口: 各变量年化实现值(窗口内数据必须完整)"""
    sub = sub.set_index("year")
    out = []
    lg_hp = np.log(sub["rhp"])
    lg_rent = np.log(sub["rent_real"])
    for k in HORIZONS:
        for t in sub.index:
            if t + k not in sub.index:
                continue
            win = sub.loc[t + 1: t + k]      # 窗口内的 k 个年度回报
            if len(win) != k or win["hyper"].any():
                continue
            row = {"iso": sub["iso"].iloc[0], "year0": int(t), "horizon": k,
                   "dev0": sub.at[t, "dev"], "ry0": sub.at[t, "ry"]}
            ok = True
            for col, src in [("g_house", lg_hp), ("g_rent", lg_rent)]:
                a, b = src.get(t, np.nan), src.get(t + k, np.nan)
                if not (np.isfinite(a) and np.isfinite(b)):
                    ok = False
                    break
                row[col] = np.expm1((b - a) / k)
            if not ok:
                continue
            for col, tr in [("r_house", "h_tr_r"), ("r_eq", "eq_tr_r"),
                            ("r_bond", "bond_tr_r")]:
                x = win[tr]
                row[col] = (np.prod(1 + x) ** (1 / k) - 1) if x.notna().all() else np.nan
            x = win["infl"]
            row["infl"] = (np.prod(1 + x) ** (1 / k) - 1) if x.notna().all() else np.nan
            out.append(row)
    return pd.DataFrame(out)


win = pd.concat([window_rows(s) for _, s in df.groupby("iso")], ignore_index=True)

# 估值三分位(按持有期分别定切点, 只用 dev0 可得的窗口)
cutoffs = {}
for k in HORIZONS:
    sub = win[(win.horizon == k) & win.dev0.notna()]
    q1, q2 = sub["dev0"].quantile([1 / 3, 2 / 3])
    cutoffs[k] = [float(q1), float(q2)]
win["tercile"] = np.nan
for k in HORIZONS:
    m = (win.horizon == k) & win.dev0.notna()
    q1, q2 = cutoffs[k]
    win.loc[m, "tercile"] = np.where(
        win.loc[m, "dev0"] <= q1, 0, np.where(win.loc[m, "dev0"] <= q2, 1, 2))
win.to_csv(DERIVED / "windows.csv", index=False)
json.dump(cutoffs, open(DERIVED / "tercile_cutoffs.json", "w"))

KS = {"n_countries": int(df.iso.nunique()), "years": [1870, 2020]}

# ============================ 事实1: 长期真实房价 ============================
fig, ax = plt.subplots(figsize=(8, 4.6))
med = {}
for iso, sub in df.groupby("iso"):
    s = sub.set_index("year")["rhp"].dropna()
    if 1950 not in s.index:
        continue
    idx = 100 * s / s.loc[1950]
    med[iso] = idx
    ax.plot(idx.index, idx.values, color=MUT, lw=0.6, alpha=0.45)
panel = pd.DataFrame(med)
median = panel.median(axis=1, skipna=True)
ax.plot(median.index, median.values, color=BLUE, lw=2.2)
ax.text(2011, median.loc[2020] * 1.1, "18国中位数", color=BLUE, fontsize=10, ha="right")
ax.set_yscale("log")
ax.set_yticks([50, 100, 200, 400, 800, 1600, 3200])
ax.set_ylim(top=3600)
ax.minorticks_off()
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax.axvline(1950, color=BASE, lw=0.8, ls="--")
style(ax)


def era_growth(y0, y1):
    r = []
    for iso in panel.columns:
        s = panel[iso].dropna()
        s = s[(s.index >= y0) & (s.index <= y1)]
        if len(s) >= 30:
            r.append((s.iloc[-1] / s.iloc[0]) ** (1 / (s.index[-1] - s.index[0])) - 1)
    return float(np.median(r))


g_pre, g_post = era_growth(1870, 1950), era_growth(1950, 2020)
KS["real_hp_growth_pre1950"] = ann(g_pre)
KS["real_hp_growth_post1950"] = ann(g_post)
KS["real_hp_growth_full"] = ann(era_growth(1870, 2020))
ax.text(1905, 210, f"1870–1950\n中位 {g_pre * 100:+.1f}%/年", color=SEC, ha="center")
ax.text(1988, 60, f"1950–2020\n中位 {g_post * 100:+.1f}%/年", color=SEC, ha="center")
ax.set_title("真实房价指数(1950=100, 对数轴): 长期上涨集中在1950年之后", color=INK, fontsize=12)
ax.set_xlabel("年份")
fig.tight_layout()
fig.savefig(FIG / "fig1_real_house_prices.png", dpi=150)
plt.close(fig)

# ====================== 事实2: 住房回报分解 + 与股票比较 ======================
ok = df[~df.hyper]
by = ok.groupby("iso").agg(ry=("ry", "mean"), cg=("h_cg_r", "mean"),
                           eq=("eq_tr_r", "mean"), n=("ry", "count"))
by = by[by["n"] >= 40].sort_values("ry")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), width_ratios=[1.25, 1])
ax = axes[0]
ypos = np.arange(len(by))
ax.barh(ypos, by["ry"] * 100, color=BLUE, edgecolor=SURF, lw=1, label="租金收益率")
ax.barh(ypos, by["cg"] * 100, left=by["ry"] * 100, color=ORANGE, edgecolor=SURF, lw=1,
        label="真实资本利得")
ax.scatter(by["eq"] * 100, ypos, color=AQUA, s=28, zorder=3, label="股票真实总回报")
ax.set_yticks(ypos, by.index)
ax.set_xlabel("年均回报(%, 1870–2020, 剔除恶性通胀年份)")
ax.legend(frameon=False, loc="lower right", fontsize=9)
style(ax, grid_axis="x")
ax.set_title("住房总回报的大头是租金,不是涨价", color=INK, fontsize=12)
KS["mean_rent_yield"] = ann(by["ry"].mean())
KS["mean_real_capgain"] = ann(by["cg"].mean())
KS["mean_housing_tr_real"] = ann(ok.groupby("iso")["h_tr_r"].mean().mean())
KS["mean_eq_tr_real"] = ann(by["eq"].mean())

ax = axes[1]
w10 = win[win.horizon == 10]
bins = np.arange(-10, 16, 1)
for col, c, lab in [("r_house", BLUE, "住房总回报"), ("r_eq", ORANGE, "股票总回报")]:
    x = w10[col].dropna() * 100
    ax.hist(x, bins=bins, histtype="step", lw=2, color=c, density=True, label=lab)
ax.set_xlabel("滚动10年年化真实回报(%)")
ax.set_ylabel("密度")
ax.legend(frameon=False, fontsize=9)
style(ax)
ax.set_title("10年期回报: 住房≈股票,但更稳", color=INK, fontsize=12)
KS["housing_tr_10y_std"] = ann(w10.r_house.dropna().std())
KS["eq_tr_10y_std"] = ann(w10.r_eq.dropna().std())
fig.tight_layout()
fig.savefig(FIG / "fig2_return_decomposition.png", dpi=150)
plt.close(fig)

# ======================== 事实3: 租售比均值回归(λ) ========================
fig, ax = plt.subplots(figsize=(7, 4.8))
reg = {}
for k in [5, 10]:
    sub = win[(win.horizon == k)].dropna(subset=["dev0", "g_house"])
    r = stats.linregress(sub.dev0, sub.g_house)
    reg[k] = {"lambda": ann(-r.slope), "r2": ann(r.rvalue ** 2),
              "n": int(len(sub)), "t": ann(r.slope / r.stderr)}
KS["mean_reversion"] = reg
sub = win[(win.horizon == 10)].dropna(subset=["dev0", "g_house"])
ax.scatter(sub.dev0, sub.g_house * 100, s=9, color=MUT, alpha=0.35, lw=0)
xs = np.linspace(sub.dev0.min(), sub.dev0.max(), 50)
r = stats.linregress(sub.dev0, sub.g_house)
ax.plot(xs, (r.intercept + r.slope * xs) * 100, color=BLUE, lw=2.2)
ax.axhline(0, color=BASE, lw=0.8)
ax.axvline(0, color=BASE, lw=0.8)
ax.text(0.03, 0.95,
        f"斜率 = {r.slope:.3f}  (t = {r.slope / r.stderr:.1f})\n"
        f"R² = {r.rvalue ** 2:.2f},  N = {len(sub)}",
        transform=ax.transAxes, va="top", color=SEC)
ax.set_xlabel("期初 log(租售比) 相对过去25年均值的偏离")
ax.set_ylabel("未来10年真实房价年化涨幅(%)")
ax.set_title("买得越贵,后面涨得越少: 租售比偏离负向预测未来房价", color=INK, fontsize=12)
style(ax, grid_axis="both")
fig.tight_layout()
fig.savefig(FIG / "fig3_mean_reversion.png", dpi=150)
plt.close(fig)

# ================= 事实4: 未来涨幅的历史分布(算法查询表) =================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), width_ratios=[1.25, 1])
ax = axes[0]
x = w10.g_house.dropna() * 100
ax.hist(x, bins=np.arange(-8, 12.5, 0.75), color=BLUE, edgecolor=SURF, lw=0.8)
for q, labpos in [(0.5, 1.0)]:
    v = np.percentile(x, q * 100)
    ax.axvline(v, color=INK, lw=1.2, ls="--")
    ax.text(v + 0.2, ax.get_ylim()[1] * 0.95, f"中位数 {v:.1f}%", color=SEC, va="top")
ax.set_xlabel("未来10年真实房价年化涨幅(%)")
ax.set_ylabel("窗口数")
ax.set_title("1870年以来所有『国家×10年』窗口的实际涨幅", color=INK, fontsize=12)
style(ax)
q = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
KS["g10_quantiles"] = {p: ann(v / 100) for p, v in zip([5, 10, 25, 50, 75, 90, 95], q)}
KS["g10_prob_exceed"] = {f"{t}%": ann((x / 100 >= t / 100).mean())
                         for t in [0, 1, 2, 3, 4, 5]}
KS["n_windows_10y"] = int(len(x))

ax = axes[1]
names = ["低估\n(租售比偏低)", "中间", "高估\n(租售比偏高)"]
cols = [BLUE, MUT, RED]
terc = {}
for i in range(3):
    xx = w10[(w10.tercile == i)].g_house.dropna() * 100
    p5, p25, p50, p75, p95 = np.percentile(xx, [5, 25, 50, 75, 95])
    ax.plot([i, i], [p5, p95], color=cols[i], lw=1.4)
    ax.plot([i, i], [p25, p75], color=cols[i], lw=6, solid_capstyle="butt")
    ax.scatter([i], [p50], color=SURF, edgecolor=cols[i], s=42, zorder=3, lw=1.6)
    ax.text(i + 0.12, p50, f"{p50:.1f}%", color=SEC, va="center", fontsize=9)
    terc[["cheap", "mid", "dear"][i]] = {
        "median": ann(p50 / 100),
        "p_ge_2pct": ann((xx / 100 >= 0.02).mean()),
        "p_le_0": ann((xx / 100 <= 0).mean())}
KS["g10_by_tercile"] = terc
ax.axhline(0, color=BASE, lw=0.8)
ax.set_xticks(range(3), names)
ax.set_ylabel("未来10年真实房价年化涨幅(%)")
ax.set_title("期初估值决定后验分布(5–95% 与四分位区间)", color=INK, fontsize=12)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "fig4_growth_distribution.png", dpi=150)
plt.close(fig)

# ==================== 事实5: 租金长期贴着收入/通胀走 ====================
fig, ax = plt.subplots(figsize=(6.4, 5.2))
pts = []
for iso, sub in df.groupby("iso"):
    s = sub.dropna(subset=["rent_real", "rwage"])
    s = s[~s.hyper]
    if len(s) < 50:
        continue
    yrs = s.year.iloc[-1] - s.year.iloc[0]
    gr = (s.rent_real.iloc[-1] / s.rent_real.iloc[0]) ** (1 / yrs) - 1
    gw = (s.rwage.iloc[-1] / s.rwage.iloc[0]) ** (1 / yrs) - 1
    pts.append((iso, gw, gr))
pts = pd.DataFrame(pts, columns=["iso", "gw", "gr"])
lim = [-0.5, 3.4]
ax.plot(lim, lim, color=BASE, lw=1, ls="--")
ax.text(lim[1], lim[1] - 0.28, "45°线:租金=收入", color=MUT, ha="right", fontsize=9)
ax.scatter(pts.gw * 100, pts.gr * 100, color=BLUE, s=34, zorder=3)
nudge = {"NLD": (5, -10), "FRA": (-22, 3), "ESP": (-26, -2)}
for _, p in pts.iterrows():
    ax.annotate(p.iso, (p.gw * 100, p.gr * 100), textcoords="offset points",
                xytext=nudge.get(p.iso, (5, 3)), color=MUT, fontsize=8)
ax.set_xlabel("真实工资年均增速(%)")
ax.set_ylabel("真实租金年均增速(%)")
ax.set_title("百年尺度上,真实租金增速低于且贴着收入增速", color=INK, fontsize=12)
style(ax, grid_axis="both")
KS["mean_real_rent_growth"] = ann(pts.gr.mean())
KS["mean_real_wage_growth"] = ann(pts.gw.mean())
fig.tight_layout()
fig.savefig(FIG / "fig5_rent_vs_income.png", dpi=150)
plt.close(fig)

json.dump(KS, open(DERIVED / "key_stats.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(KS, indent=2, ensure_ascii=False))
print("\nfigures ->", FIG, "\nderived ->", DERIVED)
