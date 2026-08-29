"""重叠窗口下的统计推断：聚类标准误与整群自助法置信区间。

论文第 4 节的窗口是滚动重叠的（相邻 10 年窗口共享 9 年），且各国房价受同一轮
全球周期驱动。因此名义 N（1510）远大于独立样本量：朴素 OLS 标准误低估、
频率类点估计没有误差度量。本脚本给出两者的修正：

- 均值回归斜率：朴素 OLS / 按国家聚类 / 按起始年聚类 / 双向聚类(Cameron-
  Gelbach-Miller) 四种标准误。按国家聚类时用 t(G−1) 而非正态分布定 p 值
  （G=16 个国家，小样本）。
- 频率类结论：以国家为整群(block)自助抽样，给出 95% 置信区间。整群抽样同时
  吸收了窗口重叠与国别内相关。

输入:  data/derived/windows.csv  (由 analysis/stylized_facts.py 生成)
输出:  data/derived/uncertainty.json
用法:  python3 analysis/uncertainty.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
B = 4000          # 自助重复次数
SEED = 0


def _ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, y - X @ b, np.linalg.inv(X.T @ X)


def _meat(X, e, groups):
    """聚类三明治的中间项 Σ_g (X_g'e_g)(X_g'e_g)'。"""
    m = np.zeros((X.shape[1], X.shape[1]))
    for gv in pd.unique(groups):
        sel = groups == gv
        u = (X[sel] * e[sel, None]).sum(axis=0)
        m += np.outer(u, u)
    return m


def clustered_se(y, X, groups):
    """一维聚类稳健标准误(含 Cameron 等的小样本修正)。"""
    b, e, XtXi = _ols(y, X)
    n, k = X.shape
    G = len(pd.unique(groups))
    c = G / (G - 1) * (n - 1) / (n - k)
    return np.sqrt(np.diag(c * XtXi @ _meat(X, e, groups) @ XtXi)), G


def twoway_se(y, X, g1, g2):
    """双向聚类(Cameron-Gelbach-Miller 2011): V1 + V2 − V12。"""
    b, e, XtXi = _ols(y, X)
    both = pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str)
    V = (XtXi @ _meat(X, e, np.asarray(g1)) @ XtXi
         + XtXi @ _meat(X, e, np.asarray(g2)) @ XtXi
         - XtXi @ _meat(X, e, both.values) @ XtXi)
    return np.sqrt(np.abs(np.diag(V)))


def mean_reversion_inference(win, horizon):
    """估值偏离 → 未来涨幅 的回归, 四种标准误口径。"""
    sub = win[win.horizon == horizon].dropna(subset=["dev0", "g_house"])
    y = sub.g_house.values
    X = np.column_stack([np.ones(len(sub)), sub.dev0.values])
    b, _, _ = _ols(y, X)
    n, k = X.shape

    se_ols = np.sqrt(np.diag(
        (lambda e: e @ e / (n - k))(y - X @ b) * np.linalg.inv(X.T @ X)))[1]
    se_ctry, G = clustered_se(y, X, sub.iso.values)
    se_ctry = se_ctry[1]
    se_yr = clustered_se(y, X, sub.year0.values)[0][1]
    se_2w = twoway_se(y, X, sub.iso.values, sub.year0.values)[1]

    slope = float(b[1])
    out = {"slope": round(slope, 5), "lambda": round(-slope, 5),
           "n": int(n), "n_countries": int(G)}
    for name, se in [("ols", se_ols), ("country", se_ctry),
                     ("year0", se_yr), ("twoway", se_2w)]:
        out[name] = {"se": round(float(se), 5), "t": round(float(slope / se), 3)}
    # 主口径(双向聚类)的 p 值以 t(G−1) 计, G 为国家数——聚类数少时的保守选择
    out["twoway"]["p_two_sided"] = round(
        float(2 * stats.t.sf(abs(slope / se_2w), G - 1)), 4)
    out["country"]["p_two_sided"] = round(
        float(2 * stats.t.sf(abs(slope / se_ctry), G - 1)), 4)
    return out


class CountryBootstrap:
    """以国家为整群的自助法: 每次有放回抽 G 个国家, 拼接其全部窗口。"""

    def __init__(self, df, seed=SEED):
        self.groups = [d for _, d in df.groupby("iso")]
        self.G = len(self.groups)
        self.rng = np.random.default_rng(seed)

    def ci(self, fn, b=B, qs=(2.5, 97.5)):
        point = fn(pd.concat(self.groups))
        draws = np.empty(b)
        for i in range(b):
            pick = self.rng.integers(0, self.G, self.G)
            draws[i] = fn(pd.concat([self.groups[j] for j in pick]))
        lo, hi = np.percentile(draws, qs)
        return {"point": round(float(point), 4), "ci95": [round(float(lo), 4),
                                                          round(float(hi), 4)],
                "p_le_0": round(float(np.mean(draws <= 0)), 4)}


def main():
    win = pd.read_csv(DERIVED / "windows.csv")
    out = {"method": "block bootstrap over countries; two-way clustered SEs",
           "n_bootstrap": B, "seed": SEED,
           "n_countries_windows": int(win.iso.nunique()),
           "countries_windows": sorted(win.iso.unique().tolist())}

    out["mean_reversion"] = {str(k): mean_reversion_inference(win, k)
                             for k in (5, 10)}

    w10 = win[win.horizon == 10].dropna(subset=["g_house"])
    bs = CountryBootstrap(w10)

    out["g10_prob_exceed_ci"] = {
        f"{t}%": bs.ci(lambda d, t=t: (d.g_house >= t / 100).mean())
        for t in (0, 1, 2, 3, 4, 5)}

    terc = {}
    for i, name in enumerate(["cheap", "mid", "dear"]):
        terc[name] = {
            "p_le_0": bs.ci(lambda d, i=i: (d[d.tercile == i].g_house <= 0).mean()),
            "median": bs.ci(lambda d, i=i: d[d.tercile == i].g_house.median())}
    # 条件化到底有没有用: 高估组与低估组的对比本身要有区间
    terc["contrast_dear_minus_cheap"] = {
        "p_le_0": bs.ci(lambda d: (d[d.tercile == 2].g_house <= 0).mean()
                        - (d[d.tercile == 0].g_house <= 0).mean()),
        "median_cheap_minus_dear": bs.ci(
            lambda d: d[d.tercile == 0].g_house.median()
            - d[d.tercile == 2].g_house.median())}
    out["g10_by_tercile_ci"] = terc

    with open(DERIVED / "uncertainty.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nderived ->", DERIVED / "uncertainty.json")


if __name__ == "__main__":
    main()
