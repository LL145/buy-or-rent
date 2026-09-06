"""第 8 节的压力测试与稳健性检查: 让正文里每一个"手算"数字都有出处。

这些数字过去是一次性算出来抄进论文的, 参数一改就悄悄失真(本脚本第一次运行时
就发现一线"横盘亏 17%、日本路径亏 36%"已经对不上当前参数)。此后它们全部从
这里产出并写入 data/derived/stress.json, 由 tests/test_paper_consistency.py 锁定。

内容:
1. 分城市命名情景: 房价横盘 / 日本路径(真实 -2%/年) / 历史中位涨幅 下的
   十年财富差(占房价, 真实口径); 三四线另按卖出成本 5% 重算。
2. 浮动利率修正的大小: 若照搬历史通胀并锁定名义月供, 胜率被抬高多少。
3. 2021 年顶部的一线城市对照(参数记录在 SCENARIO_2021)。
4. g* 对各因素的弹性(8.4 节"哪个因素配当表格的轴")。
5. 一线城市月供 / 租金 之比。
6. 以国家等权而非窗口等权的胜率——窗口数最多的六国占十年期窗口的一半以上,
   这里检查结论是否被它们主导。

用法: python3 analysis/stress.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buyrent import (History, Scenario, breakeven_growth_real,  # noqa: E402
                     run, simulate, valuation_dev)
from buyrent.model import annuity_payment  # noqa: E402
from buyrent.montecarlo import nominal  # noqa: E402

# 与 analysis/china_2026.py 完全相同的 2026 共同参数与城市画像
COMMON = dict(down=0.30, mort_rate=0.031, mort_years=30, buy_cost=0.025,
              sell_cost=0.015, carry=0.007, infl=0.005, g_rent=0.005,
              r_invest=0.02, hold=10)
CITIES = [
    ("一线城市(京沪广深)", 0.018, 0.024),
    ("强二线(杭州苏州)", 0.020, 0.024),
    ("二线(成都武汉等)", 0.024, 0.028),
    ("三四线(人口流出)", 0.022, 0.030),
]
R_CONS = 0.015          # 保守投资者真实收益率
SELL_COST_SHRINKING = 0.05   # 人口流出城市的卖出摩擦
JAPAN_G = -0.02         # 日本 1990 年起十年真实房价年化约 -2.2%, 取整为 -2%

# 2021 年顶部的一线城市: 租金收益率约 1.6%, 首套房贷约 5.3%, 契税+中介约 3%,
# 卖出(增值税/个税未必满免)约 2%, 通胀约 2%, 理财收益 4–5%(取 4.5%)。
SCENARIO_2021 = dict(rent_yield=0.016, ry_typical=0.024, down=0.30,
                     mort_rate=0.053, mort_years=30, buy_cost=0.03,
                     sell_cost=0.02, carry=0.007, infl=0.02, g_rent=0.02,
                     r_invest=0.045, hold=10)


def gap_under(s: Scenario, g_real: float) -> float:
    """给定真实房价年涨幅下的十年财富差(占房价, 真实口径)。"""
    out = simulate(replace(s, g_house=nominal(g_real, s.infl)))
    return round(out["gap_real"] / s.price, 4)


def equal_weight_win_rate(s: Scenario, hist: History, tercile: int) -> dict:
    """窗口等权 vs 国家等权的胜率。"""
    pool = hist.pool(s.hold, tercile)
    wins, isos = [], []
    for row in pool.itertuples(index=False):
        out = simulate(replace(
            s, g_house=nominal(row.g_house, 0.005), g_rent=nominal(row.g_rent, 0.005),
            r_invest=nominal(R_CONS, 0.005), infl=0.005))
        wins.append(out["gap"] > 0)
        isos.append(row.iso)
    wins, isos = np.array(wins), np.array(isos)
    by_country = {c: float(wins[isos == c].mean()) for c in np.unique(isos)}
    return {"window_weighted": round(float(wins.mean()), 4),
            "country_weighted": round(float(np.mean(list(by_country.values()))), 4),
            "by_country": {c: round(v, 3) for c, v in by_country.items()},
            "n_countries": len(by_country)}


def main():
    hist = History()
    med10 = float(np.median(hist.pool(10).g_house))
    out = {"median_g10_real": round(med10, 5), "japan_path_g_real": JAPAN_G,
           "cities": {}}

    for name, ry, ry_typ in CITIES:
        s = Scenario(rent_yield=ry, **COMMON)
        tc = hist.tercile_of(valuation_dev(ry, ry_typ), s.hold)
        row = {"ry": ry, "tercile": tc,
               "gap_flat": gap_under(s, 0.0),
               "gap_japan": gap_under(s, JAPAN_G),
               "gap_median": gap_under(s, med10)}
        s5 = replace(s, sell_cost=SELL_COST_SHRINKING)
        row["sell_cost_5pct"] = {"gap_flat": gap_under(s5, 0.0),
                                 "gap_japan": gap_under(s5, JAPAN_G)}
        # 浮动利率修正: 固定通胀(正文口径) vs 照搬历史通胀
        p_float = run(s, hist, tercile=tc, r_invest_real=R_CONS,
                      infl_fixed=0.005, n_boot=0)["p_buy_wins"]
        p_hist_infl = run(s, hist, tercile=tc, r_invest_real=R_CONS,
                          n_boot=0)["p_buy_wins"]
        row["p_buy_float"] = round(p_float, 4)
        row["p_buy_if_historical_inflation"] = round(p_hist_infl, 4)
        row["hedge_uplift_pp"] = round((p_hist_infl - p_float) * 100, 1)
        row["weighting"] = equal_weight_win_rate(s, hist, tc)
        out["cities"][name] = row

    uplifts = [c["hedge_uplift_pp"] for c in out["cities"].values()]
    out["hedge_uplift_pp_range"] = [min(uplifts), max(uplifts)]

    # 一线: 月供是租金的几倍
    s1 = Scenario(rent_yield=CITIES[0][1], **COMMON)
    pay = annuity_payment((1 - s1.down) * s1.price, s1.mort_rate, s1.mort_years)
    out["tier1_payment_to_rent"] = round(pay / (s1.rent_yield * s1.price), 2)

    # 2021 年顶部对照
    p21 = dict(SCENARIO_2021)
    ry_typ21 = p21.pop("ry_typical")
    s21 = Scenario(**p21)
    tc21 = hist.tercile_of(valuation_dev(s21.rent_yield, ry_typ21), s21.hold)
    g21 = breakeven_growth_real(s21)
    s26 = Scenario(rent_yield=CITIES[0][1], **COMMON)
    tc26 = hist.tercile_of(valuation_dev(CITIES[0][1], CITIES[0][2]), s26.hold)
    g26 = breakeven_growth_real(s26)
    out["tier1_2021_vs_2026"] = {
        "assumptions_2021": SCENARIO_2021,
        "2021": {"g_star_real": round(g21, 4), "tercile": tc21,
                 "p_hist_cond": round(hist.prob_exceed(g21, 10, tc21)[0], 4)},
        "2026": {"g_star_real": round(g26, 4), "tercile": tc26,
                 "p_hist_cond": round(hist.prob_exceed(g26, 10, tc26)[0], 4)},
    }

    # g* 弹性(二线基准): 每个因素动 1 个百分点(或持有期换档), g* 动多少
    b = Scenario(rent_yield=CITIES[2][1], **COMMON)
    g0 = breakeven_growth_real(b)
    d = lambda s_: round((breakeven_growth_real(s_) - g0) * 100, 2)  # noqa: E731
    out["g_star_elasticity_pp"] = {
        "baseline_g_star_real": round(g0, 4),
        "rent_yield_+1pp": d(replace(b, rent_yield=b.rent_yield + 0.01)),
        "mort_rate_+1pp": d(replace(b, mort_rate=b.mort_rate + 0.01)),
        "r_invest_+1pp": d(replace(b, r_invest=nominal(R_CONS + 0.01, b.infl))),
        "g_rent_+1pp": d(replace(b, g_rent=b.g_rent + 0.01)),
        "hold_10_to_5": d(replace(b, hold=5)),
        "hold_10_to_20": d(replace(b, hold=20)),
    }

    # "不靠涨价也划算"的分水岭: 使 g* 恰好等于历史中位涨幅的租金收益率。
    # 论文第 6 节的经验法则(按揭 4.5%、默认成本结构)约 4.5%; 中国 2026 口径
    # (按揭 3.1%、中国成本结构、租方收益真实 1.5%)下门槛更低——README 引用两者。
    from scipy.optimize import brentq
    def yield_at_median(base: Scenario) -> float:
        f = lambda ry: breakeven_growth_real(replace(base, rent_yield=ry)) - med10  # noqa: E731
        return round(float(brentq(f, 0.005, 0.15, xtol=1e-6)), 4)
    out["breakeven_yield_at_median_g"] = {
        "paper_rule_of_thumb": yield_at_median(Scenario(hold=10)),   # 按揭 4.5%, 默认成本
        "china_2026": yield_at_median(Scenario(rent_yield=0.024, **COMMON)),
    }

    with open(ROOT / "data" / "derived" / "stress.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nderived ->", ROOT / "data" / "derived" / "stress.json")


if __name__ == "__main__":
    main()
