"""生成论文第 7 节的三个风格化画像, 使表中每个数字都由代码产出而非手抄。

输入:  data/derived/windows.csv, tercile_cutoffs.json
输出:  data/derived/cases.json
用法:  python3 analysis/cases.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
sys.path.insert(0, str(ROOT))
from buyrent import (History, Scenario, breakeven_growth,  # noqa: E402
                     breakeven_growth_real, run, valuation_dev)
from buyrent.history import snap_horizon  # noqa: E402

CASES = {
    "A_low_yield_metro": dict(
        label="A: 低租售比大城市", rent_yield=0.018, ry_typical=0.022,
        mort_rate=0.040, hold=10),
    "B_high_yield_market": dict(
        label="B: 高租售比市场", rent_yield=0.055, ry_typical=0.055,
        mort_rate=0.065, hold=10),
    "C_short_hold": dict(
        label="C: 短持有期", rent_yield=0.030, ry_typical=None,
        mort_rate=0.035, hold=3),
}


def evaluate(hist, rent_yield, mort_rate, hold, ry_typical=None, label="",
             infl=0.02):
    s = Scenario(rent_yield=rent_yield, mort_rate=mort_rate, hold=hold,
                 infl=infl, g_rent=infl, r_invest=0.06)
    tercile = dev = None
    if ry_typical:
        dev = valuation_dev(rent_yield, ry_typical)
        tercile = hist.tercile_of(dev, hold)

    g_star_real = breakeven_growth_real(s)
    p_u, n_u = hist.prob_exceed(g_star_real, hold)
    mc = run(s, hist, tercile=tercile)
    out = {
        "label": label, "rent_yield": rent_yield, "mort_rate": mort_rate,
        "hold": hold, "ry_typical": ry_typical, "horizon_used": snap_horizon(hold),
        "g_star_nominal": round(breakeven_growth(s), 5),
        "g_star_real": round(g_star_real, 5),
        "p_hist_uncond": round(p_u, 4),
        "p_hist_uncond_ci95": [round(v, 4) for v in
                               hist.prob_exceed_ci(g_star_real, hold)],
        "n_windows_uncond": n_u,
        "p_buy_wins": round(mc["p_buy_wins"], 4),
        "p_buy_wins_ci95": [round(v, 4) for v in mc["p_buy_wins_ci95"]],
        "gap_median": round(mc["gap_real_pct_of_price"]["median"], 4),
        "gap_p5": round(mc["gap_real_pct_of_price"]["p5"], 4),
        "gap_p95": round(mc["gap_real_pct_of_price"]["p95"], 4),
    }
    if tercile is not None:
        p_c, n_c = hist.prob_exceed(g_star_real, hold, tercile)
        out.update(tercile=int(tercile), dev=round(dev, 4),
                   p_hist_cond=round(p_c, 4),
                   p_hist_cond_ci95=[round(v, 4) for v in
                                     hist.prob_exceed_ci(g_star_real, hold, tercile)],
                   n_windows_cond=n_c)
    return out


def main():
    hist = History()
    out = {k: evaluate(hist, **v) for k, v in CASES.items()}

    # 体制稳健性(论文第9节第1条): 只用战后窗口时结论移动多少?
    # 战后房价涨幅整体更高, 但那些高涨幅多从"便宜"的估值起点出发——
    # 因此条件化之后, 样本期的选择在多大程度上还重要, 是个可以直接算的问题。
    post = History(since=1950)
    for k, v in CASES.items():
        d = evaluate(post, **v)
        out[k]["post1950"] = {
            "p_hist_uncond": d["p_hist_uncond"],
            "p_hist_cond": d.get("p_hist_cond"),
            "p_buy_wins": d["p_buy_wins"],
            "n_windows_uncond": d["n_windows_uncond"],
        }
    with open(DERIVED / "cases.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for k, c in out.items():
        print(f"{c['label']:20s} g*_real {c['g_star_real']:+.2%}  "
              f"hist {c['p_hist_uncond']:.0%}{c['p_hist_uncond_ci95']}  "
              f"cond {c.get('p_hist_cond', float('nan')):.0%}  "
              f"P(buy) {c['p_buy_wins']:.0%}{c['p_buy_wins_ci95']}  "
              f"gap {c['gap_median']:+.0%} [{c['gap_p5']:+.0%},{c['gap_p95']:+.0%}]")
        p = c["post1950"]
        print(f"{'':20s}   战后样本: 无条件 {p['p_hist_uncond']:.0%} "
              f"(全样本 {c['p_hist_uncond']:.0%}) | "
              f"条件 {p['p_hist_cond'] if p['p_hist_cond'] is None else format(p['p_hist_cond'], '.0%')} "
              f"(全样本 {c.get('p_hist_cond') if c.get('p_hist_cond') is None else format(c['p_hist_cond'], '.0%')}) | "
              f"胜率 {p['p_buy_wins']:.0%} (全样本 {c['p_buy_wins']:.0%})")
    print("\nderived ->", DERIVED / "cases.json")


if __name__ == "__main__":
    main()
