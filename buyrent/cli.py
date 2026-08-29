"""命令行入口: 输入 A 层可观察变量, 输出概率化的买租判断。

示例(一线城市画像: 租金收益率1.8%, 历史常态2.2%, 按揭4.0%, 持有10年):
    python3 -m buyrent.cli --rent-yield 0.018 --ry-typical 0.022 \
        --mort-rate 0.040 --down 0.30 --hold 10
"""
import argparse

from .breakeven import breakeven_growth, breakeven_growth_real
from .history import History, TERCILE_NAMES, snap_horizon, valuation_dev
from .model import Scenario, simulate
from .montecarlo import run


def main():
    ap = argparse.ArgumentParser(description="买房还是租房: 概率化估算")
    ap.add_argument("--rent-yield", type=float, required=True,
                    help="当前租金收益率 = 年租金/房价, 如 0.02")
    ap.add_argument("--ry-typical", type=float, default=None,
                    help="本市过去~20年的典型租金收益率(用于估值条件化, 可省略)")
    ap.add_argument("--down", type=float, default=0.30, help="首付比例(默认0.30)")
    ap.add_argument("--mort-rate", type=float, default=0.045, help="按揭利率")
    ap.add_argument("--mort-years", type=int, default=30, help="按揭年限")
    ap.add_argument("--hold", type=int, default=10, help="预期持有年数")
    ap.add_argument("--buy-cost", type=float, default=0.03, help="购房交易成本率")
    ap.add_argument("--sell-cost", type=float, default=0.04, help="卖房交易成本率")
    ap.add_argument("--carry", type=float, default=0.012,
                    help="年持有成本率(维护+物业+持有税费)")
    ap.add_argument("--infl", type=float, default=0.02, help="通胀假设(仅报告用)")
    ap.add_argument("--eq-share", type=float, default=0.6,
                    help="租方替代资产中股票占比(默认0.6, 其余国债)")
    ap.add_argument("--r-invest-real", type=float, default=None,
                    help="固定租方真实投资收益率(如 0.015), 替代历史抽样; "
                         "用于可投资渠道受限的市场")
    ap.add_argument("--infl-fixed", type=float, default=None,
                    help="把蒙特卡洛的窗口通胀固定为该值(如 0.005); "
                         "浮动利率按揭市场(如中国LPR)应设置, 以关闭固定利率"
                         "按揭独有的通胀对冲通道")
    ap.add_argument("--since", type=int, default=None,
                    help="只用起始年 ≥ 该年份的历史窗口(如 1950)。默认全样本; "
                         "战后样本对买方更有利, 见论文第9节")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="整群自助重复次数(算置信区间用; 0=不算)")
    a = ap.parse_args()

    s = Scenario(rent_yield=a.rent_yield, down=a.down, mort_rate=a.mort_rate,
                 mort_years=a.mort_years, hold=a.hold, buy_cost=a.buy_cost,
                 sell_cost=a.sell_cost, carry=a.carry, infl=a.infl,
                 g_rent=a.infl, r_invest=0.06)
    hist = History(since=a.since)
    k = snap_horizon(a.hold)

    tercile = None
    if a.ry_typical:
        dev = valuation_dev(a.rent_yield, a.ry_typical)
        tercile = hist.tercile_of(dev, a.hold)

    g_star = breakeven_growth(s)
    g_star_real = breakeven_growth_real(s)
    p_uncond, n_uncond = hist.prob_exceed(g_star_real, a.hold)
    mc = run(s, hist, tercile=tercile, eq_share=a.eq_share,
             r_invest_real=a.r_invest_real, infl_fixed=a.infl_fixed,
             n_boot=a.n_boot)

    W = 62
    print("=" * W)
    print("买房还是租房 —— 概率化估算")
    print("=" * W)
    print(f"输入: 租金收益率 {a.rent_yield:.2%} | 首付 {a.down:.0%} | "
          f"按揭 {a.mort_rate:.2%}×{a.mort_years}年 | 持有 {a.hold} 年")
    print(f"      交易成本 买{a.buy_cost:.1%}/卖{a.sell_cost:.1%} | "
          f"持有成本 {a.carry:.2%}/年")
    print("-" * W)
    print(f"[1] 盈亏平衡涨幅 g*")
    print(f"    买房打平需要房价名义年涨 {g_star:+.2%} "
          f"(真实 {g_star_real:+.2%}), 并持续 {a.hold} 年")
    ci_u = hist.prob_exceed_ci(g_star_real, a.hold, n_boot=a.n_boot or 1)
    era = f"{a.since}–2020" if a.since else "1870–2020"
    print(f"[2] 历史频率 (JST 16国 {era}, {k}年窗口, N={n_uncond})")
    print(f"    真实年涨 ≥ {g_star_real:+.2%} 的窗口占比: {p_uncond:.0%}  "
          f"[{ci_u[0]:.0%}, {ci_u[1]:.0%}]")
    if tercile is not None:
        p_c, n_c = hist.prob_exceed(g_star_real, a.hold, tercile)
        ci_c = hist.prob_exceed_ci(g_star_real, a.hold, tercile,
                                   n_boot=a.n_boot or 1)
        print(f"    当前估值分组: {TERCILE_NAMES[tercile]} "
              f"(偏离 {dev:+.2f}) → 条件频率: {p_c:.0%} "
              f"[{ci_c[0]:.0%}, {ci_c[1]:.0%}] (N={n_c})")
    print(f"[3] 历史窗口全量求值 ({mc['n_windows']} 个窗口, "
          f"{mc['n_countries']} 国)")
    line = f"    P(买 优于 租) = {mc['p_buy_wins']:.0%}"
    if "p_buy_wins_ci95" in mc:
        lo, hi = mc["p_buy_wins_ci95"]
        line += f"  95%区间 [{lo:.0%}, {hi:.0%}]"
    print(line)
    q = mc["gap_real_pct_of_price"]
    print(f"    期末财富差(买−租, 占房价, 真实): 中位 {q['median']:+.0%}, "
          f"90%区间 [{q['p5']:+.0%}, {q['p95']:+.0%}]")
    print("-" * W)
    print("    方括号为95%区间(以国家为整群自助, 16个国家)——区间宽度说明"
          "这个胜率\n    只能读到十位数, 不要当作精确到个位的数字。")
    if a.hold < 5:
        print("提示: 持有期不足5年时, 交易成本摊薄几乎注定租房更优。")
    base = simulate(s)
    print(f"参考: 若房价名义年涨 {s.g_house:.1%}(租金随通胀), "
          f"财富差为 {base['gap_pct_of_price']:+.0%} 房价。")
    print("=" * W)


if __name__ == "__main__":
    main()
