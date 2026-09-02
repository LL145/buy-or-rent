"""论文、README 与派生数据的一致性。

CLAUDE.md 的工作约定要求"改了口径就同步更新四处数字"。靠人记不住——
这里把它变成测试: 每个断言都用 data/derived/*.json 里的值**构造**出应当出现
在文中的字符串, 再断言它确实出现。数据变了而文档没跟上, 测试立刻失败。

历史教训: 第 7 节案例表曾把 g* 的 +0.95% 写成 +1.1%, 因为表是手抄的。
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def load(name):
    with open(DERIVED / name, encoding="utf-8") as f:
        return json.load(f)


def norm(text: str) -> str:
    """剥掉排版标记, 只留下数字与文字本身。

    同一个数字在三份文档里写法不同: Markdown 的 `**粗体**`、LaTeX 的
    `\\textbf{}` 与转义的 `\\%`、以及 U+2212 与 ASCII 两种负号。
    比较的是内容, 不是排版。
    """
    t = text.replace("−", "-").replace("–", "-").replace("\\%", "%")
    for tok in ("\\textbf", "\\mathbf", "\\scriptsize", "\\texttt"):
        t = t.replace(tok, "")
    for ch in ("**", "$", "{", "}"):
        t = t.replace(ch, "")
    return re.sub(r"\s+", " ", t)


@pytest.fixture(scope="module")
def docs():
    return {p.name: norm((ROOT / p).read_text(encoding="utf-8"))
            for p in (Path("paper/paper.md"), Path("paper/paper.tex"),
                      Path("README.md"))}


def must_contain(docs, filename, snippet, why=""):
    assert norm(snippet) in docs[filename], (
        f"{filename} 与派生数据不一致{': ' + why if why else ''}\n"
        f"  期望出现: {norm(snippet)!r}\n"
        f"  → 重跑对应的 analysis/ 脚本后同步更新文档")


# ------------------------------------------------------------------ 样本量
def test_country_counts(docs):
    ks = load("key_stats.json")
    assert ks["n_countries"] == 18
    assert ks["n_countries_windows"] == 16
    assert ks["n_countries_hp_panel"] == 13
    assert ks["n_countries_by_era"]["pre1950"] == 12
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, "16 国", "窗口类结论的国家数")
        must_contain(docs, f, "13 国", "事实一房价面板的国家数")
        must_contain(docs, f, "12 国", "1870-1950 中位数的国家数")


def test_no_stale_18_country_window_claim(docs):
    """窗口/历史频率的经验基础是 16 国, 不得再写成 18 国。"""
    for f in ("paper.md", "paper.tex"):
        assert "150 年 x 18 国" not in docs[f].replace("$\\times$", "x")
        assert "18 国的历史中出现过" not in docs[f]


# ------------------------------------------------------- 事实一/二的关键数字
def test_stylized_facts_numbers(docs):
    ks = load("key_stats.json")
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, f"+{ks['real_hp_growth_pre1950'] * 100:.1f}%")
        must_contain(docs, f, f"+{ks['real_hp_growth_post1950'] * 100:.1f}%")
        must_contain(docs, f, f"+{ks['real_hp_growth_full'] * 100:.1f}%")


# ------------------------------------------------------------ 均值回归 t 值
def test_mean_reversion_t_values(docs):
    mr = load("uncertainty.json")["mean_reversion"]
    for k in ("5", "10"):
        for method in ("ols", "country", "year0", "twoway"):
            t = mr[k][method]["t"]
            assert abs(t) > 1.0, "t 值异常, 推断脚本可能坏了"
    # 论文以双向聚类为主口径, 两个持有期的 t 都必须出现在文中
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, f"{mr['10']['twoway']['t']:.2f}")
        must_contain(docs, f, f"{mr['5']['twoway']['t']:.2f}")
        must_contain(docs, f, f"{mr['10']['ols']['t']:.2f}", "朴素 OLS 的对照值")


def test_mean_reversion_conclusion_still_holds():
    """论文的定性表述: 10年勉强显著、5年不显著。数据变了就该改表述。"""
    mr = load("uncertainty.json")["mean_reversion"]
    assert 0.01 < mr["10"]["twoway"]["p_two_sided"] < 0.10
    assert mr["5"]["twoway"]["p_two_sided"] > 0.10
    assert abs(mr["10"]["ols"]["t"]) > 2 * abs(mr["10"]["twoway"]["t"]) * 0.9


# --------------------------------------------------- 事实四: 频率表与区间行
def test_prob_exceed_table_row(docs):
    pe = load("uncertainty.json")["g10_prob_exceed_ci"]
    ts = (0, 1, 2, 3, 4, 5)
    freq = " | ".join(f"{pe[f'{t}%']['point']:.0%}" for t in ts)
    ci = " | ".join(f"[{pe[f'{t}%']['ci95'][0] * 100:.0f}, "
                    f"{pe[f'{t}%']['ci95'][1] * 100:.0f}]" for t in ts)
    must_contain(docs, "paper.md", f"| 历史频率 | {freq} |")
    must_contain(docs, "paper.md", f"| 95% 区间 | {ci} |")


def test_tercile_contrast_is_reported_as_marginal(docs):
    c = load("uncertainty.json")["g10_by_tercile_ci"]["contrast_dear_minus_cheap"]
    p = c["p_le_0"]
    assert p["ci95"][0] < 0 < p["ci95"][1], "区间不再跨零, 论文的弱化表述需重写"
    lo, hi = p["ci95"]
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, f"{p['point'] * 100:.1f} 个百分点")
        must_contain(docs, f, f"[{lo * 100:+.1f}, {hi * 100:+.1f}]")


# ---------------------------------------------------------- 第7节 案例表
def test_case_table_rows(docs):
    c = load("cases.json")
    K = ["A_low_yield_metro", "B_high_yield_market", "C_short_hold"]

    def ci(v):
        return f"[{v[0] * 100:.0f}, {v[1] * 100:.0f}]"

    row = " | ".join(f"{c[k]['p_hist_uncond']:.0%} {ci(c[k]['p_hist_uncond_ci95'])}"
                     for k in K)
    must_contain(docs, "paper.md", f"| 历史频率(无条件) | {row} |")
    for k in K:
        d = c[k]
        must_contain(docs, "paper.md", f"{d['p_buy_wins']:.0%} {ci(d['p_buy_wins_ci95'])}",
                     f"{d['label']} 的胜率与区间")
        must_contain(docs, "paper.md", f"{d['gap_median']:+.0%}",
                     f"{d['label']} 的财富差中位")


def test_case_b_breakeven_matches_code(docs):
    """曾把 +0.95% 写成 +1.1% —— 正是这个测试要防的漂移。"""
    g = load("cases.json")["B_high_yield_market"]["g_star_real"]
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, f"+{g * 100:.1f}%/年")


# --------------------------------------------------------- 第8节 中国案例
def test_china_table_rows(docs):
    rows = load("china_2026.json")

    def ci(v):
        return f"[{v[0] * 100:.0f}, {v[1] * 100:.0f}]"

    for key, label in [("p_buy_conservative", "保守"), ("p_buy_balanced", "均衡")]:
        cells = " | ".join(f"{r[key]:.0%} {ci(r[key + '_ci95'])}" for r in rows[:3])
        assert cells in docs["paper.md"], (
            f"第8.2节 {label}投资者行与 china_2026.json 不一致\n  期望: {cells}")


def test_china_valuation_assumption_sensitivity(docs):
    """各城"常态租金收益率"是假设; 论文必须报告翻转门槛与换组后的胜率。"""
    rows = load("china_2026.json")

    def ci(v):
        return f"[{v[0] * 100:.0f}, {v[1] * 100:.0f}]"

    for r in rows:
        s = r["tercile_sensitivity"]
        # 门槛的定义: 常态低于 ry·exp(q2) 即落入中间组
        assert r["ry_typical_to_mid"] < r["ry_typical"], "该城本应在高估组"
        # 放松高估假设只会利好买方——论文据此说正文口径是"对买方最保守"的一端
        assert s["mid"]["p_buy"] > r["p_buy_conservative"]
        assert s["uncond"]["p_buy"] > r["p_buy_conservative"]
        for f in ("paper.md", "paper.tex"):
            must_contain(docs, f, f"{r['ry_typical_to_mid'] * 100:.2f}%", f"{r['city']} 的门槛")
            must_contain(docs, f, f"{s['mid']['p_buy']:.0%} {ci(s['mid']['p_buy_ci95'])}",
                         f"{r['city']} 中间组胜率")
            must_contain(docs, f, f"{s['uncond']['p_buy']:.0%} {ci(s['uncond']['p_buy_ci95'])}",
                         f"{r['city']} 不分组胜率")
    # 论文的论断: 二线卡在边界上(与门槛相差不到 25 个基点), 换组后仍不越过 55%
    t2 = rows[2]
    assert t2["ry_typical"] - t2["ry_typical_to_mid"] < 0.0025
    assert t2["tercile_sensitivity"]["mid"]["p_buy"] < 0.55
    bp = round((t2["ry_typical"] - t2["ry_typical_to_mid"]) * 10000)
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, f"只有 {bp} 个基点")


def test_china_tier_intervals_overlap():
    """论文据此说"支持梯度而非排序"。若不再重叠, 该表述需重写。"""
    rows = load("china_2026.json")
    first, second = rows[0]["p_buy_conservative_ci95"], rows[2]["p_buy_conservative_ci95"]
    assert first[1] > second[0], "一线与二线的区间不再重叠, 第8.2节读表须知需重写"


# ------------------------------------------------------------ 8.4 速查表
def test_lookup_table_cells(docs):
    g = load("lookup_table.json")
    for ry in (0.015, 0.020, 0.025, 0.030, 0.040, 0.050):
        for h in (3, 5, 10, 15, 20):
            c = g[f"{h}|{ry}"]
            cell = f"({c['p_cons'] * 100:.0f}/{c['p_bal'] * 100:.0f})"
            assert cell in docs["paper.md"], (
                f"8.4 速查表缺少格子 ry={ry:.1%} hold={h}: {cell}")


# ---------------------------------------------------------------- README
def test_readme_headline_numbers(docs):
    u = load("uncertainty.json")
    pe = u["g10_prob_exceed_ci"]
    for t in ("2%", "4%"):
        d = pe[t]
        must_contain(docs, "README.md", f"{d['point']:.0%}")
        must_contain(docs, "README.md",
                     f"[{d['ci95'][0] * 100:.0f}, {d['ci95'][1] * 100:.0f}]")
    must_contain(docs, "README.md", f"{u['mean_reversion']['10']['twoway']['t']:.2f}")
    must_contain(docs, "README.md", "16 国")
    # README 里对二线估值假设敏感性的一句话概括
    t2 = load("china_2026.json")[2]
    bp = round((t2["ry_typical"] - t2["ry_typical_to_mid"]) * 10000)
    must_contain(docs, "README.md", f"只有 {bp} 个基点")
    must_contain(docs, "README.md",
                 f"换组后胜率 {t2['tercile_sensitivity']['mid']['p_buy']:.0%}")


# --------------------------------------------------- 体制稳健性(局限第1条)
def test_regime_robustness_claims(docs):
    """局限第1条的论断: 高估组对样本期不敏感, 中/低估值组敏感。"""
    c = load("cases.json")
    a, b = c["A_low_yield_metro"], c["B_high_yield_market"]
    # 画像A(高估组): 全样本与战后样本的胜率差应很小
    assert abs(a["post1950"]["p_buy_wins"] - a["p_buy_wins"]) <= 0.03
    # 画像B(中间组): 差应明显更大
    assert b["post1950"]["p_buy_wins"] - b["p_buy_wins"] > 0.05
    # 无条件频率整体右移
    assert a["post1950"]["p_hist_uncond"] > a["p_hist_uncond"] + 0.03
    for f in ("paper.md", "paper.tex"):
        must_contain(docs, f, f"{a['p_hist_uncond']:.0%}")
        must_contain(docs, f, f"{a['post1950']['p_hist_uncond']:.0%}")
        must_contain(docs, f, f"{b['post1950']['p_buy_wins']:.0%}")


# ------------------------------------------- 特异性风险敏感性(局限第5条)
def test_idio_sensitivity_table(docs):
    c = load("cases.json")
    a = c["A_low_yield_metro"]["idio_sensitivity"]
    b = c["B_high_yield_market"]["idio_sensitivity"]
    # 论文的论断: A 的胜率随离散度上升, 中位数不动
    assert a["0.15"]["p_buy_wins"] > a["0.00"]["p_buy_wins"] + 0.05
    assert abs(a["0.15"]["gap_median"] - a["0.00"]["gap_median"]) < 0.02
    assert a["0.15"]["gap_p5"] < a["0.00"]["gap_p5"]
    for f in ("paper.md", "paper.tex"):
        for sd in ("0.00", "0.05", "0.10", "0.15"):
            must_contain(docs, f, f"{a[sd]['p_buy_wins']:.0%}", f"画像A idio={sd}")
            must_contain(docs, f, f"{b[sd]['gap_p5']:+.0%}", f"画像B p5 idio={sd}")
