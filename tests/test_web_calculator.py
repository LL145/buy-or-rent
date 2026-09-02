"""网页计算器与 Python 必须给出同一个答案。

web/calculator.html 把模型重新实现了一遍(JavaScript, 好让不装 Python 的人也能用)。
两份实现一旦分叉, 页面就会安静地给出与论文不同的结论——比没有页面更糟。
这里把页面里的算法抽出来在 node 里跑, 与 buyrent 逐场景对照。
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from buyrent import History, Scenario, breakeven_growth, run
from buyrent.montecarlo import nominal

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "verify_web.mjs"
PAGE = ROOT / "web" / "calculator.html"
STANDALONE = ROOT / "web" / "index.html"

SCENARIOS = [
    # 一线画像(高估组), 中国式参数
    dict(rent_yield=0.018, down=0.30, mort_rate=0.031, mort_years=30, hold=10,
         buy_cost=0.025, sell_cost=0.015, carry=0.007, r_invest_real=0.015,
         infl_fixed=0.005, tercile=2),
    # 二线画像(高估组), 更长持有期
    dict(rent_yield=0.024, down=0.30, mort_rate=0.031, mort_years=30, hold=20,
         buy_cost=0.025, sell_cost=0.015, carry=0.007, r_invest_real=0.030,
         infl_fixed=0.005, tercile=2),
    # 不做估值分组, 短持有期(会 snap 到 5 年窗口)
    dict(rent_yield=0.030, down=0.20, mort_rate=0.045, mort_years=25, hold=3,
         buy_cost=0.03, sell_cost=0.04, carry=0.012, r_invest_real=0.02,
         infl_fixed=0.02, tercile=None),
    # 高租售比 + 低估组
    dict(rent_yield=0.055, down=0.40, mort_rate=0.065, mort_years=30, hold=15,
         buy_cost=0.03, sell_cost=0.04, carry=0.012, r_invest_real=0.035,
         infl_fixed=0.02, tercile=0),
    # 固定利率口径(论文第7节画像 B 的美式市场): 通胀与股六债四回报取窗口联合值,
    # 对应 run() 的默认路径; infl_fixed 此时只用于把 g* 折成真实值
    dict(rent_yield=0.055, down=0.30, mort_rate=0.065, mort_years=30, hold=10,
         buy_cost=0.03, sell_cost=0.04, carry=0.012, r_invest_real=0.039,
         infl_fixed=0.02, tercile=1, fixed_rate=True),
    # 固定利率 + 不分组 + 取整到 20 年窗口
    dict(rent_yield=0.030, down=0.20, mort_rate=0.050, mort_years=30, hold=25,
         buy_cost=0.03, sell_cost=0.04, carry=0.012, r_invest_real=0.030,
         infl_fixed=0.03, tercile=None, fixed_rate=True),
]

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not PAGE.exists(),
    reason="需要 node 与已构建的 web/calculator.html (python3 analysis/build_web.py)")


def python_side(c):
    s = Scenario(
        rent_yield=c["rent_yield"], down=c["down"], mort_rate=c["mort_rate"],
        mort_years=c["mort_years"], hold=c["hold"], buy_cost=c["buy_cost"],
        sell_cost=c["sell_cost"], carry=c["carry"], infl=c["infl_fixed"],
        g_rent=c["infl_fixed"], r_invest=nominal(c["r_invest_real"], c["infl_fixed"]))
    g_nom = breakeven_growth(s)
    g_real = (1 + g_nom) / (1 + c["infl_fixed"]) - 1
    hist = History()
    p_hist, n_hist = hist.prob_exceed(g_real, c["hold"], c["tercile"])
    if c.get("fixed_rate"):
        mc = run(s, hist, tercile=c["tercile"], n_boot=0)
    else:
        mc = run(s, hist, tercile=c["tercile"], r_invest_real=c["r_invest_real"],
                 infl_fixed=c["infl_fixed"], n_boot=0)
    q = mc["gap_real_pct_of_price"]
    return dict(g_star_real=g_real, p_hist=p_hist, n_hist=n_hist,
                p_buy_wins=mc["p_buy_wins"], n_windows=mc["n_windows"],
                gap_p5=q["p5"], gap_median=q["median"], gap_p95=q["p95"])


@pytest.fixture(scope="module")
def js_results():
    proc = subprocess.run(["node", str(HARNESS)], input=json.dumps(SCENARIOS),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"node 运行失败:\n{proc.stderr}"
    return json.loads(proc.stdout)


@pytest.mark.parametrize("i", range(len(SCENARIOS)))
def test_js_matches_python(js_results, i):
    js, py = js_results[i], python_side(SCENARIOS[i])
    # 窗口数必须完全相同 —— 差一个就说明两边的取样口径分叉了
    assert js["n_hist"] == py["n_hist"], "历史频率的窗口集不一致"
    assert js["n_windows"] == py["n_windows"], "全量求值的窗口集不一致"
    for k in ("g_star_real", "p_hist", "p_buy_wins",
              "gap_p5", "gap_median", "gap_p95"):
        assert js[k] == pytest.approx(py[k], abs=1e-9), (
            f"场景 {i} 的 {k} 不一致: JS {js[k]} vs Python {py[k]}")


def test_js_ci_brackets_point(js_results):
    """两边用不同的伪随机数, 区间不会逐位相同; 但必须包住点估计且宽度合理。"""
    for js in js_results:
        lo, hi = js["p_buy_ci"]
        assert lo <= js["p_buy_wins"] <= hi
        assert 0.0 < hi - lo < 0.35


def test_page_is_built_from_current_data():
    """页面里嵌的窗口数必须与 windows.csv 当前的行数一致。"""
    import pandas as pd
    html = PAGE.read_text(encoding="utf-8")
    start = html.index('<script id="windowData" type="application/json">')
    blob = html[start:].split(">", 1)[1].split("</script>")[0]
    data = json.loads(blob)
    win = pd.read_csv(ROOT / "data" / "derived" / "windows.csv").dropna(
        subset=["g_house"])
    embedded = sum(len(v["iso"]) for v in data["horizons"].values())
    assert embedded == len(win), "web/calculator.html 已过期, 请重跑 analysis/build_web.py"


def test_embedded_medians_match_key_stats():
    """页面判断"打平是否要求跑赢历史"用的中位涨幅不再手写, 必须等于派生数据。"""
    html = PAGE.read_text(encoding="utf-8")
    start = html.index('<script id="windowData" type="application/json">')
    data = json.loads(html[start:].split(">", 1)[1].split("</script>")[0])
    ks = json.load(open(ROOT / "data" / "derived" / "key_stats.json", encoding="utf-8"))
    assert set(data["median_g"]) == {"5", "10", "15", "20"}
    assert round(data["median_g"]["10"], 5) == ks["g10_quantiles"]["50"]
    assert len(data["countries"]) == ks["n_countries_windows"]


def test_standalone_document_is_publishable():
    """web/index.html 要能作为独立网页发布(GitHub Pages), 片段版不能。

    缺 charset 的页面靠浏览器嗅探编码, 整页中文可能变乱码;
    缺 viewport 的页面在手机上按桌面宽度渲染再整体缩小。
    """
    html = STANDALONE.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    head = html[:html.index("</head>")]
    assert '<meta charset="utf-8">' in head
    assert 'name="viewport"' in head and "width=device-width" in head
    assert "<title>" in head and 'lang="zh-CN"' in html
    # 片段版反过来不能带这些标签——Artifact 发布时会自己包一层
    frag = PAGE.read_text(encoding="utf-8")
    for tag in ("<!doctype", "<html", "<head>", "<body>"):
        assert tag not in frag.lower(), f"片段版不应包含 {tag}"


def test_both_outputs_share_one_source():
    """两份产物必须来自同一模板: 内容体与嵌入数据完全一致。"""
    frag = PAGE.read_text(encoding="utf-8")
    body = STANDALONE.read_text(encoding="utf-8")
    marker = '<script id="windowData" type="application/json">'
    assert frag[frag.index(marker):] == body[body.index(marker):].replace(
        "\n</body>\n</html>\n", "\n")
