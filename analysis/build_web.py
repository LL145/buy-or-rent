"""把历史窗口数据嵌进网页计算器, 生成单文件 web/calculator.html。

论文与 CLI 面向会读论文、会装 Python 的人; 这个页面面向要做决定的普通人——
项目目标里的"给当代的普通人一个参考"最终得靠它兑现。

页面在浏览器里跑完整算法(而非查预制表), 因此任何输入都能算, 且与 CLI 同源。
产出两份, 同一个模板:
  web/calculator.html — 片段(无 doctype/head), 供 Artifact 发布, 宿主会包一层
  web/index.html      — 独立文档, 供 GitHub Pages 与本地直接打开
片段少了 charset 与 viewport, 直接当网页发会导致中文乱码、移动端布局失效,
所以两者不能混用。

输入: web/calculator.template.html 中的 __WINDOW_DATA__ 占位符
用法: python3 analysis/build_web.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
COLS = ["g_house", "g_rent", "r_eq", "r_bond", "infl"]
DESC = ("输入房价、月租和打算住几年，用 1870 年以来 16 国的历史窗口算出"
        "买房划算的概率——每个结果都带 95% 置信区间。")


def payload() -> dict:
    """两个口径必须与 Python 完全一致, 否则网页与 CLI 会给出不同答案。

    History.prob_exceed 只要求 g_house 可得(第3层历史频率);
    History.pool 要求五列齐全(第4层全量求值)。因此这里导出前者的全部窗口,
    另附 `full` 标记指出哪些满足后者。
    """
    win = pd.read_csv(ROOT / "data" / "derived" / "windows.csv")
    win = win.dropna(subset=["g_house"])
    isos = sorted(win.iso.unique())
    imap = {c: i for i, c in enumerate(isos)}
    with open(ROOT / "data" / "derived" / "tercile_cutoffs.json") as f:
        cutoffs = json.load(f)
    out = {"countries": isos, "cutoffs": cutoffs, "horizons": {}}
    for h, sub in win.groupby("horizon"):
        full = sub[COLS].notna().all(axis=1)
        out["horizons"][str(int(h))] = {
            "iso": [imap[c] for c in sub.iso],
            # -1 = 该窗口起点缺少足够的租售比历史, 无法定估值分组
            "tercile": [-1 if pd.isna(t) else int(t) for t in sub.tercile],
            # 1 = 五列齐全, 可进入全量求值; 0 = 只能用于历史频率
            "full": [int(v) for v in full],
            # 不做四舍五入: 网页与 Python 要逐位相同, 否则边界窗口的输赢会翻面,
            # 胜率随之跳动 1/N。多出的约 300 KB 换一个可断言的等价性。
            **{c: [0.0 if pd.isna(v) else float(v) for v in sub[c]]
               for c in COLS},
        }
    return out


def standalone(fragment: str) -> str:
    """把片段包成完整 HTML 文档。

    charset 必不可少: 缺了它, 浏览器靠嗅探决定编码, 整页中文可能变成乱码。
    viewport 同理, 缺了移动端会按桌面宽度渲染再缩小。
    """
    head, body = fragment.split("<!--HEAD-END-->", 1)
    return (
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<meta name="description" content="{DESC}">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:title" content="买房还是租房">\n'
        f'<meta property="og:description" content="{DESC}">\n'
        f'{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n')


def main():
    data = json.dumps(payload(), separators=(",", ":"))
    tpl = (WEB / "calculator.template.html").read_text(encoding="utf-8")
    assert "__WINDOW_DATA__" in tpl, "模板缺少 __WINDOW_DATA__ 占位符"
    assert "<!--HEAD-END-->" in tpl, "模板缺少 <!--HEAD-END--> 分界标记"
    fragment = tpl.replace("__WINDOW_DATA__", data)
    (WEB / "calculator.html").write_text(fragment, encoding="utf-8")
    (WEB / "index.html").write_text(standalone(fragment), encoding="utf-8")
    n = sum(len(v["iso"]) for v in json.loads(data)["horizons"].values())
    print(f"嵌入 {n} 个历史窗口 ({len(data) / 1024:.0f} KB)")
    print("->", WEB / "calculator.html", "(片段, 供 Artifact)")
    print("->", WEB / "index.html", "(独立文档, 供 GitHub Pages)")


if __name__ == "__main__":
    main()
