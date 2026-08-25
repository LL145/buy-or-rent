# 数据说明

## 原始数据（不入库，运行 `bash data/download.sh` 获取）

**Jordà-Schularick-Taylor Macrohistory Database, Release 6**（macrohistory.net）：
18 个发达经济体 1870–2020 年的房价指数、租金收益率、住房/股票/债券总回报、
利率、CPI、工资、GDP 等 59 个变量。仅限非商业用途，使用须引用：

- Jordà, Ò., M. Schularick, and A. M. Taylor (2017). "Macrofinancial History and
  the New Business Cycle Facts." *NBER Macroeconomics Annual* 31.
- Jordà, Ò., K. Knoll, D. Kuvshinov, M. Schularick, and A. M. Taylor (2019).
  "The Rate of Return on Everything, 1870–2015." *Quarterly Journal of
  Economics* 134(3)。（住房回报、租金收益率序列）
- Knoll, K., M. Schularick, and T. Steger (2017). "No Price Like Home: Global
  House Prices, 1870–2012." *American Economic Review* 107(2)。（房价序列）

## 派生数据（入库，由 `analysis/stylized_facts.py` 生成）

| 文件 | 内容 |
|---|---|
| `derived/windows.csv` | 所有 国家×起始年×持有期(5/10/15/20年) 窗口的年化实现值：真实房价涨幅 `g_house`、真实租金涨幅 `g_rent`、股/债/住房真实总回报 `r_eq/r_bond/r_house`、通胀 `infl`、期初估值偏离 `dev0`（log 租售比相对过去25年均值）、租金收益率 `ry0`、估值三分位 `tercile` |
| `derived/key_stats.json` | 论文引用的全部关键统计量 |
| `derived/tercile_cutoffs.json` | 各持有期的估值三分位切点（算法查询用） |

处理约定：恶性通胀年份（\|通胀\|>50%，如德国 1920–24）不进入回报统计；
窗口要求区间内数据完整。
