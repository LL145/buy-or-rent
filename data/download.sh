#!/usr/bin/env bash
# 下载 Jordà-Schularick-Taylor Macrohistory Database R6 (含 RORE 回报序列)。
# 原始数据不入库(遵循其非商用+引用的使用条款), 派生统计量在 data/derived/。
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL -o JSTdatasetR6.xlsx \
  "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx"
ls -la JSTdatasetR6.xlsx
