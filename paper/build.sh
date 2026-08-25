#!/usr/bin/env bash
# 编译论文 PDF。依赖: texlive-xetex texlive-lang-chinese texlive-latex-recommended
set -euo pipefail
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error paper.tex
echo "OK -> paper/paper.pdf"
