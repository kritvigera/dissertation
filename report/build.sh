#!/bin/sh
# Build the submission report. Requires a TeX distribution on the PATH.
#
#   sh report/build.sh
#
# Tables in report/tables/ and figures in report/figures/ are generated from
# the committed data by report/make_assets.py, so no number in the report is
# typed by hand. Regenerate them first if the data tables have changed.
set -e
cd "$(dirname "$0")"
PATH="/Library/TeX/texbin:$PATH"; export PATH
pdflatex -interaction=nonstopmode report.tex > /dev/null
bibtex report > /dev/null
pdflatex -interaction=nonstopmode report.tex > /dev/null
pdflatex -interaction=nonstopmode report.tex | tail -3
