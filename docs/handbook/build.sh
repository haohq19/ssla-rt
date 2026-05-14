#!/bin/bash
# Build the SSLA-RT Engineering Handbook PDF from handbook.tex.
# Run from this directory.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error handbook.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error handbook.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error handbook.tex >/dev/null
rm -f handbook.aux handbook.log handbook.out handbook.toc
echo "built handbook.pdf ($(pdfinfo handbook.pdf | grep -i pages | awk '{print $2}') pages)"
