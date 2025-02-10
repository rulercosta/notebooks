#!/bin/bash
pdflatex -shell-escape LiquefactionOfSoil.tex
pdflatex -shell-escape LiquefactionOfSoil.tex
# Clean up auxiliary files if desired
rm -rf *.aux *.log *.toc *.out *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz svg-inkscape
