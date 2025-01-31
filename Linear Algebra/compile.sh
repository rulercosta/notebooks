#!/bin/bash
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
# Clean up auxiliary files if desired
rm -f *.aux *.log *.toc *.out *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
