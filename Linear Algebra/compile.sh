#!/bin/bash
pdflatex -shell-escape LinearAlgebra.tex
pdflatex -shell-escape LinearAlgebra.tex
# Clean up auxiliary files if desired
rm -rf *.aux *.log *.toc *.out *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz svg-inkscape
