#!/bin/bash
cd "$(dirname "$0")"

mkdir -p source/_static/figures

# Build TikZ figures
echo "Building TikZ figures..."
if [ -d "_mysource/figures/" ] && [ "$(ls -A _mysource/figures/*.tex 2>/dev/null)" ]; then
    cd _mysource/figures/
    for file in *.tex; do
        echo "Processing $file..."
        pdflatex -interaction=nonstopmode $file
        pdf_file="${file%.tex}.pdf"
        svg_file="${file%.tex}.svg"
        pdf2svg $pdf_file ../../source/_static/figures/$svg_file
    done
    cd ../../
else
    echo "No .tex files found in _mysource/figures/"
fi

# Remove previously generated files
echo "Cleaning up previously generated rst files..."
rm -rf source/generated

# Build the documentation
echo "Building documentation..."
make clean
make html

echo "Documentation build complete!"