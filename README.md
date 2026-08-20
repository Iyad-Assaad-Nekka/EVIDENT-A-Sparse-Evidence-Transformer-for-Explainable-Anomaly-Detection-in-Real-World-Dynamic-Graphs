# Paper sources

`patch_table3_uci.tex` — drop-in blocks for the injected-benchmark
results. Seven numbered blocks, each labelled with where it goes:
the Table III row, the replacement caption, a new results subsection
with the density-matched ladder, the revised heterogeneity paragraph,
the Limitations text, an optional abstract sentence, and the preamble
macros. Nothing in it touches any figure or TikZ source.

`fig_cardinality.tex` — standalone TikZ figure showing that
fixed-cardinality resampling closes the neighbourhood-cardinality leak
channel. Compile with `pdflatex fig_cardinality.tex`, or `\input` it
inside a `figure` environment.

The main `.tex` is not committed here.
