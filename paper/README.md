# The paper

Draft of the DTU 02805 paper in the PNAS template the course asks for. It lives in Overleaf as
"Log-Log Legends - 02805 paper (PNAS)" (made from the course's template link); the files here are the
sources we upload to it.

- `main.tex`: the paper. Lines marked TODO need the group (author contributions, dates, hand-label results, repo link).
- `references.bib`: every reference. The `checked` field says whether the full text (`read`) or only the
  abstract was checked. BibTeX ignores it. `archive/docs/overview/literature.pdf` lists the same entries grouped by strand.
- `figures/`: the paper uses four figures. Three are copies of figures an analysis script draws; after rerunning
  one, copy its figure over the file here (paths are from the repo root):

| In the paper | Copied from | Drawn by |
|---|---|---|
| `fig_genre.pdf` | `src/analysis/genre/outputs/figures/q1_genre.pdf` | `python -m analysis.genre.q1_genre`, run from `src/` |
| `fig_people.png` | `src/analysis/network/figures/q2_network.png` | `src/scripts/network/q2_network.py` |
| `fig_script_similarity.pdf` | `src/analysis/screenplays/figures/fig_script_similarity.pdf` | `src/analysis/screenplays/figures/make_similarity_figure.py` |
| `fig_head_to_head.pdf` | (drawn here) | `make_figures.py` in this folder |

  `make_figures.py` also draws `fig_network.pdf`, `fig_script.pdf` and `fig_reception.pdf`, and
  `fig_script_vectors.pdf` is a copy of `src/analysis/screenplays/figures/fig1_signal_vs_null.pdf`. Those four are
  from earlier drafts and the paper doesn't use them. To redraw the head-to-head figure, from the repo root:

```bash
python paper/make_figures.py
```

`src/tests/test_paper_numbers.py` checks the numbers in `main.tex` against the committed outputs. The reception
correlation in the discussion (ρ ≈ 0.3) comes from `analysis/reviews/outputs/`, which is only on the unmerged branch
`reviews-exploration`: the test reads it from that branch and skips those numbers when no ref to it exists.

The PNAS class files (`pnas-new.cls` and friends) come with the Overleaf template and are not copied here.
