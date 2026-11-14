# The paper

Draft of the DTU 02805 paper in the PNAS template the course asks for. It lives in Overleaf as
"Log-Log Legends - 02805 paper (PNAS)" (made from the course's template link); the files here are the
sources we upload to it.

- `main.tex`: the paper. Lines marked TODO need the group (author contributions, dates, hand-label results, repo link).
- `references.bib`: every reference. The `checked` field says whether the full text (`read`) or only the
  abstract was checked. BibTeX ignores it. `docs/overview/literature.pdf` lists the same entries grouped by strand.
- `figures/`: drawn by `make_figures.py` from committed outputs only. Rebuild from the repo root:

```bash
python paper/make_figures.py
```

The PNAS class files (`pnas-new.cls` and friends) come with the Overleaf template and are not copied here.
