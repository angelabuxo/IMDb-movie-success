"""Do headline results change when TMDB's budget and gross replace Wikipedia's?

Run from the repo root:  python scripts/imdb/tmdb_robustness.py
Needs data/raw/tmdb/tmdb_project_movies.csv and the inputs of the analyses below. Writes
analysis/checks/tmdb_robustness.json (aggregates only: this repo is public).

For each headline it does three runs, using each analysis's own code, not a copy:
  reproduce   the committed number on the original sample, with Wikipedia money (it must match exactly)
  wikipedia   the same analysis restricted to films where TMDB has both a budget and a gross
  tmdb        the same films, with TMDB's budget and gross in place of Wikipedia's everywhere money enters:
              the outcome, the log-budget control, and the hit and flop cut-offs, which are recomputed
Comparing the last two isolates the money source; comparing the first two shows the sample change alone.

Headlines covered:
  arcs Step 7 primary test (D1, D2), analysis/arcs/s07_primary.association
  scripts_and_reviews section 2: script features vs multiple, net of budget and year
  scripts_and_reviews section 8: review sentiment vs multiple, net of budget and year
Not covered: the network results. Their track-record predictors are built from earlier films' Wikipedia
returns, so swapping only the outcome would be half a swap; a fair test reruns scripts/network with TMDB money.

Assumptions: TMDB figures are US dollars, and a TMDB 0 means no figure.
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.arcs import s07_primary as s7  # noqa: E402
from analysis.arcs.common import pooled_cells  # noqa: E402
from analysis.arcs.config import cfg  # noqa: E402

DATA = Path(os.environ.get("IMDB_DATA_DIR", REPO / "data" / "raw"))
OUT = REPO / "analysis" / "checks" / "tmdb_robustness.json"
NOTEBOOK = REPO / "notebooks" / "scripts_and_reviews.ipynb"
CLEAN_LINE = "m = m[clean_mask(m)].copy()\n"


def tmdb_money() -> pd.DataFrame:
    t = pd.read_csv(DATA / "tmdb" / "tmdb_project_movies.csv", usecols=["imdb_id", "budget", "revenue"])
    return t[(t.budget > 0) & (t.revenue > 0)].drop_duplicates("imdb_id").set_index("imdb_id")


def _short(r: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
            if k in ("rho", "n", "z", "p", "ci_low", "ci_high")}


# ---------------------------------------------------------------- arcs Step 7

def arcs(t: pd.DataFrame) -> dict:
    c = cfg()
    pc, ctl = c["primary"], c["controls"]
    inst = json.loads((REPO / "analysis/arcs/outputs/s03_instrument_decision.json").read_text())["primary_instrument"]
    saved = json.loads((REPO / "analysis/arcs/outputs/s07_primary.json").read_text())["results"]

    def run(df):
        rng = np.random.default_rng(c["seed"])
        return {k: s7.association(df, k, pc["outcome"], ctl["covariates"], pc["permutations"], pc["bootstrap"], rng)
                for k in s7.DESCRIPTORS}

    df = s7.sample(inst)
    rep = run(df)
    for k in s7.DESCRIPTORS:
        assert abs(rep[k]["rho"] - saved[k]["rho"]) < 1e-12 and rep[k]["p"] == saved[k]["p"], f"arcs {k} doesn't reproduce"
    ids = pd.read_csv(DATA / "features" / "moviesum_ids.csv", dtype=str).set_index("movie_name").imdb_id
    sub = df.assign(tconst=df.movie_name.map(ids))
    sub = sub[sub.tconst.isin(t.index)].reset_index(drop=True)
    sub["cell"] = pooled_cells(sub.primary_genre, sub.decade_bin)
    tm = sub.assign(log_mult=np.log(sub.tconst.map(t.revenue) / sub.tconst.map(t.budget)),
                    log_budget=np.log(sub.tconst.map(t.budget)))
    wk, tk = run(sub), run(tm)
    return {"instrument": inst, "covariates": ctl["covariates"],
            "flops_wikipedia": int((sub.log_mult < 0).sum()), "flops_tmdb": int((tm.log_mult < 0).sum()),
            **{k: {"reproduce": _short(rep[k]), "wikipedia": _short(wk[k]), "tmdb": _short(tk[k])}
               for k in s7.DESCRIPTORS}}


# ---------------------------------------------------------------- scripts_and_reviews, sections 2 and 8

def _notebook_cells() -> dict[int, str]:
    nb = json.loads(NOTEBOOK.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def _saved_output(i: int) -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "".join("".join(o["text"]) for o in nb["cells"][i].get("outputs", []) if o["output_type"] == "stream")


def notebook_run(hook) -> dict:
    """Run the notebook's own cells 2, 4, 6, 9 and 28 with `hook` applied to the clean movie table right after
    the clean rule, and return the namespace and what the cells printed."""
    cells = _notebook_cells()
    assert CLEAN_LINE in cells[4], "the notebook's sample cell changed; update tmdb_robustness.py"
    ns, out = {"HOOK": hook}, io.StringIO()
    here = os.getcwd()
    os.chdir(NOTEBOOK.parent)
    try:
        with contextlib.redirect_stdout(out):
            for i in (2, 4, 6, 9, 28):
                src = cells[i].replace(CLEAN_LINE, CLEAN_LINE + "m = HOOK(m)\n") if i == 4 else cells[i]
                exec(compile(src, f"scripts_and_reviews cell {i}", "exec"), ns)
    finally:
        os.chdir(here)
        matplotlib.pyplot.close("all")
    ns["printed"] = out.getvalue()
    return ns


def _features(ns) -> dict:
    t = ns["t"]
    rho = "rho with multiple, net of budget and year"
    return {"films": int(len(ns["m"])), "hits_with_script": int((ns["s"].group == "hit").sum()),
            "flops_with_script": int((ns["s"].group == "flop").sum()),
            "hit_cut_multiple": round(float(np.exp(ns["hi"])), 3),
            "features_q_partial_below_005": int((t["q partial"] < 0.05).sum()),
            "features": {r.feature: {"rho": round(float(r[rho]), 4), "q_partial": round(float(r["q partial"]), 4)}
                         for _, r in t.iterrows()},
            "review_sentiment": {"rho": round(float(ns["rho"]), 4), "p": float(f"{ns['p']:.3g}"), "n": int(len(ns["r8"]))}}


def scripts_and_reviews(t: pd.DataFrame) -> dict:
    rep = notebook_run(lambda m: m)
    for i in (9, 28):   # the printed lines of the committed run must come back exactly
        for line in _saved_output(i).splitlines():
            assert line in rep["printed"], f"scripts_and_reviews cell {i} doesn't reproduce: {line!r}"

    def subset(m):
        return m[m.tconst.isin(t.index)].copy()

    def swap(m):
        m = subset(m)
        m["b"], m["g"] = m.tconst.map(t.budget).astype(float), m.tconst.map(t.revenue).astype(float)
        return m

    wk, tk = notebook_run(subset), notebook_run(swap)
    g = wk["m"].set_index("tconst").group.rename("wikipedia").to_frame().join(tk["m"].set_index("tconst").group.rename("tmdb"))
    moved = pd.crosstab(g.wikipedia, g.tmdb)
    return {"reproduce": _features(rep), "wikipedia": _features(wk), "tmdb": _features(tk),
            "group_changes": {f"{a} -> {b}": int(moved.loc[a, b]) for a in moved.index for b in moved.columns if a != b},
            "films_same_group": round(float((g.wikipedia == g.tmdb).mean()), 4)}


def main() -> None:
    t = tmdb_money()
    res = {"tmdb_films_with_both": int(len(t)), "arcs_step7": arcs(t), "scripts_and_reviews": scripts_and_reviews(t),
           "not_covered": "network results: their predictors use earlier films' Wikipedia returns"}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    a, s = res["arcs_step7"], res["scripts_and_reviews"]
    for k in ("D1", "D2"):
        print(f"arcs {k}: " + " | ".join(f"{v}: rho {a[k][v]['rho']:+.3f} p {a[k][v]['p']:.3f} n {a[k][v]['n']}"
                                         for v in ("reproduce", "wikipedia", "tmdb")))
    print(f"arcs flops on the TMDB films: Wikipedia {a['flops_wikipedia']}, TMDB {a['flops_tmdb']}")
    for v in ("reproduce", "wikipedia", "tmdb"):
        x = s[v]
        print(f"scripts_and_reviews {v}: {x['films']} films, {x['hits_with_script']} hits / {x['flops_with_script']} "
              f"flops with a script, {x['features_q_partial_below_005']} of 15 features q < 0.05, "
              f"review rho {x['review_sentiment']['rho']:+.3f} (n {x['review_sentiment']['n']})")
    print("strongest script features:", {v: {f: s[v]["features"][f]["rho"] for f in
                                              ("degree centralisation", "network density", "clustering")}
                                         for v in ("wikipedia", "tmdb")})
    print("group changes:", s["group_changes"], "| same group:", s["films_same_group"])
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
