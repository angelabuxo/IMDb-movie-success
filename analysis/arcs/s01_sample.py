"""Step 1: the sample and the outcome. Reproduces the funnel, builds the film table, describes log(gross/budget)."""
from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.scripts_text.report import HIGH, HIGH_SYMBOL, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

from .common import decade_bin, design, pooled_cells, primary_genre, residualise
from .config import cfg, log_run
from .paths import EARLIER, FEATURES, MOVIESUM, OUTPUTS, PRIVATE

FILMS = OUTPUTS / "s01_films.csv"            # committed: MovieSum name, derived bins, money, flags
PRIVATE_FILMS = PRIVATE / "films.parquet"    # IMDb ID, title, year, raw genres, ratings, votes
CLEAN = PRIVATE / "clean_all.parquet"        # all 2,446 clean films (for the "compared to what" of the outcome)
SUMMARY = OUTPUTS / "s01_summary.json"
DIALOGUE = re.compile(r"<dialogue>(.*?)</dialogue>", re.S)
WORD = re.compile(cfg()["text"]["word_regex"])


def _ids(name: str) -> set[str]:
    t = pd.read_csv(FEATURES / name, sep="\t", header=0, quoting=3, dtype=str, on_bad_lines="skip")
    return set(t.iloc[:, 0].str.strip('"'))


def moviesum_index() -> pd.DataFrame:
    """One row per MovieSum IMDb ID (longest script kept), with structure counts. No text kept."""
    rows, dup = {}, {}
    for path in MOVIESUM:
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                t, tid = r["script"] or "", r["imdb_id"]
                dup[tid] = dup.get(tid, 0) + 1
                if tid not in rows or len(t) > rows[tid]["script_chars"]:
                    words = sum(len(WORD.findall(html.unescape(b))) for b in DIALOGUE.findall(t))
                    rows[tid] = {"tconst": tid, "movie_name": r["movie_name"], "script_chars": len(t),
                                 "n_scene_tags": t.count("<scene>"), "n_dialogue_tags": t.count("<dialogue>"),
                                 "dialogue_words_raw": words}
    df = pd.DataFrame(rows.values())
    df["moviesum_rows"] = df.tconst.map(dup)
    df["usable"] = (df.n_scene_tags > 0) & (df.n_dialogue_tags > 0)
    return df


def run() -> dict:
    c, s = cfg(), cfg()["sample"]
    m = pd.read_csv(FEATURES / "movies_features.csv", dtype={"tconst": str})
    clean = ((m.budget_cur == "USD") & (m.gross_cur == "USD") & (m.b >= s["min_budget_usd"])
             & ~m.gross_flag.fillna("").str.contains("rentals|regional")
             & ~m.distributors.fillna("").str.contains(s["streaming_distributors"], case=False)
             & (m.y < s["max_year_exclusive"]))
    d = m[clean].rename(columns={"primaryTitle": "title_imdb", "y": "year", "b": "budget_usd", "g": "gross_usd",
                                 "votes": "imdb_votes", "rating": "imdb_rating", "rt": "rotten_tomatoes",
                                 "mc": "metacritic"}).copy()
    d["year"] = d.year.astype(int)
    d["log_mult"] = np.log(d.gross_usd / d.budget_usd)
    d["log_gross"] = np.log(d.gross_usd)
    d["log_budget"] = np.log(d.budget_usd)
    d["unprofitable"] = d.gross_usd < d.budget_usd
    d["primary_genre"] = d.genres.map(primary_genre)
    d["primary_genre_first_listed"] = d.genres.fillna("").str.split(",").str[0].replace("", "Other")
    d["decade_bin"] = d.year.map(decade_bin)

    idx = moviesum_index()
    d = d.merge(idx[idx.usable], on="tconst", how="left")
    d["has_script"] = d.movie_name.notna()
    d[["tconst", "log_mult", "has_script", "log_budget", "primary_genre", "decade_bin"]].to_parquet(CLEAN, index=False)

    f = d[d.has_script].copy()
    f["excluded_dialogue_markup"] = f.dialogue_words_raw < s["min_dialogue_words"]
    follows, named = _ids("wd_follows.tsv"), _ids("wd_franchise.tsv")
    f["sequel_follows"] = f.tconst.isin(follows)
    f["named_franchise"] = f.tconst.isin(named)
    f["likely_franchise"] = f.sequel_follows | f.named_franchise
    g = pd.read_csv(FEATURES / "genome_review_counts.csv", dtype={"tconst": str}).set_index("tconst")
    f["genome_reviews"] = f.tconst.map(g.reviews).fillna(0).astype(int)
    f["has_genome_reviews"] = f.genome_reviews > 0
    # The earlier branch's groups (cut on all clean films), to describe the hit skew of the script sample.
    old = pd.read_csv(EARLIER / "script_table.csv", dtype={"tconst": str}).set_index("tconst")
    f["earlier_group_B"] = f.tconst.map(old.group_B)
    f["earlier_group_C"] = f.tconst.map(old.group_C)
    assert f.movie_name.is_unique, "MovieSum names must identify films uniquely in committed tables"

    f = f.sort_values("movie_name").reset_index(drop=True)
    f.to_parquet(PRIVATE_FILMS, index=False)
    # Committed table: no IMDb fields (ID, title, raw genres, exact year, ratings, votes).
    keep = ["movie_name", "decade_bin", "primary_genre", "primary_genre_first_listed", "budget_usd", "gross_usd",
            "log_mult", "log_gross", "log_budget", "unprofitable", "script_chars", "n_scene_tags", "n_dialogue_tags",
            "dialogue_words_raw", "moviesum_rows", "excluded_dialogue_markup", "sequel_follows", "named_franchise",
            "likely_franchise", "has_genome_reviews", "earlier_group_B", "earlier_group_C"]
    f[keep].to_csv(FILMS, index=False)

    kept = f[~f.excluded_dialogue_markup]
    nf = kept[~kept.likely_franchise]
    e = s["expected"]
    checks = {"clean_movies": (len(d), e["clean_movies"]),
              "clean_with_usable_script": (len(f), e["clean_with_usable_script"]),
              "after_dialogue_exclusion": (len(kept), e["after_dialogue_exclusion"]),
              "not_likely_franchise": (len(nf), e["not_likely_franchise"]),
              "with_genome_reviews": (int(kept.has_genome_reviews.sum()), e["with_genome_reviews"]),
              "with_genome_reviews_not_franchise": (int(nf.has_genome_reviews.sum()),
                                                    e["with_genome_reviews_not_franchise"])}
    cells = pooled_cells(nf.primary_genre, nf.decade_bin)
    X = design(cells, nf[["log_budget"]])
    resid = residualise(nf.log_mult.to_numpy(), X)
    summary = {
        "checks": {k: {"got": int(a), "expected": int(b), "match": a == b} for k, (a, b) in checks.items()},
        "moviesum_ids": int(len(idx)), "moviesum_duplicate_ids": int((idx.moviesum_rows > 1).sum()),
        "excluded_dialogue_markup": sorted(f.movie_name[f.excluded_dialogue_markup]),
        "franchise": {"follows": int(kept.sequel_follows.sum()), "named": int(kept.named_franchise.sum()),
                      "both": int((kept.sequel_follows & kept.named_franchise).sum()),
                      "either": int(kept.likely_franchise.sum())},
        "earlier_groups_after_exclusion": {k: {"low": int((kept[f"earlier_group_{k}"] == "low").sum()),
                                               "high": int((kept[f"earlier_group_{k}"] == "high").sum())}
                                           for k in "BC"},
        "earlier_groups_primary": {k: {"low": int((nf[f"earlier_group_{k}"] == "low").sum()),
                                       "high": int((nf[f"earlier_group_{k}"] == "high").sum())}
                                   for k in "BC"},
        "outcome": {name: {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()),
                           "max": float(v.max()), "q25": float(v.quantile(.25)), "q75": float(v.quantile(.75)),
                           "median": float(v.median()),
                           "share_unprofitable": float((v < 0).mean())}
                    for name, v in (("all_clean", d.log_mult), ("scripted", kept.log_mult),
                                    ("primary_not_franchise", nf.log_mult))},
        "sd_after_cell_and_budget_primary": float(resid.std(ddof=X.shape[1])),
        "cells_primary": int(cells.nunique()), "films_in_pooled_cells": int(cells.str.contains("pooled").sum()),
        "latest_year_with_reviews": int(kept.year[kept.has_genome_reviews].max()),
        "primary_vs_first_listed_agree": float((kept.primary_genre == kept.primary_genre_first_listed).mean()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s01_sample", [FILMS.name, SUMMARY.name], {"sample": s, "controls": c["controls"]})
    if not all(v["match"] for v in summary["checks"].values()):
        raise RuntimeError(f"Feasibility numbers don't reproduce: {summary['checks']}")
    return load()


def load() -> dict:
    films = pd.read_csv(FILMS)
    clean = pd.read_parquet(CLEAN) if CLEAN.exists() else None
    return {"films": films, "clean": clean, "summary": json.loads(SUMMARY.read_text())}


def private_films() -> pd.DataFrame:
    return pd.read_parquet(PRIVATE_FILMS)


# ---------------------------------------------------------------- tables

def _funnel(res: dict) -> pd.DataFrame:
    s = res["summary"]
    ch = {k: v["got"] for k, v in s["checks"].items()}
    rows = [("Clean theatrical USD films, 100k+ IMDb votes", ch["clean_movies"], "scripts/imdb/coverage_clean_sample.sql"),
            ("… with a usable MovieSum script (≥ 1 scene and ≥ 1 dialogue tag; longest of 12 duplicates)",
             ch["clean_with_usable_script"], ""),
            ("… minus scripts whose dialogue markup failed (< 500 dialogue words)", ch["after_dialogue_exclusion"],
             f"excluded: {len(s['excluded_dialogue_markup'])}"),
            ("… not likely franchise (PRIMARY, before the arc-ready filter of Step 2)", ch["not_likely_franchise"],
             f"likely franchise: {s['franchise']['either']} (follows {s['franchise']['follows']}, "
             f"named {s['franchise']['named']}, both {s['franchise']['both']})"),
            ("Scripts with Tag Genome reviews (Step 11 frame)", ch["with_genome_reviews"],
             f"no film after {s['latest_year_with_reviews']}"),
            ("… not likely franchise", ch["with_genome_reviews_not_franchise"], "")]
    t = pd.DataFrame(rows, columns=["stage", "films", "note"])
    t["expected"] = [s["checks"][k]["expected"] for k in ("clean_movies", "clean_with_usable_script",
                                                           "after_dialogue_exclusion", "not_likely_franchise",
                                                           "with_genome_reviews", "with_genome_reviews_not_franchise")]
    return t[["stage", "films", "expected", "note"]]


def _outcome(res: dict) -> pd.DataFrame:
    o = res["summary"]["outcome"]
    names = {"all_clean": "all clean films", "scripted": "scripted (1,087)", "primary_not_franchise": "primary: not franchise"}
    return pd.DataFrame([{"sample": names[k], "films": v["n"], "mean": round(v["mean"], 3), "SD": round(v["sd"], 3),
                          "min": round(v["min"], 2), "Q1": round(v["q25"], 2), "Q3": round(v["q75"], 2),
                          "max": round(v["max"], 2), "share gross < budget": round(v["share_unprofitable"], 3),
                          "median multiple (×)": round(float(np.exp(v["median"])), 2)}
                         for k, v in o.items()])


def _primary(f: pd.DataFrame) -> pd.DataFrame:
    return f[~f.excluded_dialogue_markup & ~f.likely_franchise]


def _composition(f: pd.DataFrame) -> pd.DataFrame:
    p = _primary(f)
    t = pd.crosstab(p.primary_genre, p.decade_bin, margins=True, margins_name="all")
    order = [x[2] for x in cfg()["controls"]["decade_bins"]] + ["all"]
    return t[[c for c in order if c in t.columns]].reset_index().rename(columns={"primary_genre": "primary genre"})


def _budget(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in (("scripted (1,087)", f[~f.excluded_dialogue_markup]), ("primary: not franchise", _primary(f)),
                    ("likely franchise", f[~f.excluded_dialogue_markup & f.likely_franchise])):
        q = g.budget_usd.quantile([.1, .25, .5, .75, .9]) / 1e6
        rows.append({"sample": name, "films": len(g), **{f"P{int(k * 100)} ($M)": round(v, 1) for k, v in q.items()},
                     "median log(gross/budget)": round(g.log_mult.median(), 3)})
    return pd.DataFrame(rows)


def _skew(res: dict) -> pd.DataFrame:
    rows = []
    for sample, key in (("scripted (1,087)", "earlier_groups_after_exclusion"), ("primary: not franchise", "earlier_groups_primary")):
        g = res["summary"][key]
        for k, name in (("B", "B: quarters of all clean films"), ("C", "C: quarters within genre × decade cells")):
            rows.append({"sample": sample, "earlier definition": name, "bottom quarter (scripts)": g[k]["low"],
                         "top quarter (scripts)": g[k]["high"],
                         "top / bottom": round(g[k]["high"] / g[k]["low"], 2)})
    return pd.DataFrame(rows)


def tables(res: dict) -> dict:
    f = res["films"]
    return {"funnel": _funnel(res), "outcome": _outcome(res), "composition": _composition(f),
            "budget": _budget(f), "skew": _skew(res)}


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    f, allc = res["films"], res["clean"]
    hist = go.Figure()
    series = [("all clean films", allc.log_mult, NEUTRAL, None)] if allc is not None else []
    series += [("primary sample (not franchise)", _primary(f).log_mult, HIGH, "/"),
               ("likely franchise (scripted)", f[~f.excluded_dialogue_markup & f.likely_franchise].log_mult, LOW, "x")]
    for name, v, colour, pat in series:
        hist.add_trace(go.Histogram(x=v, name=f"{name} (n={len(v):,})", marker_color=colour, opacity=0.6,
                                    histnorm="probability", xbins=dict(start=-6, end=8, size=0.25),
                                    marker_pattern_shape=pat))
    hist.add_vline(line_width=1.5, x=0, line_dash="dash", line_color="#444", annotation_text="break-even (gross = budget)")
    hist.update_layout(barmode="overlay", xaxis_title="log(gross / budget)   (0 = break-even; 1 ≈ 2.7×; 2 ≈ 7.4×)",
                       yaxis_title="share of films in the sample")
    style(hist, "The outcome: log(gross / budget)", height=400)

    p = _primary(f)
    bud = go.Figure(go.Histogram(x=np.log10(p.budget_usd), marker_color=HIGH, nbinsx=40,
                                 name=f"primary sample (n={len(p)})"))
    bud.update_layout(xaxis_title="budget, log10 USD   (6 = 1M USD, 7 = 10M, 8 = 100M)", yaxis_title="films",
                      showlegend=False)
    style(bud, "Budget in the primary sample", height=340)

    comp = _composition(f).set_index("primary genre").drop(index="all", columns="all")
    heat = go.Figure(go.Heatmap(z=comp.values, x=comp.columns, y=comp.index, colorscale="Blues",
                                text=comp.values, texttemplate="%{text}", colorbar=dict(title="films")))
    heat.update_layout(xaxis_title="decade bin", yaxis_title="primary genre (priority rule)")
    style(heat, "Films per genre × decade cell, primary sample", height=380)

    sc = go.Figure(go.Scattergl(x=p.log_budget / np.log(10), y=p.log_mult, mode="markers",
                                marker=dict(color=HIGH, size=5, opacity=0.5, symbol=HIGH_SYMBOL), name="film"))
    sc.update_layout(xaxis_title="budget, log10 USD", yaxis_title="log(gross / budget)", showlegend=False)
    style(sc, "Cheaper films reach higher multiples (and lower ones)", height=380)
    return {"outcome": hist, "budget": bud, "cells": heat, "budget_vs_outcome": sc}


# ---------------------------------------------------------------- reading

def reading(res: dict) -> str:
    s = res["summary"]
    o = s["outcome"]
    a, sc, p = o["all_clean"], o["scripted"], o["primary_not_franchise"]
    sk, skp = s["earlier_groups_after_exclusion"], s["earlier_groups_primary"]
    ok = all(v["match"] for v in s["checks"].values())
    lean_scripted = np.exp(sc["median"]) > np.exp(a["median"])
    lean_primary = skp["B"]["high"] / skp["B"]["low"] > 1.1
    return (
        f"{'All' if ok else 'NOT all'} six funnel numbers match the feasibility note "
        f"({', '.join(f'{v['got']:,}' for v in s['checks'].values())}). "
        f"The scripted films {'lean' if lean_scripted else 'do not lean'} towards hits. Their median film grossed "
        f"{np.exp(sc['median']):.2f}× its budget, against {np.exp(a['median']):.2f}× for all clean films, and "
        f"{sk['B']['high']} scripts fall in the top quarter against {sk['B']['low']} in the bottom ({sk['C']['high']} vs "
        f"{sk['C']['low']} when the quarters are cut within genre × decade cells). "
        + ("Most of that lean comes from franchise films. " if lean_scripted and not lean_primary else "")
        + f"In the primary sample the counts are {skp['B']['high']} vs {skp['B']['low']} ({skp['C']['high']} vs "
        f"{skp['C']['low']} within cells), the median multiple is {np.exp(p['median']):.2f}×, "
        f"{p['share_unprofitable']:.1%} grossed less than their budget (all clean films: "
        f"{a['share_unprofitable']:.1%}), and the SD of log(gross/budget) is {p['sd']:.2f} (all clean films: "
        f"{a['sd']:.2f}). "
        + ("So the restricted range the project brief expects is still visible in the primary sample. "
           if lean_primary else
           "So the restricted range the project brief expects is small in the primary sample. It's mainly a "
           "property of the scripted sample as a whole. ")
        + f"The selection still isn't random, though: these are films popular enough to get 100,000+ IMDb votes "
        f"and to have a public screenplay. Once genre × decade cells and log budget are taken out, the SD left to "
        f"explain is {s['sd_after_cell_and_budget_primary']:.2f}. "
        f"{s['films_in_pooled_cells']} primary films sit in cells with fewer than "
        f"{cfg()['controls']['min_films_per_cell']} films, so they're pooled ({s['cells_primary']} cells in all). "
        f"Tag Genome reviews stop at {s['latest_year_with_reviews']}."
    )


def takeaway(res: dict) -> str:
    s = res["summary"]
    o = s["outcome"]
    a, p = o["all_clean"], o["primary_not_franchise"]
    ok = all(v["match"] for v in s["checks"].values())
    return (f"{'The rebuilt sample matches' if ok else 'The rebuilt sample does NOT match'} the feasibility numbers. "
            f"The primary sample has {p['n']:,} non-franchise films with a usable screenplay (before Step 2's "
            f"arc-ready filter). Its median film grossed {np.exp(p['median']):.2f}× its budget, against "
            f"{np.exp(a['median']):.2f}× for all clean films.")


SECTION = [
    Md("""## Step 1 · Sample and outcome

Which films do we study, and how is their box-office success spread out?

Before we measure anything, we rebuild the sample from the repo's own rules and check that it lands on the
feasibility numbers. Then we describe the outcome, log(gross / budget), in the primary sample.

### The sample, step by step
- Start from clean theatrical films with a US-dollar budget and gross and 100,000+ IMDb votes. We don't use
  the 10k tier, because its money fields aren't validated.
- Keep the films with a MovieSum screenplay that has scene and dialogue markup.
- Drop the 8 scripts whose dialogue markup failed.
- For the **primary sample**, also drop films that are likely part of a franchise (Wikidata says the film
  "follows" another one, or names its franchise). We made this choice after the earlier word-level analysis,
  where the signal vanished once franchise films were left out. We say that openly, and Steps 9 and 10 rerun
  everything with all scripts.

### Why a continuous outcome
The earlier branch compared the bottom and top quarters of films. That threw away the middle half of the data
and the ordering inside each quarter. A correlation on the continuous outcome uses every film, so it has more
power with the same number of films.

### What it measures
log(gross / budget). A value of 0 means the film grossed exactly its budget, 1 means about 2.7 times its
budget, and 2 about 7.4 times. Taking the log stops a few huge multiples from dominating.

### What it misses
Gross / budget is **not profit**:
- Marketing isn't in the budget.
- Cinemas keep a large share of the gross.
- Nothing is adjusted for inflation. The ratio still compares like with like, because budget and gross are in
  the same year's dollars.
- The ratio favours micro-budget films, horror especially. A cheap film can earn 50 times its budget, a
  blockbuster rarely more than 5."""),
    Run("s01_sample"),
    Table("s01_sample", "funnel", "How the sample is built, step by step, rebuilt from the raw data. The 'expected' "
          "column holds the feasibility numbers, and a mismatch would stop the analysis."),
    Fig("s01_sample", "outcome", "log(gross/budget) for all clean films (grey), the primary sample (blue, hatched) "
        "and the franchise films we set aside (orange, crossed). Bars show the share of each sample. If the primary "
        "sample were a narrow slice of hits, its blue bars would sit to the right of the grey ones and cover a "
        "narrower range. The table below gives the numbers."),
    Table("s01_sample", "outcome", "The outcome in three samples: spread (SD), range, quartiles, and the share of "
          "films that grossed less than their budget."),
    Table("s01_sample", "skew", "Does the script sample lean towards hits? We count films in the bottom and top "
          "quarters the earlier branch used (cut-offs from all 2,446 clean films), for all scripted films and for the "
          "primary sample. With no lean the two counts would be about equal (top / bottom ≈ 1)."),
    Fig("s01_sample", "budget", "Budgets of the primary films, on a log scale. They run from under a million to "
        "hundreds of millions of dollars, which is why we control for log budget."),
    Fig("s01_sample", "budget_vs_outcome", "Each dot is a primary film: budget (x) against log(gross/budget) (y). "
        "Cheap films spread much more widely in y, so we have to control for budget before relating anything else to "
        "the outcome."),
    Table("s01_sample", "budget", "Budget percentiles for the scripted films, the primary sample and the franchise "
          "films. Franchise films cost much more, which is one reason to treat them separately."),
    Fig("s01_sample", "cells", "How many primary films fall in each primary-genre × decade cell. These cells are "
        "the fixed effects of the analysis, so a film is compared with others from the same genre and decade. Genre follows the "
        "frozen priority rule (Horror, Sci-Fi, Thriller, Adventure, Action, Comedy, Drama). Cells with fewer than 5 "
        "films are pooled, so no film is compared only with itself."),
    Table("s01_sample", "composition", "The same counts as a table."),
    Reading("s01_sample"),
]
