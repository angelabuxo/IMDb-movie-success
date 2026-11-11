"""Step 1: who counts as a success. Builds the script table and the three group definitions."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .config import cfg, log_run
from .paths import FEATURES, MOVIESUM, OUTPUTS, PRIVATE
from .report import HIGH, HIGH_SYMBOL, LOW, LOW_SYMBOL, NEUTRAL, Fig, Md, Reading, Run, Table, style

TABLE = OUTPUTS / "script_table.csv"            # committed: no IMDb votes or ratings
PRIVATE_TABLE = PRIVATE / "clean_table.parquet"  # all 2,446 clean movies, with IMDb votes
INDEX = PRIVATE / "moviesum_index.parquet"
SUMMARY = OUTPUTS / "s01_summary.json"
DEFS = {"A": "unprofitable (gross < budget) vs all others",
        "B": "bottom vs top quartile of log(gross/budget), all films",
        "C": "bottom vs top quartile within genre × decade cells"}


# ---------------------------------------------------------------- data

def moviesum_index() -> pd.DataFrame:
    """One row per MovieSum IMDb ID (longest script kept), with structure counts. No text kept."""
    rows = {}
    dup = {}
    for path in MOVIESUM:
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                t = r["script"] or ""
                tid = r["imdb_id"]
                dup[tid] = dup.get(tid, 0) + 1
                if tid not in rows or len(t) > rows[tid]["script_chars"]:
                    rows[tid] = {"tconst": tid, "movie_name": r["movie_name"], "split": path.stem,
                                 "script_chars": len(t), "n_scene_tags": t.count("<scene>"),
                                 "n_dialogue_tags": t.count("<dialogue>")}
    df = pd.DataFrame(rows.values()).sort_values("tconst").reset_index(drop=True)
    df["moviesum_rows"] = df.tconst.map(dup)
    df["usable"] = (df.n_scene_tags > 0) & (df.n_dialogue_tags > 0)
    df.to_parquet(INDEX, index=False)
    return df


def primary_genre(genres: str) -> str:
    g = set((genres or "").split(","))
    for name in cfg()["groups"]["primary_genre_priority"]:
        if name in g:
            return name
    return cfg()["groups"]["other_genre"]


def decade_bin(year: int) -> str:
    for lo, hi, label in cfg()["groups"]["decade_bins"]:
        if lo <= year <= hi:
            return label
    return "other"


def quartile_labels(values: pd.Series, basis: pd.Series) -> pd.Series:
    """'low' at or below the basis' first quartile, 'high' at or above its third, else None."""
    q1, q3 = basis.quantile([0.25, 0.75])
    return pd.Series(np.where(values <= q1, "low", np.where(values >= q3, "high", None)), index=values.index)


def run() -> dict:
    c = cfg()
    s = c["sample"]
    m = pd.read_csv(FEATURES / "movies_features.csv", dtype={"tconst": str})
    clean = ((m.budget_cur == "USD") & (m.gross_cur == "USD") & (m.b >= s["min_budget_usd"])
             & ~m.gross_flag.fillna("").str.contains("rentals|regional")
             & ~m.distributors.fillna("").str.contains(s["streaming_distributors"], case=False)
             & (m.y < s["max_year_exclusive"]))
    d = m[clean].copy().rename(columns={"primaryTitle": "title_imdb", "y": "year", "b": "budget_usd",
                                        "g": "gross_usd", "votes": "imdb_votes"})
    d["year"] = d.year.astype(int)
    d["log_mult"] = np.log(d.gross_usd / d.budget_usd)
    d["log_budget"] = np.log(d.budget_usd)
    d["primary_genre"] = d.genres.map(primary_genre)
    d["primary_genre_first_listed"] = d.genres.fillna("").str.split(",").str[0]
    d["decade_bin"] = d.year.map(decade_bin)
    d["cell"] = d.primary_genre + " | " + d.decade_bin

    idx = moviesum_index()
    usable = set(idx.tconst[idx.usable])
    d["has_script"] = d.tconst.isin(usable)
    d = d.merge(idx[["tconst", "script_chars", "n_scene_tags", "n_dialogue_tags", "moviesum_rows"]],
                on="tconst", how="left")

    # (A) literal: gross below budget
    d["group_A"] = np.where(d.gross_usd < d.budget_usd, "low", "high")
    # (B) global quartiles, cut on all clean movies
    d["group_B"] = quartile_labels(d.log_mult, d.log_mult)
    # (C) quartiles within genre x decade cells, cut on all clean movies in the cell
    size = d.groupby("cell").tconst.transform("size")
    d["cell_ok"] = size >= c["groups"]["min_clean_movies_per_cell"]
    d["group_C"] = None
    for cell, g in d[d.cell_ok].groupby("cell"):
        d.loc[g.index, "group_C"] = quartile_labels(g.log_mult, g.log_mult)
    # (C) reported within one genre, or within one decade, alone
    d["group_C_genre"] = d.groupby("primary_genre", group_keys=False).log_mult.apply(
        lambda v: quartile_labels(v, v))
    d["group_C_decade"] = d.groupby("decade_bin", group_keys=False).log_mult.apply(
        lambda v: quartile_labels(v, v))

    d = d.sort_values("tconst").reset_index(drop=True)
    d.to_parquet(PRIVATE_TABLE, index=False)
    sc = d[d.has_script].copy()
    # The committed table carries no IMDb fields (title, raw genres, exact year, votes): IMDb's terms don't
    # allow republishing them. MovieSum's own name ("Pulp Fiction_1994") identifies the film; genre and
    # decade appear only as derived bins. load() adds the IMDb columns back from the private table.
    keep = ["tconst", "movie_name", "decade_bin", "primary_genre", "primary_genre_first_listed", "cell", "cell_ok",
            "budget_usd", "gross_usd", "log_mult", "log_budget", "script_chars", "n_scene_tags", "n_dialogue_tags",
            "moviesum_rows", "group_A", "group_B", "group_C", "group_C_genre", "group_C_decade"]
    sc["movie_name"] = sc.tconst.map(dict(zip(idx.tconst, idx.movie_name)))
    sc[keep].to_csv(TABLE, index=False)

    e = s["expected"]
    checks = {
        "clean_movies": (len(d), e["clean_movies"]),
        "clean_with_usable_script": (len(sc), e["clean_with_usable_script"]),
        "gross_below_budget": (int((sc.group_A == "low").sum()), e["gross_below_budget"]),
        "global_quartile_bottom": (int((sc.group_B == "low").sum()), e["global_quartile_bottom"]),
        "global_quartile_top": (int((sc.group_B == "high").sum()), e["global_quartile_top"]),
    }
    summary = {
        "checks": {k: {"got": int(a), "expected": int(b), "match": a == b} for k, (a, b) in checks.items()},
        "moviesum_ids": int(len(idx)), "moviesum_usable": int(idx.usable.sum()),
        "moviesum_duplicate_ids": int((idx.moviesum_rows > 1).sum()),
        "scripts_under_min_scenes": int((sc.n_scene_tags < c["text"]["min_scenes_flag"]).sum()),
        "cells_total": int(d.cell.nunique()), "cells_ok": int(d[d.cell_ok].cell.nunique()),
        "scripts_in_dropped_cells": int((~sc.cell_ok).sum()),
        "primary_vs_first_listed_agree": float((sc.primary_genre == sc.primary_genre_first_listed).mean()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s01_groups", [TABLE.name, SUMMARY.name], {"sample": s, "groups": c["groups"]})
    if not all(v["match"] for v in summary["checks"].values()):
        raise RuntimeError(f"Coverage numbers don't reproduce: {summary['checks']}")
    return load()


def load() -> dict:
    sc = pd.read_csv(TABLE, dtype={"tconst": str})
    allc = pd.read_parquet(PRIVATE_TABLE) if PRIVATE_TABLE.exists() else None
    if allc is not None:   # IMDb fields, kept out of the committed table
        sc = sc.merge(allc[["tconst", "title_imdb", "year", "genres"]], on="tconst", how="left")
    return {"scripts": sc, "clean": allc, "summary": json.loads(SUMMARY.read_text())}


# ---------------------------------------------------------------- tables

def _sizes(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in "ABC":
        col = sc[f"group_{k}"]
        rows.append({"definition": k, "meaning": DEFS[k], "tier": "100k+",
                     "low (unsuccessful)": int((col == "low").sum()),
                     "high (successful)": int((col == "high").sum()),
                     "not in either group": int((~col.isin(["low", "high"])).sum())})
    return pd.DataFrame(rows)


def _strata(sc: pd.DataFrame) -> pd.DataFrame:
    n = cfg()["groups"]["min_scripts_per_group"]
    rows = []
    for kind, col, key in (("genre", "group_C_genre", "primary_genre"), ("decade", "group_C_decade", "decade_bin")):
        for val, g in sc.groupby(key):
            lo, hi = int((g[col] == "low").sum()), int((g[col] == "high").sum())
            rows.append({"stratum": kind, "value": val, "scripts": len(g), "low": lo, "high": hi,
                         "usable": "yes" if min(lo, hi) >= n else f"no: a group has fewer than {n}"})
    return pd.DataFrame(rows)


def _cells(sc: pd.DataFrame) -> pd.DataFrame:
    t = sc.groupby("cell").agg(scripts=("tconst", "size"), cell_ok=("cell_ok", "first"),
                               low=("group_C", lambda v: int((v == "low").sum())),
                               high=("group_C", lambda v: int((v == "high").sum()))).reset_index()
    return t.sort_values(["cell_ok", "scripts"], ascending=[False, False])


def _bias(sc: pd.DataFrame) -> pd.DataFrame:
    """Is 'success' really 'cheap'? Budget, year and horror share by group, with a Mann-Whitney test
    of log budget between the two groups."""
    rows = []
    for k in "ABC":
        col = sc[f"group_{k}"]
        lo, hi = sc[col == "low"], sc[col == "high"]
        p = stats.mannwhitneyu(lo.log_budget, hi.log_budget).pvalue
        rows.append({"definition": k,
                     "median budget, low ($M)": round(lo.budget_usd.median() / 1e6, 1),
                     "median budget, high ($M)": round(hi.budget_usd.median() / 1e6, 1),
                     "median year, low": int(lo.year.median()), "median year, high": int(hi.year.median()),
                     "horror share, low": round((lo.primary_genre == "Horror").mean(), 3),
                     "horror share, high": round((hi.primary_genre == "Horror").mean(), 3),
                     "Mann-Whitney p, log budget": float(f"{p:.2g}")})
    return pd.DataFrame(rows)


def _genre_mix(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in "ABC":
        col = sc[f"group_{k}"]
        for grp in ("low", "high"):
            share = sc[col == grp].primary_genre.value_counts(normalize=True)
            rows.append({"definition": k, "group": grp, **{g: round(share.get(g, 0.0), 3)
                         for g in cfg()["groups"]["primary_genre_priority"] + ["Other"]}})
    return pd.DataFrame(rows)


def tables(res: dict) -> dict:
    sc = res["scripts"]
    return {"sizes": _sizes(sc), "strata": _strata(sc), "cells": _cells(sc), "bias": _bias(sc),
            "genre_mix": _genre_mix(sc),
            "checks": pd.DataFrame([{"check": k, **v} for k, v in res["summary"]["checks"].items()])}


# ---------------------------------------------------------------- figures

def _by_group(sc, value, title, xaxis, log=False):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"({k}) {DEFS[k]}" for k in "ABC"], shared_yaxes=True)
    for i, k in enumerate("ABC", 1):
        col = sc[f"group_{k}"]
        for grp, colour, name in (("low", LOW, "unsuccessful"), ("high", HIGH, "successful")):
            v = sc.loc[col == grp, value]
            fig.add_trace(go.Box(x=v, name=f"{name} (n={len(v)})", marker_color=colour, boxpoints="outliers",
                                 orientation="h", showlegend=False), row=1, col=i)
        fig.update_xaxes(type="log" if log else "linear", title_text=xaxis, row=1, col=i)
    fig.update_annotations(font_size=11)
    return style(fig, title, height=380)


def figures(res: dict) -> dict:
    sc = res["scripts"]
    g = _genre_mix(sc)
    order = cfg()["groups"]["primary_genre_priority"] + ["Other"]
    mix = go.Figure()
    palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    patterns = ["", "/", "\\", "x", "-", "|", "+", "."]
    labels = [f"({r.definition}) {'unsuccessful' if r.group == 'low' else 'successful'}" for _, r in g.iterrows()]
    for gname, colour, pat in zip(order, palette, patterns):
        mix.add_trace(go.Bar(y=labels, x=g[gname], name=gname, orientation="h", marker_color=colour,
                             marker_pattern_shape=pat))
    mix.update_layout(barmode="stack", xaxis_title="share of the group's scripts", yaxis_autorange="reversed")
    style(mix, "Primary genre mix by group", height=420)

    allc = res["clean"]
    cov = go.Figure()
    if allc is not None:
        for flag, colour, sym, name in ((True, HIGH, HIGH_SYMBOL, "has a script"),
                                        (False, NEUTRAL, LOW_SYMBOL, "no script")):
            v = allc[allc.has_script == flag]
            cov.add_trace(go.Histogram(x=v.log_mult, name=f"{name} (n={len(v)})", marker_color=colour,
                                       opacity=0.65, histnorm="probability", nbinsx=50))
        q1, q3 = allc.log_mult.quantile([0.25, 0.75])
        for q, lab in ((q1, "Q1 cut"), (q3, "Q3 cut"), (0.0, "break-even")):
            cov.add_vline(x=q, line_dash="dash", line_color="#444", annotation_text=lab)
        cov.update_layout(barmode="overlay", xaxis_title="log(gross / budget)  (0 = gross equals budget)",
                          yaxis_title="share of films")
        style(cov, "Films with a script lean towards hits", height=380)
    return {"budget": _by_group(sc, "budget_usd", "Budget by group", "budget (USD, log scale)", log=True),
            "year": _by_group(sc, "year", "Release year by group", "release year"),
            "genre_mix": mix, "coverage": cov}


# ---------------------------------------------------------------- reading

def reading(res: dict) -> str:
    sc, s = res["scripts"], res["summary"]
    b = _bias(sc).set_index("definition")
    st = _strata(sc)
    skipped = st[st.usable != "yes"]
    return (
        f"All five coverage checks reproduce ({', '.join(f'{k} {v['got']:,}' for k, v in s['checks'].items())}). "
        f"Under (A) only {int((sc.group_A == 'low').sum())} scripts are unsuccessful, so any word-level test on (A) "
        f"has little power. Under (B) the median budget is ${b.loc['B', 'median budget, low ($M)']}M for the bottom "
        f"quartile against "
        f"${b.loc['B', 'median budget, high ($M)']}M for the top (Mann-Whitney p = {b.loc['B', 'Mann-Whitney p, log budget']}), "
        f"and horror makes up {b.loc['B', 'horror share, high']:.0%} of the top quartile against "
        f"{b.loc['B', 'horror share, low']:.0%} of the bottom. Within genre × decade cells (C) the budget gap is "
        f"${b.loc['C', 'median budget, low ($M)']}M against ${b.loc['C', 'median budget, high ($M)']}M "
        f"(p = {b.loc['C', 'Mann-Whitney p, log budget']}), so (C) removes the genre and era difference but not "
        f"all of the budget difference; Step 4's matching handles budget. "
        f"{s['cells_ok']} of {s['cells_total']} genre × decade cells have at least "
        f"{cfg()['groups']['min_clean_movies_per_cell']} clean films; {s['scripts_in_dropped_cells']} scripts sit in "
        f"cells too small to cut quartiles and are left out of (C). "
        f"{len(skipped)} single-genre or single-decade strata have fewer than "
        f"{cfg()['groups']['min_scripts_per_group']} scripts in a group and are skipped: "
        f"{', '.join(skipped.value.astype(str))}. "
        f"The priority genre rule agrees with Àngela's first-listed rule for "
        f"{s['primary_vs_first_listed_agree']:.0%} of scripts."
    )


SECTION = [
    Md("""## Step 1 · Who counts as a success?

**What we do.** We define success as a film's gross divided by its budget, and use log(gross/budget) when we
need a number on a continuous scale: 0 means the film grossed its budget, 1 means about 2.7 times its budget.
From it we build three pairs of groups for the 1,095 clean 100k+ films that have a usable MovieSum script:

- **(A)** unprofitable (gross below budget) against everyone else. The most literal definition, but tiny.
- **(B)** the bottom quarter against the top quarter of log(gross/budget). The cut-offs are computed on all
  2,446 clean films, scripted or not, so a film's label doesn't depend on which other films happen to have a
  script.
- **(C)** the same quarters, but cut inside each primary-genre × decade cell. This compares a horror film from
  the 2000s with other horror films from the 2000s, not with dramas from the 1970s.

**Why.** Scattertext compares two groups, so we need to say which two. The course's "compared to what"
question starts here: a word can only be distinctive of success relative to a definition of failure.

**What it measures.** How much a film earned relative to what it cost to make, as reported in its Wikipedia
infobox.

**What it misses.** Gross minus budget is **not profit**: marketing isn't in the budget, cinemas keep a large
share of the gross (often around half), and nothing is adjusted for inflation (the ratio is unaffected by
inflation only because budget and gross are in the same year's dollars). The ratio also rewards very cheap
films: a $1 million horror film that grosses $50 million beats a $200 million blockbuster that grosses $1
billion.

**Primary analysis, fixed in advance:** definition (C), dialogue layer, character names removed, minimum
document frequency 5. Everything else in these notebooks is exploratory."""),
    Run("s01_groups"),
    Table("s01_groups", "checks", "Reproduction of the coverage numbers before anything else is computed."),
    Table("s01_groups", "sizes", "Group sizes under each definition. All films are in the 100k+ vote tier."),
    Fig("s01_groups", "coverage", "log(gross/budget) for the 2,446 clean films, split by whether MovieSum has "
        "their script. Dashed lines: the quartile cuts used for (B) and break-even. The scripted films sit "
        "further right: the script sample over-represents hits, which is why (B) has 306 successful and only "
        "243 unsuccessful scripts."),
    Fig("s01_groups", "budget", "Budget of the unsuccessful (orange) and successful (blue) scripts under each "
        "definition, log scale. Look for whether the two boxes overlap: where they don't, a word difference "
        "between the groups could be a budget difference."),
    Fig("s01_groups", "year", "Release year by group. A shift between the boxes means era could explain word "
        "differences (language changes over decades)."),
    Fig("s01_groups", "genre_mix", "Share of each primary genre in each group, with a fill pattern per genre "
        "so the chart reads without colour. Horror's share among successful films under (A) and (B) is the "
        "known micro-budget bias; (C) equalises genre by construction."),
    Table("s01_groups", "bias", "The budget-bias test: median budget, median year and horror share in each "
          "group, and a Mann-Whitney test of log budget between the two groups."),
    Table("s01_groups", "strata", "Definition (C) within one genre or one decade alone. A stratum is used only "
          "if both groups have at least 30 scripts (the brief's rule)."),
    Table("s01_groups", "cells", "Genre × decade cells. A cell needs at least 20 clean films (scripted or "
          "not) to cut quartiles; scripts in smaller cells are left out of (C) and counted here."),
    Reading("s01_groups"),
    Md("""**Why we keep three definitions for now.** (A) is the most literal reading of "unprofitable" but has so
few scripts that rare words can't be tested. (B) is larger and more robust, but mixes genres and eras, so it
partly measures "horror vs drama" or "1980s vs 2010s". (C) controls genre and era by construction, which is
why it's the primary analysis. None of them controls budget fully; Step 4 does that by matching."""),
]
