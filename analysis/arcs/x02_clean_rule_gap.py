"""Exploratory, not a pre-registered step: why Step 1 no longer reproduces, and what the primary test gives on
the films the current clean rule keeps. See LOG.md, 2026-11-16.

    python -m analysis.arcs.x02_clean_rule_gap

Committed:  outputs/x02_clean_rule_gap.json (counts and test statistics only)

The 14 November sample was drawn under the clean rule before the 15 November changes. Its 2,446 clean films
are rebuilt from the infobox table in data/bundles/wikipedia-wikidata-2026-11-11.tar.zst (extracted to a
temporary folder), with the rule as it stood then: USD budget and gross, budget at least $1,000, no rentals or
regional flag, no streaming distributor, released before 2026. The scripted samples are read from the committed
Step 1 and Step 2 tables, so they are exact.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.clean_rule import STREAMING_DISTRIBUTORS, why_not_clean

from . import s07_primary as s7
from .common import pooled_cells
from .config import cfg
from .paths import FEATURES, OUTPUTS, REPO

OUT = OUTPUTS / "x02_clean_rule_gap.json"
OLD_BUNDLE = REPO / "data" / "bundles" / "wikipedia-wikidata-2026-11-11.tar.zst"


def old_clean_ids(m: pd.DataFrame) -> set[str]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["tar", "--zstd", "-xf", str(OLD_BUNDLE), "-C", tmp, "features/wiki_infobox.csv"], check=True)
        old = pd.read_csv(Path(tmp) / "features" / "wiki_infobox.csv", dtype={"tconst": str}).drop_duplicates("tconst")
    o = m[["tconst", "y", "distributors"]].merge(
        old[["tconst", "budget", "budget_cur", "gross_cur", "gross_flag"]], on="tconst", how="left")
    ok = ((o.budget_cur == "USD") & (o.gross_cur == "USD") & (o.budget >= 1000)
          & ~o.gross_flag.fillna("").str.contains("rentals|regional")
          & ~o.distributors.fillna("").str.contains("|".join(STREAMING_DISTRIBUTORS), case=False) & (o.y < 2026))
    return set(o.tconst[ok])


def run() -> dict:
    m = pd.read_csv(FEATURES / "movies_features.csv", dtype={"tconst": str})
    m["why"] = why_not_clean(m)
    old = old_clean_ids(m)
    new = set(m.tconst[m.why == "clean"])
    lost = m[m.tconst.isin(old - new)]
    counts = [{"count": "clean_movies", "expected": 2446, "rebuilt": len(old), "still_clean": len(old & new),
               "gained": len(new - old), "lost_pre_1975": int(lost.why.str.startswith("released").sum()),
               "lost_other": int((~lost.why.str.startswith("released")).sum()),
               "lost_other_reasons": lost.why[~lost.why.str.startswith("released")].value_counts().to_dict()}]

    ids = pd.read_csv(FEATURES / "moviesum_ids.csv", dtype=str).rename(columns={"imdb_id": "tconst"})
    why = ids.merge(m[["tconst", "why"]], on="tconst").set_index("movie_name").why
    f = pd.read_csv(OUTPUTS / "s01_films.csv", dtype={"movie_name": str})
    t = pd.read_csv(OUTPUTS / "analysis_table.csv", dtype={"movie_name": str})
    f["why"], t["why"] = f.movie_name.map(why), t.movie_name.map(why)
    assert f.why.notna().all() and t.why.notna().all()
    kept = f[~f.excluded_dialogue_markup]
    nf = kept[~kept.likely_franchise]
    for name, d in (("clean_with_usable_script", f), ("after_dialogue_exclusion", kept), ("not_likely_franchise", nf),
                    ("not_likely_franchise_arc_ready", t[t.in_primary]),
                    ("with_genome_reviews", kept[kept.has_genome_reviews]),
                    ("with_genome_reviews_not_franchise", nf[nf.has_genome_reviews])):
        out = d[d.why != "clean"]
        counts.append({"count": name, "expected": len(d), "still_clean": int((d.why == "clean").sum()),
                       "lost_pre_1975": int(out.why.str.startswith("released").sum()),
                       "lost_other": int((~out.why.str.startswith("released")).sum())})

    c = cfg()
    pc, ctl = c["primary"], c["controls"]
    inst = json.loads((OUTPUTS / "s03_instrument_decision.json").read_text())["primary_instrument"]
    saved = json.loads((OUTPUTS / "s07_primary.json").read_text())["results"]
    df = s7.sample(inst)

    def test(d):
        rng = np.random.default_rng(c["seed"])
        return {k: s7.association(d, k, pc["outcome"], ctl["covariates"], pc["permutations"], pc["bootstrap"], rng)
                for k in s7.DESCRIPTORS}

    rep = test(df)
    for k in s7.DESCRIPTORS:
        assert abs(rep[k]["rho"] - saved[k]["rho"]) < 1e-12 and rep[k]["p"] == saved[k]["p"], f"{k} doesn't reproduce"
    dropped = df.merge(t[["movie_name", "why"]], on="movie_name")
    years = dropped[dropped.why != "clean"]
    keep = dropped[dropped.why == "clean"].drop(columns="why").reset_index(drop=True)
    keep["cell"] = pooled_cells(keep.primary_genre, keep.decade_bin)
    sens = test(keep)
    res = {"counts": counts,
           "primary_films_dropped": {"n": int(len(years)), "unprofitable": int(years.unprofitable.sum()),
                                     "unprofitable_share_of_kept": round(float(keep.unprofitable.mean()), 4)},
           "sensitivity_current_rule": {
               "n": int(len(keep)), "cells": int(keep.cell.nunique()),
               "results": {k: {x: round(float(v[x]), 4) for x in ("rho", "z", "p", "ci_low", "ci_high")}
                           for k, v in sens.items()},
               "note": "exploratory: Step 7's association() on the 834 minus the films the current rule drops; "
                       "cells re-pooled; same seed, permutations and bootstrap as Step 7"}}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    r = run()
    print(pd.DataFrame(r["counts"]).drop(columns=["lost_other_reasons"], errors="ignore").to_string(index=False))
    print(r["primary_films_dropped"])
    print(json.dumps(r["sensitivity_current_rule"]["results"]))
