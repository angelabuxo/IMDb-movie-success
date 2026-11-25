"""Compare Wikipedia infobox budget and gross with TMDB's budget and revenue, as a second source.

Run from the data folder (data/raw):  python ../../scripts/imdb/tmdb_compare.py [--sample]
Reads features/movies_features_10k.csv and tmdb/tmdb_project_movies.csv (or tmdb/TMDB_all_movies.csv).

Writes two files:
  analysis/checks/tmdb_compare.json   committed: counts, agreement rates and ratio histograms only
  features/tmdb_compare.csv           in the data folder, never committed: one row per clean movie with both
                                      sources' figures (TMDB's per-film numbers stay out of the public repo)

What it captures: whether two sources report about the same dollar figure for a film.
What it misses: both can copy the same wrong number, and TMDB is edited by users with no citation,
so agreement is not proof and a disagreement does not say which side is wrong.

Three things the headline agreement hides, each reported separately:
  identical      how often the two figures are the same number, a sign one was copied from the other
  flips          films whose under-budget label differs between the sources, split by distance from break-even
  outside_clean  films the clean rule drops, by reason, and how many TMDB has both figures for

Assumptions: TMDB's figures are US dollars (TMDB has no currency field) and a TMDB 0 means "no figure".
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))   # repo root, for the shared clean rule
from analysis.clean_rule import REASONS, clean_mask, why_not_clean  # noqa: E402

OUT_JSON = REPO / "src" / "analysis" / "checks" / "tmdb_compare.json"
OUT_CSV = Path("features") / "tmdb_compare.csv"
TOL = (0.01, 0.10, 0.25)                      # relative difference bands
IDENTICAL = 0.005                             # within 0.5%: the same figure, allowing for rounding to 3 digits
NEAR_BREAK_EVEN = 1.25                        # Wikipedia gross / budget within this factor of 1
EDGES = np.round(np.arange(-2.0, 2.0001, 0.1), 1)   # log10(TMDB / Wikipedia); values outside go in the end bins
TIERS = ("all clean", "100k+", "10k-100k")


def agreement(wiki: pd.Series, tmdb: pd.Series) -> dict:
    """Agreement of one money field on the movies where TMDB has a figure (0 means none)."""
    ok = (tmdb > 0) & wiki.notna()
    w, t = wiki[ok], tmdb[ok]
    rel = (t - w).abs() / w
    out = {"n": int(ok.sum()), "tmdb_missing": int((~ok).sum())}
    for tol in TOL:
        out[f"within_{int(tol * 100)}pct"] = round(float((rel <= tol).mean()), 4)
    out["exactly_equal"] = round(float((t == w).mean()), 4)
    out["identical_within_half_pct"] = round(float((rel <= IDENTICAL).mean()), 4)
    out["median_ratio_tmdb_over_wiki"] = round(float((t / w).median()), 4)
    out["tmdb_higher_2x"] = int((t / w >= 2).sum())
    out["tmdb_lower_2x"] = int((t / w <= 0.5).sum())
    out["off_by_2x_or_more"] = round(float(((t / w >= 2) | (t / w <= 0.5)).mean()), 4)
    out["spearman_log"] = round(float(spearmanr(np.log(w), np.log(t)).statistic), 4)
    out["ratio_hist"] = np.histogram(np.clip(np.log10(t / w), EDGES[0], EDGES[-1] - 1e-9), bins=EDGES)[0].tolist()
    return out


def tier_rows(d: pd.DataFrame, tier: str) -> pd.DataFrame:
    return d if tier == "all clean" else d[d.tier == "100k+"] if tier == "100k+" else d[d.tier != "100k+"]


def summarise(sub: pd.DataFrame) -> dict:
    both = sub[(sub.tmdb_budget > 0) & (sub.tmdb_revenue > 0)]
    wf, tf = both.wiki_gross < both.wiki_budget, both.tmdb_revenue < both.tmdb_budget
    wm = both.wiki_gross / both.wiki_budget
    near = (wm < NEAR_BREAK_EVEN) & (wm > 1 / NEAR_BREAK_EVEN)
    flip = wf != tf
    flips = {"flips": int(flip.sum()), "near_break_even_films": int(near.sum()),
             "flips_near_break_even": int((flip & near).sum()), "flips_elsewhere": int((flip & ~near).sum()),
             "flop_on_wikipedia_only": int((wf & ~tf).sum()), "flop_on_tmdb_only": int((~wf & tf).sum())}
    return {"n": int(len(sub)), "flips": flips,
            "budget": agreement(sub.wiki_budget, sub.tmdb_budget),
            "gross": agreement(sub.wiki_gross, sub.tmdb_revenue),
            "both_tmdb_fields": int(len(both)),
            "under_budget_wikipedia": int(wf.sum()), "under_budget_tmdb": int(tf.sum()),
            "same_under_budget_call": round(float((wf == tf).mean()), 4),
            "spearman_log_multiple": round(float(spearmanr(np.log(both.wiki_gross / both.wiki_budget),
                                                           np.log(both.tmdb_revenue / both.tmdb_budget)).statistic), 4)}


def outside_clean(a: pd.DataFrame) -> list[dict]:
    """Films the clean rule drops, by the first reason that applies, and what TMDB has for them. A film with
    no parsed Wikipedia figure but both TMDB figures is where a parser miss, or a recoverable film, would be."""
    rows = []
    for reason, g in a[a.why != "clean"].groupby("why", sort=False):
        both = (g.tmdb_budget > 0) & (g.tmdb_revenue > 0)
        rows.append({"reason": reason, "films": int(len(g)), "tmdb_has_both": int(both.sum()),
                     "tmdb_has_both_100k": int((both & (g.tier == "100k+")).sum()),
                     "tmdb_has_both_10k_100k": int((both & (g.tier != "100k+")).sum())})
    return sorted(rows, key=lambda r: -r["films"])


UNPARSED = ("no Wikipedia article linked", "article has no infobox", "no infobox field for the missing figure",
            "raw value present, no number parsed")


def unparsed(a: pd.DataFrame) -> pd.DataFrame:
    """Films with no parsed Wikipedia budget or gross but both figures on TMDB, split by where the figure went
    missing. Only the last group can be parser misses. The two infobox tables are combined the way
    refresh_name_columns.py combines them for movies_features_10k.csv (100k+ table first, then the 10k table,
    first row per IMDb ID), so this reads the same rows the pipeline copied."""
    w = (pd.concat([pd.read_csv(f, dtype=str) for f in ("features/wiki_infobox.csv", "features/wiki_infobox_10k.csv")])
         .drop_duplicates("tconst"))
    x = a[(a.why == REASONS[0]) & (a.tmdb_budget > 0) & (a.tmdb_revenue > 0)]
    x = x.merge(w[["tconst", "has_infobox", "budget_raw", "gross_raw"]], on="tconst", how="left", indicator=True)
    raw_b = x.budget_raw.fillna("").str.strip() != ""
    raw_g = x.gross_raw.fillna("").str.strip() != ""
    group = pd.Series(UNPARSED[2], index=x.index)
    group[(x.b.isna() & raw_b) | (x.g.isna() & raw_g)] = UNPARSED[3]
    group[(x._merge == "both") & ~x.has_infobox.fillna("").str.lower().isin(["true", "1"])] = UNPARSED[1]
    group[x._merge != "both"] = UNPARSED[0]
    missing = np.where(x.b.isna() & x.g.isna(), "both", np.where(x.b.isna(), "budget", "gross"))
    return x.assign(group=group.to_numpy(), missing=missing)


LABELS = REPO / "src" / "analysis" / "checks" / "tmdb_unparsed_labels.csv"
FIELD = {k: re.compile(rf"^\s*\|\s*{k}\s*=(.*)$", re.M) for k in ("budget", "gross")}
COMMENT = re.compile(r"<!--.*?-->", re.S)


def wikitext_field_lines(u: pd.DataFrame) -> dict:
    """An independent check on "no infobox field": does the article's own wikitext have a non-empty
    `| budget =` or `| gross =` line for the figure that's missing? Plain regex, not the parser under test."""
    want = set(u.tconst)
    found = {}
    for path in ("wiki/articles.jsonl", "wiki/articles_10k.jsonl"):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                for t in set(r.get("tconst") or []) & want:
                    if t not in found:
                        text = COMMENT.sub("", r.get("wikitext") or "")
                        found[t] = {k: any(v.strip() for v in rx.findall(text)) for k, rx in FIELD.items()}
    has = [found.get(t, {}).get("budget" if miss == "budget" else "gross", False) if miss != "both"
           else any(found.get(t, {}).values()) for t, miss in zip(u.tconst, u.missing)]
    return {"films": int(len(u)), "article_found": int(sum(t in found for t in u.tconst)),
            "nonempty_line_for_missing_figure": int(sum(has))}


def load() -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """One row per clean movie with both sources' figures, the TMDB file used, and every movie with its
    clean-rule reason. Run from the data folder."""
    m = pd.read_csv("features/movies_features_10k.csv", dtype={"tconst": str})
    src = "tmdb/tmdb_project_movies.csv" if os.path.exists("tmdb/tmdb_project_movies.csv") else "tmdb/TMDB_all_movies.csv"
    t = pd.read_csv(src, usecols=["imdb_id", "budget", "revenue", "vote_count"], low_memory=False)
    t = t[t.imdb_id.isin(set(m.tconst))].sort_values("vote_count", ascending=False).drop_duplicates("imdb_id")
    a = m.assign(why=why_not_clean(m)).merge(t, left_on="tconst", right_on="imdb_id", how="left")
    a = a.rename(columns={"budget": "tmdb_budget", "revenue": "tmdb_revenue"})
    a[["tmdb_budget", "tmdb_revenue"]] = a[["tmdb_budget", "tmdb_revenue"]].fillna(0)
    d = m[clean_mask(m)].merge(t, left_on="tconst", right_on="imdb_id", how="left")
    d["in_tmdb"] = d.imdb_id.notna()
    d = d.rename(columns={"primaryTitle": "title", "y": "year", "b": "wiki_budget", "g": "wiki_gross",
                          "budget": "tmdb_budget", "revenue": "tmdb_revenue"})
    d[["tmdb_budget", "tmdb_revenue"]] = d[["tmdb_budget", "tmdb_revenue"]].fillna(0)
    d["budget_ratio"] = (d.tmdb_budget / d.wiki_budget).where(d.tmdb_budget > 0)
    d["gross_ratio"] = (d.tmdb_revenue / d.wiki_gross).where(d.tmdb_revenue > 0)
    cols = ["tconst", "title", "year", "tier", "in_tmdb", "wiki_budget", "tmdb_budget", "budget_ratio",
            "wiki_gross", "tmdb_revenue", "gross_ratio"]
    return d[cols], src, a


def main() -> None:
    d, src, a = load()
    state = json.loads(Path("tmdb/etl_state.json").read_text()) if os.path.exists("tmdb/etl_state.json") else {}
    res = {"tmdb_source": src, "tmdb_snapshot": state.get("last_successful_date"),
           "clean_movies": int(len(d)), "matched_to_tmdb_row": int(d.in_tmdb.sum()),
           "tmdb_revenue_under_1000": int(((d.tmdb_revenue > 0) & (d.tmdb_revenue < 1000)).sum()),
           "tmdb_budget_under_1000": int(((d.tmdb_budget > 0) & (d.tmdb_budget < 1000)).sum()),
           "identical_tolerance": IDENTICAL, "near_break_even_factor": NEAR_BREAK_EVEN,
           "all_movies": int(len(a)), "outside_clean": outside_clean(a),
           "unparsed_with_tmdb": [{"group": g, "films": int(len(v)), **{f"missing_{k}": int((v.missing == k).sum())
                                   for k in ("budget", "gross", "both")},
                                   "tier_100k": int((v.tier == "100k+").sum())}
                                  for g in UNPARSED for v in [unparsed(a)[lambda u, g=g: u.group == g]]],
           "ratio_hist_edges_log10": EDGES.tolist(),
           "tiers": {t: summarise(tier_rows(d, t)) for t in TIERS}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")
    d.to_csv(OUT_CSV, index=False)

    print("TMDB source:", src, "| snapshot:", res["tmdb_snapshot"])
    print("clean movies:", res["clean_movies"], "| matched to a TMDB row:", res["matched_to_tmdb_row"])
    for t, s in res["tiers"].items():
        print(f"\n{t} (n = {s['n']})")
        for f in ("budget", "gross"):
            print(f"  {f}:", {k: v for k, v in s[f].items() if k != "ratio_hist"})
        print(f"  both TMDB fields present: {s['both_tmdb_fields']} | grossed under budget: Wikipedia "
              f"{s['under_budget_wikipedia']}, TMDB {s['under_budget_tmdb']}, same call on {s['same_under_budget_call']}")
        print(f"  Spearman of log(gross/budget) between sources: {s['spearman_log_multiple']}")
        print("  flips:", s["flips"])
    print(f"\noutside the clean sample ({res['all_movies'] - res['clean_movies']} of {res['all_movies']} films):")
    for r in res["outside_clean"]:
        print("  ", r)
    u = unparsed(a)
    res["no_field_wikitext_check"] = wikitext_field_lines(u[u.group == UNPARSED[2]])
    if LABELS.exists():   # the seeded 30, labelled by hand: counts only
        lab = pd.read_csv(LABELS, dtype=str)
        res["unparsed_sample_labels"] = {"n": int(len(lab)), "labelled_by": sorted(lab.labelled_by.unique()),
                                         "categories": lab.category.value_counts().to_dict()}
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")
    print("\nno parsed Wikipedia figure but both on TMDB, by where it went missing:")
    for r in res["unparsed_with_tmdb"]:
        print("  ", r)
    print("independent check of 'no infobox field':", res["no_field_wikitext_check"])
    print("hand labels of the seeded 30:", res.get("unparsed_sample_labels"))
    print(f"\nwrote {OUT_JSON.relative_to(REPO)} and {OUT_CSV}")

    if "--sample" in sys.argv:   # the largest disagreements and a random draw, to read by hand against the infobox
        show = ["tconst", "title", "year", "tier", "wiki_budget", "tmdb_budget", "wiki_gross", "tmdb_revenue"]
        worst = np.fmax(np.log(d.budget_ratio).abs(), np.log(d.gross_ratio).abs())
        print(d.loc[worst.nlargest(25).index, show].to_string(index=False))
        print(d[(d.tmdb_budget > 0) & (d.tmdb_revenue > 0)].sample(15, random_state=2805)[show].to_string(index=False))
    if "--sample-unparsed" in sys.argv:   # a seeded random 30 of the possible parser misses, to read by hand
        u = unparsed(a)
        u = u[u.group == UNPARSED[3]]
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 80)
        print(u.sample(min(30, len(u)), random_state=2805)[["tconst", "primaryTitle", "y", "missing", "budget_raw",
                                                             "gross_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
