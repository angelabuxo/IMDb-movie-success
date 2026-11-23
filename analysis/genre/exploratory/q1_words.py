"""Q1, inside one genre: which plot words go with Horror films that beat their budget, and which with those that
don't? (Course week 6, comparing groups of documents.) Added after the main results in q1_genre.py.

  Films     usable-money films tagged Horror that have a Wikipedia plot section (wiki/articles*.jsonl, the same plot
            headings as scripts/reviews/plot_embeddings.py).
  Labels    each film's return net of its budget and era: ln(gross / budget) minus the mean of all usable-money films
            in the same budget decile x era bin. Hit = top quarter of that within Horror, flop = bottom quarter, the
            middle half dropped (as Q3 cuts hits and flops).
  Words     lowercase runs of letters (an inner apostrophe kept). Plot summaries are full of character names, so a
            word is dropped when more than half of its uses across all usable-money plots are capitalised in the
            middle of a sentence. A word must appear in at least MIN_DF labelled films.
  Statistic the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and Quinn 2008; the one
            Scattertext calls LogOddsRatioInformativeDirichletPrior, as in Q3's scripts-analysis/words/words.py),
            prior = the word's count over all usable-money plots, reported as a z-score.
  Null      hit/flop labels shuffled 1000 times among the labelled films within budget deciles; words ranked by the
            shuffle-calibrated z, a family-wise check on the largest |z|, and a list-level test of the k-th largest.

What it misses: a Wikipedia plot is written after release, by editors, at whatever length they chose, so the
words describe the story as remembered, not the film as marketed. Counts and words only are written; no plot text
leaves data/raw (the token counts are cached in analysis/genre/private/, which git ignores).
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_words [--perms 1000] [--genre Horror]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from analysis.genre.q1_genre import DATA, SEED, load, prepare
from analysis.genre.exploratory import EXP_OUT

OUT = EXP_OUT / "words"
PRIVATE = Path(__file__).resolve().parent / "private"
SOURCES = ("wiki/articles.jsonl", "wiki/articles_10k.jsonl")
HEADINGS = ("plot", "plot summary", "plot synopsis", "synopsis", "premise", "summary", "story", "storyline", "plot overview")
SECTION = re.compile(r"^==\s*([^=].*?)\s*==\s*$", re.M)
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MIN_DF = 10
TOP = 15
KS = (5, 10, 20)
# post hoc, chosen from the Horror list after seeing it: a film counts as supernatural if its plot has any of these
SUPERNATURAL = ("demon", "demonic", "possessed", "possessing", "paranormal", "haunting", "supernatural", "occult")


def plot_section(wikitext):
    """The first plot-like level-2 section as plain text (same rule as scripts/reviews/plot_embeddings.py)."""
    import mwparserfromhell
    heads = list(SECTION.finditer(wikitext))
    for k, h in enumerate(heads):
        if re.sub(r"[^a-z ]", "", h.group(1).lower()).strip() in HEADINGS:
            end = heads[k + 1].start() if k + 1 < len(heads) else len(wikitext)
            raw = re.sub(r"<ref[^>]*?/>|<ref.*?</ref>", "", wikitext[h.end():end], flags=re.S)
            return re.sub(r"\s+", " ", mwparserfromhell.parse(raw).strip_code(normalize=True, collapse=True)).strip()
    return ""


def count_plots(ids):
    """{imdb_id: Counter of lowercase words}, and corpus counts of mid-sentence capitalised vs lowercase uses."""
    cache = PRIVATE / "plot_counts.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    want, counts, cap, low = set(ids), {}, Counter(), Counter()
    for src in SOURCES:
        with open(DATA / src, encoding="utf8") as fh:
            for line in fh:
                r = json.loads(line)
                hit = [t for t in (r.get("tconst") or []) if t in want and t not in counts]
                if not hit or not r.get("wikitext"):
                    continue
                text = plot_section(r["wikitext"])
                if not text:
                    continue
                c = Counter()
                for sent in re.split(r"(?<=[.!?])\s+", text):
                    toks = WORD.findall(sent)
                    for k, t in enumerate(toks):
                        lt = t.lower()
                        c[lt] += 1
                        if k > 0:
                            (cap if t[0].isupper() else low)[lt] += 1
                for t in hit:
                    counts[t] = c
    PRIVATE.mkdir(exist_ok=True)
    out = (counts, cap, low)
    cache.write_bytes(pickle.dumps(out))
    return out


def log_odds_z(yh, yf, a):
    a0 = a.sum()
    nh, nf = yh.sum(axis=-1, keepdims=True), yf.sum(axis=-1, keepdims=True)
    d = np.log((yh + a) / (nh + a0 - yh - a)) - np.log((yf + a) / (nf + a0 - yf - a))
    return d / np.sqrt(1 / (yh + a) + 1 / (yf + a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--genre", default="Horror")
    ap.add_argument("--membership", choices=("imdb", "tag_genome", "imdb_on_tag_genome"), default="imdb",
                    help="imdb: films tagged with the genre; tag_genome: films whose largest Tag Genome share is the genre; "
                         "imdb_on_tag_genome: films tagged with the genre that have Tag Genome scores")
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    counts, cap, low = count_plots(f["imdb_id"])
    names = {w for w in cap if cap[w] > low[w]}
    # return net of budget decile x era, from all usable-money films
    cell = f["budget_bin"].astype(str) + "|" + f["era_bin"].astype(str)
    f["net"] = f["ln_gross_over_budget"] - f.groupby(cell)["ln_gross_over_budget"].transform("mean")
    member = M[:, keep.index(a.genre)] == 1
    if a.membership != "imdb":
        from analysis.genre.exploratory.q1_shares import shares
        S, has = shares("genre_shares.csv", f, keep)
        member = has & (np.nan_to_num(S).argmax(1) == keep.index(a.genre)) if a.membership == "tag_genome" else member & has
    g = f[member & f["imdb_id"].isin(counts)].reset_index(drop=True)
    q1, q3 = g["net"].quantile([.25, .75])
    lab = g[(g["net"] <= q1) | (g["net"] >= q3)].reset_index(drop=True)
    y = (lab["net"] >= q3).to_numpy().astype(int)
    # vocabulary and the prior
    prior = Counter()
    for t in f["imdb_id"]:
        if t in counts:
            prior.update(counts[t])
    df = Counter(w for t in lab["imdb_id"] for w in counts[t])
    vocab = sorted(w for w, d in df.items() if d >= MIN_DF and w not in names)
    vi = {w: j for j, w in enumerate(vocab)}
    r, c, v = [], [], []
    for i, t in enumerate(lab["imdb_id"]):
        for w, n in counts[t].items():
            if w in vi:
                r.append(i), c.append(vi[w]), v.append(n)
    X = sparse.csr_matrix((v, (r, c)), shape=(len(lab), len(vocab)))
    alpha = np.array([prior[w] for w in vocab], float)

    def z_for(yy):
        return log_odds_z(np.asarray(X[yy == 1].sum(0)).ravel(), np.asarray(X[yy == 0].sum(0)).ravel(), alpha)

    obs = z_for(y)
    rng = np.random.default_rng(SEED + 1000)
    groups = lab["budget_bin"].to_numpy()
    idx = [np.flatnonzero(groups == s) for s in np.unique(groups)]
    nl = np.empty((a.perms, len(vocab)))
    for p in range(a.perms):
        yp = y.copy()
        for ix in idx:
            yp[ix] = y[rng.permutation(ix)]
        nl[p] = z_for(yp)
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    zc = (obs - mu) / sd
    zc_null = (nl - mu) / sd
    fw = float((1 + (np.abs(zc_null).max(1) >= np.abs(zc).max()).sum()) / (1 + a.perms))
    hf = np.asarray((X[y == 1] > 0).sum(0)).ravel()
    ff = np.asarray((X[y == 0] > 0).sum(0)).ravel()
    words = pd.DataFrame({"word": vocab, "calibrated_z": zc.round(2), "monroe_z": obs.round(2),
                          "hit_films": hf, "flop_films": ff,
                          "hit_count": np.asarray(X[y == 1].sum(0)).ravel(), "flop_count": np.asarray(X[y == 0].sum(0)).ravel(),
                          "shuffle_p": ((1 + (np.abs(zc_null) >= np.abs(zc)).sum(0)) / (1 + a.perms)).round(4)})
    top = pd.concat([words.nlargest(TOP, "calibrated_z").assign(side="hit"),
                     words.nsmallest(TOP, "calibrated_z").assign(side="flop")])
    lists = []
    for side, sgn in (("hit", 1), ("flop", -1)):
        o = np.sort(sgn * zc)[::-1]
        n = np.sort(sgn * zc_null, axis=1)[:, ::-1]
        for k in KS:
            lists.append({"side": side, "k": k, "kth_calibrated_z": round(o[k - 1], 2),
                          "p": round(float((1 + (n[:, k - 1] >= o[k - 1]).sum()) / (1 + a.perms)), 4)})
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.genre.lower() + ("" if a.membership == "imdb" else "_" + a.membership)
    words.sort_values("calibrated_z", ascending=False).to_csv(OUT / f"{tag}_words.csv", index=False)
    top.to_csv(OUT / f"{tag}_top.csv", index=False)
    pd.DataFrame(lists).to_csv(OUT / f"{tag}_list_null.csv", index=False)
    summ = {"genre": a.genre, "membership": a.membership, "films_with_plot": len(g), "hits": int(y.sum()), "flops": int(len(y) - y.sum()),
            "vocab": len(vocab), "names_dropped": len(names), "min_df": MIN_DF, "perms": a.perms,
            "familywise_p_max_abs_calibrated_z": round(fw, 4), "max_abs_calibrated_z": round(float(np.abs(zc).max()), 2),
            "median_plot_words_hit": int(np.median([sum(counts[t].values()) for t in lab["imdb_id"][y == 1]])),
            "median_plot_words_flop": int(np.median([sum(counts[t].values()) for t in lab["imdb_id"][y == 0]]))}
    if a.genre == "Horror":
        # post hoc: return net of budget and era for plots with any supernatural word, before 2015 and from 2015,
        # with a label shuffle inside each period (the words were picked from this list, so this isn't independent)
        rows, rng2 = [], np.random.default_rng(SEED + 1400)
        g["supernatural"] = [any(counts[t].get(w, 0) > 0 for w in SUPERNATURAL) for t in g["imdb_id"]]
        for name, d in (("before 2015", g[g["year"] < 2015]), ("2015 and later", g[g["year"] >= 2015]), ("all", g)):
            sup, net = d["supernatural"].to_numpy(), d["net"].to_numpy()
            obs_d = net[sup].mean() - net[~sup].mean()
            nl_d = np.array([(lambda m: net[m].mean() - net[~m].mean())(rng2.permutation(sup)) for _ in range(5000)])
            rows.append({"period": name, "films": len(d), "supernatural_films": int(sup.sum()),
                         "share_supernatural": round(sup.mean(), 3), "net_supernatural": round(net[sup].mean(), 3),
                         "net_other": round(net[~sup].mean(), 3), "difference": round(obs_d, 3),
                         "p": round(float((1 + (np.abs(nl_d) >= abs(obs_d)).sum()) / 5001), 4)})
        pd.DataFrame(rows).to_csv(OUT / f"{tag}_supernatural_by_period.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    print(pd.DataFrame(lists).to_string(index=False))
    print(top[["side", "word", "calibrated_z", "hit_films", "flop_films"]].to_string(index=False))


if __name__ == "__main__":
    main()
