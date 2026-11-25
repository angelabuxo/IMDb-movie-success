"""Do the scripts add anything to facts known before release, and which films have a script at all? (post hoc)

The similarity test is in budget_matched.py (D29). Design fixed before any number was seen:
  A. auc        The classifier, folds and labels of run_tests.py. Baseline "pre" = era, the film's other genres and
                log(budget). "tier" = the old baseline (vote tier, era, other genres), rerun to check it reproduces
                primary.csv. The vector is added to "pre"; the gain has a paired film bootstrap (2000, no refit).
  B. coverage   Films with usable money: hit / middle / flop labels cut on ALL of them, then the share with a
                MovieSum script, per label and vote tier. The hit - flop gap in coverage is compared with 2000
                shuffles of the labels within genre x era (x tier).

Budget is the denominator of the return, so its link to the label is partly mechanical. Films under 10,000 votes are
in none of the tables. Every vectors_w5_s*.npz in W2V_VECTOR_DIR is used, so the seeds are those found there.

Run:  IMDB_DATA_DIR=... python -I analysis/screenplays/word2vec/prerelease.py
Outputs (numbers only): results/prerelease_{auc,coverage,coverage_gap}.csv
"""
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import all_genres as ag  # noqa: E402
import run_tests as rt  # noqa: E402
from final_set import final_films, read_table  # noqa: E402
from genre_cells import label, pairs_for  # noqa: E402
from genre_counts import DATA  # noqa: E402

OUT, B, PERMS, SEED = HERE / "results", 2000, 2000, 2805
TABLE = {r["imdb_id"]: r for r in read_table()}
BUDGET = {i: float(r["budget"]) for i, r in TABLE.items() if r["budget"]}


def write(name, rows):
    with open(OUT / f"prerelease_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    for r in rows:
        print(name, *[f"{v:.3f}" if isinstance(v, float) else v for v in r.values()], flush=True)


def raw_vectors():
    out, ids = [], None
    for p in ag.VECTORS:
        z = np.load(p)
        these = [str(i) for i in z["ids"]]
        assert ids in (None, these)
        ids = these
        out.append({k: z[f"{k}_tfidf_mean"] for k in ag.LAYERS})
    return {i: k for k, i in enumerate(ids)}, out


def part_a():
    pos, vecs = raw_vectors()
    rows = []
    for genre in ("Comedy", "Action", "Drama"):
        films = rt.labelled(genre)
        ids = [f["id"] for f, _ in films]
        idx = [pos[i] for i in ids]
        y = np.array([lb == "hit" for _, lb in films], int)
        tier = rt.baseline_matrix(films, genre)
        nob = tier[:, 1:]
        lb = np.log([BUDGET[i] for i in ids])[:, None]
        pre = np.hstack([nob, lb])
        s_pre = rt.oof(pre, y, "dense")
        for name, X in (("tier+era+genres", tier), ("era+genres", nob), ("era+genres+budget", pre)):
            rows.append(dict(genre=genre, layer="-", model=name, n=len(y), seeds=0,
                             auc=rt.mean_auc(rt.oof(X, y, "dense"), y), gain="", lo="", hi=""))
        for layer in ag.LAYERS:
            V = [v[layer][idx] for v in vecs]
            rows.append(dict(genre=genre, layer=layer, model="vector", n=len(y), seeds=len(V),
                             auc=float(np.mean([rt.mean_auc(rt.oof(X, y, "dense"), y) for X in V])), gain="", lo="", hi=""))
            sc = np.array([rt.oof(np.hstack([X, pre]), y, "dense") for X in V])
            a = float(np.mean([rt.mean_auc(s, y) for s in sc]))
            rng, bs = np.random.default_rng(SEED), []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y))
                bs.append(np.mean([rt.mean_auc(s[:, i], y[i]) for s in sc]) - rt.mean_auc(s_pre[:, i], y[i]))
            lo, hi = map(float, np.percentile(bs, [2.5, 97.5]))
            rows.append(dict(genre=genre, layer=layer, model="era+genres+budget + vector", n=len(y), seeds=len(V),
                             auc=a, gain=a - rt.mean_auc(s_pre, y), lo=lo, hi=hi))
    write("auc", rows)


def part_b():
    with open(DATA / "features" / "movies_features_10k.csv", newline="") as f:
        votes = {r["tconst"]: float(r["votes"]) for r in csv.DictReader(f)}
    films = []
    for i, r in TABLE.items():
        if r["usable_money"] == "True":
            y = int(r["year"])
            films.append(dict(id=i, genres=[g for g in r["genres"].split(",") if g], decade=y // 10 * 10, tier=r["tier"],
                              ln=float(r["ln_gross_over_budget"]), script=r["in_moviesum"] == "True"))
    print("usable money", len(films), "with script", sum(f["script"] for f in films))
    rows, gaps, rng = [], [], np.random.default_rng(SEED)
    for s in (True, False):
        g = [f for f in films if f["script"] == s]
        rows.append(dict(group="all films", tier="all", label="script" if s else "no script", films=len(g), with_script="",
                         coverage="", median_votes=int(np.median([votes[f["id"]] for f in g])),
                         median_budget_musd=float(np.median([BUDGET[f["id"]] for f in g]) / 1e6),
                         mean_ln_return=float(np.mean([f["ln"] for f in g]))))
    lab, _ = label(pairs_for(films, "a"))
    for group in ("all film-genre pairs", "Comedy", "Action", "Drama", "Horror"):
        sel = [(f, g, e, lb) for f, g, e, lb in lab if group.startswith("all") or g == group]
        for tier in ("all", "100k+", "10k-100k"):
            s = [x for x in sel if tier == "all" or (x[0]["tier"] == "100k+") == (tier == "100k+")]
            for lb in ("hit", "middle", "flop"):
                k = [x[0] for x in s if x[3] == lb]
                sc = [f for f in k if f["script"]]
                rows.append(dict(group=group, tier=tier, label=lb, films=len(k), with_script=len(sc),
                                 coverage=len(sc) / len(k), median_votes=int(np.median([votes[f["id"]] for f in k])),
                                 median_budget_musd=float(np.median([BUDGET[f["id"]] for f in k]) / 1e6),
                                 mean_ln_return=float(np.mean([f["ln"] for f in k]))))
            hf = [x for x in s if x[3] != "middle"]
            y = np.array([x[3] == "hit" for x in hf], int)
            c = np.array([x[0]["script"] for x in hf], float)
            st = defaultdict(list)
            for k, x in enumerate(hf):
                st[x[1], x[2]].append(k)
            st = [np.array(v) for v in st.values()]
            gap = lambda q: c[q == 1].mean() - c[q == 0].mean()  # noqa: E731
            null = np.array([gap(ag.shuffle_within(y, st, rng)) for _ in range(PERMS)])
            o = gap(y)
            gaps.append(dict(group=group, tier=tier, hits=int(y.sum()), flops=int(len(y) - y.sum()), gap=float(o),
                             null_lo=float(np.quantile(null, .025)), null_hi=float(np.quantile(null, .975)),
                             p_two_sided=float((1 + (abs(null - null.mean()) >= abs(o - null.mean())).sum()) / (1 + PERMS))))
    write("coverage", rows)
    write("coverage_gap", gaps)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    part_b()
    part_a()
